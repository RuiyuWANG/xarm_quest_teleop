import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import rospy
import message_filters

from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
from xarm_msgs.msg import RobotMsg

from teleop_msgs.msg import OVR2ROSInputsStamped

from src.io.quest2 import Quest2Interface
from src.robots.xarm import XArmRobot, XArmState
from src.configs.teleop_config import TeleopConfig
from src.configs.robot_config import GRIPPER_MIN, GRIPPER_MAX

from src.utils.teleop_utils import (
    clamp,
    quat_xyzw_to_rot,
    rot_to_axis_angle,
    wrap_to_pi,
    vec_clamp_norm,
    _smoothstep01,
    LatchedReference,
    AdaptivePoseFilter
)


@dataclass
class SyncedSample:
    stamp_ros: float
    stamp_sync: float
    quest_pose: PoseStamped
    quest_inputs: OVR2ROSInputsStamped
    robot_state_msg: RobotMsg
    robot_state: XArmState
    desired_pose6_mm_rpy: Optional[List[float]]
    cmd_pose6_mm_rpy: Optional[List[float]]
    cmd_gripper: Optional[float]
    cameras: Dict[str, Optional[Image]]


class CameraBuffer:
    def __init__(self, topic: str, keep_s: float):
        self.topic = topic
        self.keep_s = float(keep_s)
        self.buf: List[Image] = []
        self.sub = rospy.Subscriber(topic, Image, self._cb, queue_size=10)

    def _cb(self, msg: Image):
        self.buf.append(msg)
        now = msg.header.stamp.to_sec()
        cutoff = now - self.keep_s
        while self.buf and self.buf[0].header.stamp.to_sec() < cutoff:
            self.buf.pop(0)

    def nearest(self, t: float, window: float) -> Optional[Image]:
        if not self.buf:
            return None
        best = None
        best_dt = 1e9
        for m in self.buf:
            dt = abs(m.header.stamp.to_sec() - t)
            if dt < best_dt:
                best_dt = dt
                best = m
        if best is None or best_dt > window:
            return None
        return best

class QuestXArmTeleopSync:
    """
    Improved Servo_Cartesian teleop:
      - ATS callback only stores latest synced inputs
      - Fixed-rate timer loop (cfg.servo_rate_hz) sends /move_servo_cart
      - Vive-style adaptive filter + spike clamp + EMA bypass
      - Smooth re-engagement after relatch/reset
      - Step-limited servo commands (<10mm each update)
    """

    def __init__(self, cfg: TeleopConfig, quest: Quest2Interface, robot: XArmRobot):
        self.cfg = cfg
        self.quest = quest
        self.robot = robot

        self._hooks: List[Callable[[SyncedSample], None]] = []
        self._last_grip_cmd_time = 0.0
        self._last_grip_pulse: Optional[float] = None

        # reference latch
        self._ref: Optional[LatchedReference] = None
        self._deadman_prev: bool = False

        # latest synced messages (from ATS)
        self._latest_pose: Optional[PoseStamped] = None
        self._latest_inputs: Optional[OVR2ROSInputsStamped] = None
        self._latest_robotmsg: Optional[RobotMsg] = None
        self._latest_sync_stamp: float = 0.0

        # cameras
        self._cams: Dict[str, CameraBuffer] = {
            t: CameraBuffer(t, keep_s=cfg.camera_buffer_seconds) for t in cfg.camera_image_topics
        }

        # adaptive filter on DESIRED pose (mm+rpy)
        self._pose_filter = AdaptivePoseFilter(
            ema_alpha_slow=cfg.pose_ema_alpha_slow,
            ema_bypass_mm=cfg.pose_ema_bypass_mm,
            ema_bypass_rot_rad=cfg.pose_ema_bypass_rot_rad,
            clamp_mm=cfg.pose_clamp_mm,
            clamp_rot_rad=cfg.pose_clamp_rot_rad,
        )

        # smooth re-engagement
        self._reengaging: bool = False
        self._reengage_i: int = 0
        self._last_sent_pose: Optional[np.ndarray] = None  # 6

        # select hand topics
        if cfg.active_hand == "right":
            pose_topic = cfg.right_pose_stamped_topic
            inputs_topic = cfg.right_inputs_stamped_topic
        else:
            pose_topic = cfg.left_pose_stamped_topic
            inputs_topic = cfg.left_inputs_stamped_topic

        # ATS subs
        self.pose_sub = message_filters.Subscriber(pose_topic, PoseStamped)
        self.inputs_sub = message_filters.Subscriber(inputs_topic, OVR2ROSInputsStamped)
        self.robot_sub = message_filters.Subscriber(cfg.robot_state, RobotMsg)

        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.pose_sub, self.inputs_sub, self.robot_sub],
            queue_size=int(cfg.sync_queue_size),
            slop=float(cfg.sync_slop_s),
            allow_headerless=bool(cfg.sync_allow_headerless),
        )
        self.sync.registerCallback(self._ats_cb)

        rospy.loginfo("[QuestXArmTeleopSync] ATS ready (servo_cart + fixed-rate loop)")

        # fixed-rate servo loop
        period = 1.0 / float(cfg.servo_rate_hz)
        self._timer = rospy.Timer(rospy.Duration(period), self._servo_tick, oneshot=False)

    def register_hook(self, fn: Callable[[SyncedSample], None]):
        self._hooks.append(fn)

    # ---------------- input helpers ----------------
    def _deadman(self, inputs_st: OVR2ROSInputsStamped) -> bool:
        if not self.cfg.require_deadman:
            return True
        return bool(getattr(inputs_st.inputs, self.cfg.deadman_field, False))

    def _reset_pressed(self, inputs_st: OVR2ROSInputsStamped) -> bool:
        if not self.cfg.enable_reset:
            return False
        return bool(getattr(inputs_st.inputs, self.cfg.reset_field, False))

    # ---------------- gripper ----------------
    def _compute_gripper_pulse(self, inputs_st: OVR2ROSInputsStamped) -> Optional[float]:
        if not self.cfg.enable_gripper:
            return None

        inp = inputs_st.inputs

        close_u = float(getattr(inp, "press_index", 0.0)) if self.cfg.grip_close_from_index else 0.0
        open_u  = float(getattr(inp, "press_middle", 0.0)) if self.cfg.grip_open_from_middle else 0.0

        close_u = clamp(close_u, 0.0, 1.0)
        open_u  = clamp(open_u, 0.0, 1.0)

        # close amount = press_index - press_middle
        u = clamp(open_u - close_u, -1.0, 1.0)

        # Incremental control around last pulse
        if self._last_grip_pulse is None:
            pulse = (GRIPPER_MAX + GRIPPER_MIN) * 0.5
        else:
            pulse = self._last_grip_pulse

        step = u * (GRIPPER_MAX - GRIPPER_MIN) * float(self.cfg.grip_speed)
        pulse += step

        return float(clamp(pulse, GRIPPER_MIN, GRIPPER_MAX))


    def _maybe_command_gripper(self, pulse: float) -> Optional[float]:
        now = time.time()
        if now - self._last_grip_cmd_time < float(self.cfg.grip_rate_limit_s):
            return None
        if (not self.cfg.grip_continuous) and (self._last_grip_pulse is not None):
            if abs(pulse - self._last_grip_pulse) < float(self.cfg.grip_change_eps):
                return None
        r = self.robot.move_gripper(pulse)
        if r.ok:
            self._last_grip_cmd_time = now
            self._last_grip_pulse = pulse
            return pulse
        return None

    # ---------------- haptics ----------------
    def _haptics(self, deadman: bool, robot_state: XArmState):
        if not self.cfg.enable_haptics:
            return
        hand = self.cfg.active_hand
        if robot_state.err is not None and int(robot_state.err) != 0:
            self.quest.vibrate(hand, self.cfg.err_haptic_freq, self.cfg.err_haptic_amp)
            return
        if deadman and self.cfg.require_deadman:
            self.quest.vibrate(hand, self.cfg.deadman_haptic_freq, self.cfg.deadman_haptic_amp)
        else:
            self.quest.vibrate(hand, 0.0, 0.0)

    # ---------------- cameras ----------------
    def _nearest_cameras(self, t_sync: float) -> Dict[str, Optional[Image]]:
        out: Dict[str, Optional[Image]] = {}
        for topic, buf in self._cams.items():
            out[topic] = buf.nearest(t_sync, window=float(self.cfg.camera_match_window_s))
        return out

    # ---------------- latch + mapping ----------------
    def _latch_reference(self, pose: PoseStamped, robot_state: XArmState) -> bool:
        if robot_state.ee_pose is None or len(robot_state.ee_pose) < 6:
            return False

        p = np.array([pose.pose.position.x, pose.pose.position.y, pose.pose.position.z], dtype=np.float32)
        q = np.array([pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w],
                     dtype=np.float32)
        Rq = quat_xyzw_to_rot(q)

        ref_tcp = np.array(robot_state.ee_pose[:6], dtype=np.float32)  # [mm,mm,mm,r,p,y]
        self._ref = LatchedReference(quest_pos_m=p, quest_rot=Rq, robot_pose6_mm_rpy=ref_tcp)

        # reset filters + re-engage
        self._pose_filter.reset()
        self._reengaging = True
        self._reengage_i = 0

        # rospy.loginfo("[teleop] reference latched (re-engage)")
        return True

    def _compute_desired_pose6(self, pose: PoseStamped) -> Optional[np.ndarray]:
        if self._ref is None:
            return None

        p = np.array([pose.pose.position.x, pose.pose.position.y, pose.pose.position.z], dtype=np.float32)
        q = np.array([pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w],
                     dtype=np.float32)
        Rq = quat_xyzw_to_rot(q)

        # delta in Quest frame
        dp_m = p - self._ref.quest_pos_m
        R_delta = self._ref.quest_rot.T @ Rq
        daa = rot_to_axis_angle(R_delta)  # axis-angle vector (rad)

        # map + scale into robot space
        dp_robot_m = (self.cfg.R_pos_map @ dp_m) * float(self.cfg.pos_scale)
        daa_robot = (self.cfg.R_rot_map @ daa) * float(self.cfg.rot_scale)

        # clamp absolute delta (safety)
        dp_robot_m = vec_clamp_norm(dp_robot_m.astype(np.float32), float(self.cfg.max_delta_pos_m))
        daa_robot = vec_clamp_norm(daa_robot.astype(np.float32), float(self.cfg.max_delta_rot_rad))

        desired = self._ref.robot_pose6_mm_rpy.copy()
        desired[:3] = self._ref.robot_pose6_mm_rpy[:3] + (dp_robot_m * 1000.0)  # m->mm
        if self.cfg.enable_orientation:
            desired[3:6] = wrap_to_pi(self._ref.robot_pose6_mm_rpy[3:6] + daa_robot)
        return desired

    def _step_toward_desired(self, cur6: np.ndarray, desired6: np.ndarray) -> np.ndarray:
        cmd = cur6.copy()

        dp_mm = desired6[:3] - cur6[:3]
        dp_mm = vec_clamp_norm(dp_mm.astype(np.float32), float(self.cfg.servo_max_step_mm))  # MUST < 10mm
        cmd[:3] = cur6[:3] + dp_mm

        if self.cfg.enable_orientation:
            dr = wrap_to_pi(desired6[3:6] - cur6[3:6])
            dr = vec_clamp_norm(dr.astype(np.float32), float(self.cfg.servo_max_step_rot_rad))
            cmd[3:6] = wrap_to_pi(cur6[3:6] + dr)
        else:
            cmd[3:6] = cur6[3:6]

        return cmd

    # ---------------- ATS callback (store latest only) ----------------
    def _ats_cb(self, pose: PoseStamped, inputs_st: OVR2ROSInputsStamped, robot_st: RobotMsg):
        self._latest_pose = pose
        self._latest_inputs = inputs_st
        self._latest_robotmsg = robot_st
        self._latest_sync_stamp = pose.header.stamp.to_sec()

    # ---------------- fixed-rate servo tick ----------------
    def _servo_tick(self, _event):
        if rospy.is_shutdown():
            return

        pose = self._latest_pose
        inputs_st = self._latest_inputs
        robot_st = self._latest_robotmsg

        if pose is None or inputs_st is None or robot_st is None:
            return

        stamp_ros = rospy.Time.now().to_sec()
        stamp_sync = float(self._latest_sync_stamp)

        robot_state = self.robot.get_state()

        deadman = self._deadman(inputs_st)
        self._haptics(deadman, robot_state)

        allow = deadman if self.cfg.require_deadman else True
        if robot_state.err is not None and int(robot_state.err) != 0:
            allow = False

        # edge detection for deadman press
        deadman_pressed = (deadman and not self._deadman_prev)
        deadman_released = ((not deadman) and self._deadman_prev)
        self._deadman_prev = deadman

        desired_pose6 = None
        cmd_pose6 = None
        cmd_gripper = None

        if allow:
            if self._reset_pressed(inputs_st) or deadman_pressed or (self._ref is None):
                self._latch_reference(pose, robot_state)

            if (self._ref is not None) and (robot_state.ee_pose is not None) and (len(robot_state.ee_pose) >= 6):
                cur6 = np.array(robot_state.ee_pose[:6], dtype=np.float32)

                desired_pose6 = self._compute_desired_pose6(pose)
                if desired_pose6 is not None:
                    # ---- Vive-style filtering on desired target ----
                    desired_pose6 = self._pose_filter.apply(
                        desired_pose6,
                        bypass=self._reengaging,  # during re-engage, avoid EMA lag
                    )

                    # ---- smooth re-engagement (blend last_sent -> desired) ----
                    if self._reengaging and (self._last_sent_pose is not None):
                        n = int(self.cfg.reengage_steps)
                        t = self._reengage_i / max(1, n)
                        a = _smoothstep01(t)
                        blended = (1.0 - a) * self._last_sent_pose + a * desired_pose6
                        blended[3:6] = wrap_to_pi(blended[3:6])
                        desired_pose6 = blended
                        self._reengage_i += 1
                        if self._reengage_i >= n:
                            self._reengaging = False

                    # ---- step-limited servo command toward desired ----
                    cmd_pose6 = self._step_toward_desired(cur6, desired_pose6)

                    r = self.robot.move_servo_cart(
                        pose6_mm_rpy=cmd_pose6.tolist(),
                        tool_coord=bool(self.cfg.servo_tool_coord),
                    )
                    if not r.ok:
                        rospy.logwarn_throttle(1.0, f"[teleop] move_servo_cart failed: {r.ret} {r.message}")

                    self._last_sent_pose = cmd_pose6.copy()

            # gripper
            gp = self._compute_gripper_pulse(inputs_st)
            if gp is not None:
                cmd_gripper = self._maybe_command_gripper(gp)

        else:
            # disallowed: optionally clear reference
            if self.cfg.clear_reference_on_deadman_release and deadman_released:
                self._ref = None
            # do not send new servo commands; robot holds last

        cams = self._nearest_cameras(stamp_sync)

        sample = SyncedSample(
            stamp_ros=stamp_ros,
            stamp_sync=stamp_sync,
            quest_pose=pose,
            quest_inputs=inputs_st,
            robot_state_msg=robot_st,
            robot_state=robot_state,
            desired_pose6_mm_rpy=desired_pose6.tolist() if desired_pose6 is not None else None,
            cmd_pose6_mm_rpy=cmd_pose6.tolist() if cmd_pose6 is not None else None,
            cmd_gripper=cmd_gripper,
            cameras=cams,
        )

        for fn in self._hooks:
            try:
                fn(sample)
            except Exception as e:
                rospy.logwarn_throttle(1.0, f"[teleop] hook exception: {e}")
