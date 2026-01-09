import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

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
    ema_vec,
    LatchedReference,
)


@dataclass
class SyncedSample:
    stamp_ros: float
    stamp_sync: float

    quest_pose: PoseStamped
    quest_inputs: OVR2ROSInputsStamped

    robot_state_msg: RobotMsg
    robot_state: XArmState

    # pose servo info
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
    Pose-delta teleop using xArm Servo_Cartesian:
      - sync: PoseStamped (stamped wrapper) + InputsStamped + RobotMsg
      - latch reference on deadman press or reset
      - desired TCP pose = ref_tcp + mapped(delta_quest_pose)
      - command TCP pose each tick via /xarm/move_servo_cart
        BUT ONLY small steps toward desired:
          - max step translation <= 10mm (xArm requirement)
          - max step rotation <= cfg.servo_max_step_rot_rad
    """

    def __init__(self, cfg: TeleopConfig, quest: Quest2Interface, robot: XArmRobot):
        self.cfg = cfg
        self.quest = quest
        self.robot = robot

        self._hooks: List[Callable[[SyncedSample], None]] = []
        self._last_grip_cmd_time = 0.0
        self._last_grip_pulse: Optional[float] = None

        # reference
        self._ref: Optional[LatchedReference] = None

        # filtered deltas (Quest -> robot)
        self._filt_dp_m = np.zeros(3, dtype=np.float32)
        self._filt_daa = np.zeros(3, dtype=np.float32)  # axis-angle vector (rad)

        # cameras
        self._cams: Dict[str, CameraBuffer] = {
            t: CameraBuffer(t, keep_s=cfg.camera_buffer_seconds) for t in cfg.camera_image_topics
        }

        # select hand topics
        if cfg.active_hand == "right":
            pose_topic = cfg.right_pose_stamped_topic
            inputs_topic = cfg.right_inputs_stamped_topic
        else:
            pose_topic = cfg.left_pose_stamped_topic
            inputs_topic = cfg.left_inputs_stamped_topic

        # message_filters subscribers
        self.pose_sub = message_filters.Subscriber(pose_topic, PoseStamped)
        self.inputs_sub = message_filters.Subscriber(inputs_topic, OVR2ROSInputsStamped)
        self.robot_sub = message_filters.Subscriber(cfg.robot_state, RobotMsg)

        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.pose_sub, self.inputs_sub, self.robot_sub],
            queue_size=int(cfg.sync_queue_size),
            slop=float(cfg.sync_slop_s),
            allow_headerless=bool(cfg.sync_allow_headerless),
        )
        self.sync.registerCallback(self._synced_cb)

        rospy.loginfo("[QuestXArmTeleopSyncServoCart] ATS ready (servo_cart)")

        # optional: configure servo_cart mode here
        if cfg.servo_auto_configure:
            self._configure_servo_cart_mode()

    def _configure_servo_cart_mode(self):
        """
        Per xArm docs for Servo_Cartesian:
          motion_ctrl 8 1
          set_mode 1
          set_state 0
        """
        rospy.loginfo("[teleop] configuring servo_cart mode (motion_ctrl 8 1, mode=1, state=0)")
        r = self.robot.enable_servo_cart()
        if not r.ok:
            rospy.logwarn(f"[teleop] enable_servo_cart failed: {r.ret} {r.message}")

    def register_hook(self, fn: Callable[[SyncedSample], None]):
        self._hooks.append(fn)

    # ---------------- gripper ----------------
    def _compute_gripper_pulse(self, inputs_st: OVR2ROSInputsStamped) -> Optional[float]:
        if not self.cfg.enable_gripper:
            return None
        inp = inputs_st.inputs
        close_u = float(getattr(inp, "press_index", 0.0)) if self.cfg.grip_close_from_index else 0.0
        open_u = float(getattr(inp, "press_middle", 0.0)) if self.cfg.grip_open_from_middle else 0.0
        close_u = clamp(close_u, 0.0, 1.0)
        open_u = clamp(open_u, 0.0, 1.0)
        u = clamp(close_u - open_u, 0.0, 1.0)
        pulse = (1.0 - u) * (GRIPPER_MAX - GRIPPER_MIN) + GRIPPER_MIN
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

    # ---------------- pose delta helpers ----------------
    def _deadman(self, inputs_st: OVR2ROSInputsStamped) -> bool:
        if not self.cfg.require_deadman:
            return True
        return bool(getattr(inputs_st.inputs, self.cfg.deadman_field, False))

    def _reset_pressed(self, inputs_st: OVR2ROSInputsStamped) -> bool:
        if not self.cfg.enable_reset:
            return False
        return bool(getattr(inputs_st.inputs, self.cfg.reset_field, False))

    def _latch_reference(self, pose: PoseStamped, robot_state: XArmState) -> bool:
        if robot_state.ee_pose is None or len(robot_state.ee_pose) < 6:
            return False

        p = np.array([pose.pose.position.x, pose.pose.position.y, pose.pose.position.z], dtype=np.float32)
        q = np.array(
            [pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w],
            dtype=np.float32,
        )
        Rq = quat_xyzw_to_rot(q)

        ref_tcp = np.array(robot_state.ee_pose[:6], dtype=np.float32)  # [mm,mm,mm,r,p,y] (rad)
        self._ref = LatchedReference(quest_pos_m=p, quest_rot=Rq, robot_pose6_mm_rpy=ref_tcp)

        self._filt_dp_m[:] = 0.0
        self._filt_daa[:] = 0.0

        rospy.loginfo("[teleop] reference latched")
        return True

    def _compute_desired_pose6(self, pose: PoseStamped) -> Optional[np.ndarray]:
        """
        desired TCP pose (absolute) in xArm representation:
          [mm,mm,mm, roll rad, pitch rad, yaw rad]
        """
        if self._ref is None:
            return None

        p = np.array([pose.pose.position.x, pose.pose.position.y, pose.pose.position.z], dtype=np.float32)
        q = np.array(
            [pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w],
            dtype=np.float32,
        )
        Rq = quat_xyzw_to_rot(q)

        # Quest delta
        dp_m = p - self._ref.quest_pos_m
        R_delta = self._ref.quest_rot.T @ Rq
        daa = rot_to_axis_angle(R_delta)  # axis-angle vector (rad)

        # Map + scale into robot space
        dp_robot_m = (self.cfg.R_pos_map @ dp_m) * float(self.cfg.pos_scale)
        daa_robot = (self.cfg.R_rot_map @ daa) * float(self.cfg.rot_scale)

        # Filter deltas
        self._filt_dp_m = ema_vec(self._filt_dp_m, dp_robot_m, self.cfg.delta_filter_alpha)
        self._filt_daa = ema_vec(self._filt_daa, daa_robot, self.cfg.delta_filter_alpha)

        # Clamp deltas (safety)
        self._filt_dp_m = vec_clamp_norm(self._filt_dp_m, float(self.cfg.max_delta_pos_m))
        self._filt_daa = vec_clamp_norm(self._filt_daa, float(self.cfg.max_delta_rot_rad))

        desired = self._ref.robot_pose6_mm_rpy.copy()
        desired[:3] = self._ref.robot_pose6_mm_rpy[:3] + (self._filt_dp_m * 1000.0)  # m -> mm

        if self.cfg.enable_orientation:
            desired[3:6] = wrap_to_pi(self._ref.robot_pose6_mm_rpy[3:6] + self._filt_daa)

        return desired

    def _step_toward_desired(self, cur6: np.ndarray, desired6: np.ndarray) -> np.ndarray:
        """
        Servo-cart requires small step from current TCP.
        - translation step <= cfg.servo_max_step_mm (MUST < 10mm)
        - rotation step <= cfg.servo_max_step_rot_rad
        """
        cmd = cur6.copy()

        # position step
        dp_mm = desired6[:3] - cur6[:3]
        dp_mm = vec_clamp_norm(dp_mm.astype(np.float32), float(self.cfg.servo_max_step_mm))
        cmd[:3] = cur6[:3] + dp_mm

        # rotation step
        if self.cfg.enable_orientation:
            dr = wrap_to_pi(desired6[3:6] - cur6[3:6])
            dr = vec_clamp_norm(dr.astype(np.float32), float(self.cfg.servo_max_step_rot_rad))
            cmd[3:6] = wrap_to_pi(cur6[3:6] + dr)
        else:
            cmd[3:6] = cur6[3:6]

        return cmd

    # ---------------- callback ----------------
    def _synced_cb(self, pose: PoseStamped, inputs_st: OVR2ROSInputsStamped, robot_st: RobotMsg):
        rospy.loginfo_throttle(1.0, "[teleop] servo_cart sync callback")

        stamp_ros = rospy.Time.now().to_sec()
        stamp_sync = pose.header.stamp.to_sec()

        deadman = self._deadman(inputs_st)
        robot_state = self.robot.get_state()
        self._haptics(deadman, robot_state)

        allow = deadman if self.cfg.require_deadman else True
        if robot_state.err is not None and int(robot_state.err) != 0:
            allow = False

        desired_pose6 = None
        cmd_pose6 = None
        cmd_gripper = None

        rospy.loginfo_throttle(1.0, f"[Teleop] allow={allow} deadman={deadman} err={robot_state.err}")

        if allow:
            if self._reset_pressed(inputs_st):
                self._latch_reference(pose, robot_state)

            if self._ref is None:
                self._latch_reference(pose, robot_state)
            else:
                if robot_state.ee_pose is not None and len(robot_state.ee_pose) >= 6:
                    cur6 = np.array(robot_state.ee_pose[:6], dtype=np.float32)

                    desired_pose6 = self._compute_desired_pose6(pose)
                    if desired_pose6 is not None:
                        cmd_pose6 = self._step_toward_desired(cur6, desired_pose6)

                        # Base-coordinate servo_cart: mvvelo/mvacc/mvtime/mvradii not effective => pass zeros.
                        # Tool-coordinate servo_cart (firmware >=1.5.0): set mvtime=1 and pose is interpreted as tool-relative.
                        r = self.robot.move_servo_cart(
                            pose6_mm_rpy=cmd_pose6.tolist(),
                            tool_coord=bool(self.cfg.servo_tool_coord),
                        )
                        if not r.ok:
                            rospy.logwarn_throttle(1.0, f"[teleop] move_servo_cart failed: {r.ret} {r.message}")

            gp = self._compute_gripper_pulse(inputs_st)
            if gp is not None:
                cmd_gripper = self._maybe_command_gripper(gp)

        else:
            if self.cfg.clear_reference_on_deadman_release:
                self._ref = None

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
