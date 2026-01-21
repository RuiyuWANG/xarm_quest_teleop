# src/eval/eval_runner.py
from __future__ import annotations

import time
import threading
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from pynput import keyboard

from src.io.process_manager import TemporalObsBuffer, ObsPacket, SampleRing
from src.utils.conversion_utils import xyz6g_to_action_abs, pose6_to_xyz6, center_square_crop
from src.configs.eval_config import EvalConfig


@dataclass
class RunState:
    running: bool = False              # policy actively controlling
    reset_requested: bool = False
    quit_requested: bool = False
    continue_requested: bool = False   # optional: require "c" to start executing
    last_episode_status: Optional[str] = None
    success_count: int = 0
    episode_idx: int = 0


class EvalRunner:
    """
    Merged evaluator + chunk runner + keyboard control.

    Data flow:
      camera sync callback -> on_light_rgb_set / on_full_rgbd_set
        -> nearest robot sample in SampleRing
        -> build ObsPacket (rgb + lowdim)
        -> push to TemporalObsBuffer

      control loop thread:
        - waits for running=True
        - builds temporal To obs from buffer
        - infer -> actions [Ta, D] (absolute xyz+rot6d(+grip))
        - choose slice with delay-comp (start offset)
        - execute Te steps sequentially
          - stream servo_cart while waiting for "finished" or timeout
        - replan each chunk from newest obs (diffusion-friendly MPC)
    """

    def __init__(self, cfg: EvalConfig, robot, net, sample_ring: SampleRing, result_log_path: Optional[str] = None):
        self.cfg = cfg
        self.robot = robot
        self.net = net
        self.sample_ring = sample_ring

        self.bridge = CvBridge()
        self.obs_buf = TemporalObsBuffer(maxlen=1200)

        self.state = RunState(running=False, continue_requested=(False))  # True if gating
        
        # optional logging
        self.result_log_path = result_log_path

        # internal
        self._stop = False
        self._thread = threading.Thread(target=self._control_loop, daemon=True)
        self._kb_thread = threading.Thread(target=self._keyboard_loop, daemon=True)

        # planning/execution state
        self._last_grip_bin: Optional[int] = None
        self._plan_t_obs: float = 0.0
        
    # ---------------- lifecycle ----------------
    def start(self):
        self._kb_thread.start()
        self._thread.start()

    def stop(self):
        self._stop = True

    # ---------------- keyboard ----------------
    def _keyboard_loop(self):
        rospy.loginfo("[EvalRunner] Keyboard: 'c' start/continue, 'p' pause, 'r' reset, 's' success, 'f' fail, 'q' quit")

        def on_press(key):
            try:
                ch = key.char
            except Exception:
                return

            if ch == "c":
                # toggle running or allow continue
                self.state.running = True
                self.state.continue_requested = True
                rospy.loginfo("[EvalRunner] continue/start requested -> running=True")

            elif ch == "p":
                self.state.running = False
                rospy.loginfo("[EvalRunner] paused -> running=False")

            elif ch == "r":
                self.state.reset_requested = True
                rospy.loginfo("[EvalRunner] reset requested")

            elif ch == "s":
                self.state.last_episode_status = "success"
                self.state.success_count += 1
                self._record_episode_result()
                self.state.reset_requested = True
                rospy.loginfo(f"[EvalRunner] marked SUCCESS (rate={self.state.success_count}/{max(1,self.state.episode_idx)})")

            elif ch == "f":
                self.state.last_episode_status = "fail"
                self._record_episode_result()
                self.state.reset_requested = True
                rospy.loginfo(f"[EvalRunner] marked FAIL (rate={self.state.success_count}/{max(1,self.state.episode_idx)})")

            elif ch == "q":
                self.state.quit_requested = True
                self.state.running = False
                rospy.loginfo("[EvalRunner] quit requested")

        keyboard.Listener(on_press=on_press).start()

        while not rospy.is_shutdown() and not self._stop and not self.state.quit_requested:
            time.sleep(0.2)

    def _record_episode_result(self):
        self.state.episode_idx += 1
        if self.result_log_path is None:
            return
        try:
            import os
            os.makedirs(os.path.dirname(self.result_log_path), exist_ok=True)
            with open(self.result_log_path, "a", encoding="utf-8") as f:
                f.write(f"{self.state.episode_idx-1}: {self.state.last_episode_status}\n")
        except Exception as e:
            rospy.logwarn(f"[EvalRunner] failed to write result log: {e}")

    # ---------------- evaluator (sync callbacks) ----------------
    def _imgmsg_to_rgb(self, msg: Image) -> Optional[np.ndarray]:
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            if img is None:
                return None
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img.astype(np.uint8)
        except Exception:
            return None

    def _build_lowdim(self, s: Any) -> Optional[Dict[str, np.ndarray]]:
        rs = getattr(s, "robot_state", None)
        if rs is None:
            return None
        ee = getattr(rs, "ee_pose", None)
        gr = getattr(rs, "gripper_qpos", None)
        if ee is None or len(ee) < 6 or gr is None:
            return None

        return {
            "ee_pose6": np.asarray(ee[:6], dtype=np.float32),
            "gripper_state": np.asarray([float(gr)], dtype=np.float32),
        }

    def on_light_rgb_set(self, t_cam: float, imgs: Dict[str, Image]):
        """
        imgs: dict cam_name -> sensor_msgs/Image
        """
        rospy.loginfo_throttle(2.0, "[EvalRunner] on_light_rgb_set called")
        t_cam = float(t_cam)
        s = self.sample_ring.nearest(t_cam, float(self.cfg.robot_sync.robot_match_window_s))
        if s is None:
            return
        
        # require all cams in cfg.rgb_cams_light
        rgb_dict: Dict[str, np.ndarray] = {}
        for cam in self.cfg.rgb_cams_light:
            msg = imgs.get(cam, None)
            if msg is None:
                return
            im = self._imgmsg_to_rgb(msg)
            if im is None:
                return
            rgb_dict[cam] = im

        low = self._build_lowdim(s)
        if low is None:
            return

        pkt = ObsPacket(
            t_obs=t_cam,
            obs={"rgb": rgb_dict, "low_dim": low},
            allow=True,
        )
        self.obs_buf.push(pkt)

    def on_full_rgbd_set(self, t_cam: float, cams_set: Dict[str, Any]):
        """
        Optional full sync: cams_set[cam] has .rgb and .depth
        We still only use rgb cams in cfg.rgb_cams_light for now.
        """
        rospy.loginfo_throttle(2.0, "[EvalRunner] on_full_rgbd_set called")
        t_cam = float(t_cam)
        s = self.sample_ring.nearest(t_cam, float(self.cfg.robot_sync.robot_match_window_s))
        if s is None:
            return
        
        rgb_dict: Dict[str, np.ndarray] = {}
        for cam in self.cfg.rgb_cams_light:
            cm = cams_set.get(cam, None)
            if cm is None or getattr(cm, "rgb", None) is None:
                return
            im = self._imgmsg_to_rgb(cm.rgb)
            if im is None:
                return
            rgb_dict[cam] = im

        low = self._build_lowdim(s)
        if low is None:
            return

        pkt = ObsPacket(
            t_obs=t_cam,
            obs={"rgb": rgb_dict, "low_dim": low},
            allow=True,
        )
        self.obs_buf.push(pkt)

    # ---------------- control loop (chunk runner) ----------------
    def _ros_now(self) -> float:
        return rospy.Time.now().to_sec()

    def _build_temporal_obs(self) -> Optional[Tuple[float, Dict[str, Any]]]:
        To = int(self.cfg.obs_horizon)
        pkts = self.obs_buf.get_horizon(To)
        if pkts is None:
            rospy.loginfo_throttle(2.0, f"[EvalRunner] obs_buf not enough frames for To={To}")
            return None

        latest = pkts[-1]
        if not bool(latest.allow):
            rospy.loginfo_throttle(2.0, "[EvalRunner] latest.allow=False")
            return None

        age = self._ros_now() - float(latest.t_obs)
        if age > float(self.cfg.obs_stale_s):
            rospy.loginfo_throttle(2.0, f"[EvalRunner] latest obs stale: age={age:.3f} > {self.cfg.obs_stale_s}")
            return None
        
        # To = int(self.cfg.obs_horizon)
        # pkts = self.obs_buf.get_horizon(To)
        # if pkts is None:
        #     return None

        # latest = pkts[-1]
        # if not bool(latest.allow):
        #     return None
        # if (self._ros_now() - float(latest.t_obs)) > float(self.cfg.obs_stale_s):
        #     return None

        temporal_obs = {
            "rgb": {k: [p.obs["rgb"][k] for p in pkts] for k in latest.obs["rgb"].keys()},
            "low_dim": {k: [p.obs["low_dim"][k] for p in pkts] for k in latest.obs["low_dim"].keys()},
        }
        return float(latest.t_obs), temporal_obs

    # HARDCODE
    def _compute_exec_slice(self) -> Tuple[int, int]:
        Ta = int(self.cfg.pred_horizon)
        Te = int(self.cfg.exec_horizon)

        start = 1
        # if bool(self.cfg.use_delay_comp):
        #     delay_s = max(0.0, self._ros_now() - float(self._plan_t_obs))
        #     offset = int(round(delay_s / float(self.cfg.dt_ctrl)))
        #     start = min(max(0, offset), max(0, Ta - 1))

        end = min(start + Te, Ta)
        return start, end

    def _pose_error(self, cur6: np.ndarray, tgt6: np.ndarray) -> Tuple[float, float]:
        dp = float(np.linalg.norm(cur6[:3] - tgt6[:3], ord=2))
        dr = float(np.linalg.norm(cur6[3:6] - tgt6[3:6], ord=2))
        return dp, dr

    def _is_step_done(self, tgt6: np.ndarray) -> bool:
        st = self.robot.get_state()
        if st is None or getattr(st, "ee_pose", None) is None:
            return False
        cur = np.asarray(st.ee_pose[:6], dtype=np.float32)
        dp, dr = self._pose_error(cur, tgt6)
        return (dp <= float(self.cfg.pos_tol_mm)) and (dr <= float(self.cfg.rot_tol_rad))

    def _convert_action_to_robot(self, act: np.ndarray) -> Tuple[np.ndarray, Optional[float]]:
        """
        act: [x,y,z, rot6d(6), grip?]
        xyz unit per cfg.xyz_unit.
        returns pose6_mm_rpy + optional gripper scalar
        """
        act = np.asarray(act, dtype=np.float32).reshape(1, -1)
        if act.shape[-1] < 9:
            raise ValueError(f"Expected >=9 dims (xyz+rot6d), got {act.shape[0]}")
        
        pose6, grip = xyz6g_to_action_abs(act)
        
        pose6 = np.asarray(pose6, dtype=np.float32).flatten()
        grip = float(grip.flatten()[0]) if grip is not None else None
        return pose6.astype(np.float32), grip

    def _send_step_command(self, pose6: np.ndarray, grip: Optional[np.ndarray]):
        self.robot.move_servo_cart(
            pose6_mm_rpy=pose6.tolist(),
            tool_coord=bool(self.cfg.servo_tool_coord),
        )

        if grip is None:
            return

        if self.cfg.gripper_binary:
            gb = 1 if grip > float(self.cfg.gripper_deadband) else 0
            if self._last_grip_bin is None or gb != self._last_grip_bin:
                if gb == 1:
                    self.robot.move_gripper(float(self.cfg.gripper_open_pulse))
                else:
                    self.robot.move_gripper(float(self.cfg.gripper_close_pulse))
                self._last_grip_bin = gb
                
        else:
            self.robot.move_gripper(grip)

    def _execute_step_until_done(self, pose6: np.ndarray, grip: Optional[float]) -> bool:
        """
        Stream servo command until reached or timeout.
        """
        t0 = self._ros_now()
        dt = 1.0 / float(self.cfg.control_hz)

        while (self._ros_now() - t0) < float(self.cfg.step_timeout_s):
            if rospy.is_shutdown() or self._stop or self.state.quit_requested:
                return False
            if self.state.reset_requested or (not self.state.running):
                return False

            self._send_step_command(pose6, grip)

            if self._is_step_done(pose6):
                return True

            time.sleep(dt)

        return True

    # def _execute_chunk(self, chunk_actions: np.ndarray) -> bool:
    #     t0 = self._ros_now()
    #     for act in chunk_actions:
    #         if (self._ros_now() - t0) > float(self.cfg.chunk_timeout_s):
    #             return False
    #         if rospy.is_shutdown() or self._stop or self.state.quit_requested:
    #             return False
    #         if self.state.reset_requested or (not self.state.running):
    #             return False

    #         pose6, grip = self._convert_action_to_robot(act)
    #         ok = self._execute_step_until_done(pose6, grip)
    #         if not ok:
    #             return False
    #     return True
    
    # TODO: interpolate in SE3
    def _execute_chunk(self, chunk_actions: np.ndarray) -> bool:
        dt = 1.0 / float(self.cfg.control_hz)

        poses = []
        grips = []
        for act in chunk_actions:
            pose6, grip = self._convert_action_to_robot(act)
            poses.append(pose6)
            grips.append(grip)

        N_interp = int(getattr(self.cfg, "interp_steps", 10))

        for i in range(len(poses) - 1):
            p0, p1 = poses[i], poses[i + 1]
            g1 = grips[i + 1]

            for a in np.linspace(0.0, 1.0, N_interp, endpoint=False):
                if rospy.is_shutdown() or self._stop or self.state.quit_requested:
                    return False
                if self.state.reset_requested or (not self.state.running):
                    return False

                p = (1 - a) * p0 + a * p1  # linear in xyz + rpy (good enough if deltas small)
                self._send_step_command(p, g1)
                time.sleep(dt)

        self._send_step_command(poses[-1], grips[-1])
        time.sleep(dt)

        return True


    def _do_reset(self):
        """
        Reset behavior: stop runner, clear buffers, optionally home robot.
        """
        rospy.logwarn("[EvalRunner] RESET: clearing obs buffer + internal state")
        self.obs_buf.clear()
        self._last_grip_bin = None
        self.state.running = False
        self.state.continue_requested = False
        self.state.reset_requested = False

        try:
            self.robot.home()
        except Exception as e:
            rospy.logwarn(f"[EvalRunner] home_robot failed: {e}")

    def _control_loop(self):
        rate = rospy.Rate(float(self.cfg.control_hz))

        while not rospy.is_shutdown() and not self._stop:
            # heartbeat so you know the thread is alive
            rospy.loginfo_throttle(2.0, "[EvalRunner] control_loop alive")

            if self.state.quit_requested:
                rospy.logwarn("[EvalRunner] quit flag set -> stopping control loop")
                return

            if self.state.reset_requested:
                rospy.logwarn("[EvalRunner] reset_requested=True -> resetting")
                self._do_reset()
                rate.sleep()
                continue

            if not self.state.running:
                rospy.loginfo_throttle(2.0, "[EvalRunner] waiting: state.running=False (press 'c')")
                rate.sleep()
                continue

            if self.state.continue_requested is False:
                rospy.loginfo_throttle(2.0, "[EvalRunner] waiting: continue_requested=False (press 'c')")
                rate.sleep()
                continue

            built = self._build_temporal_obs()
            if built is None:
                rospy.loginfo_throttle(2.0, "[EvalRunner] waiting: temporal obs not ready / stale / insufficient")
                rate.sleep()
                continue

            t_obs, temporal_obs = built
            self._plan_t_obs = float(t_obs)

            rospy.loginfo_throttle(
                1.0,
                f"[EvalRunner] built obs at t_obs={t_obs:.3f} (now={self._ros_now():.3f})"
            )

            acts = self.net.infer_action(temporal_obs)
            rospy.loginfo_throttle(1.0, "[EvalRunner] infer_action called")
            if acts is None:
                rospy.logwarn_throttle(2.0, "[EvalRunner] infer_action returned None")
                rate.sleep()
                continue

            acts = np.asarray(acts, dtype=np.float32)
            if acts.ndim == 3:
                acts = acts[0]
            if acts.ndim == 1:
                acts = acts[None, :]

            s_idx, e_idx = self._compute_exec_slice()
            chunk = acts[s_idx:e_idx]

            rospy.loginfo(f"[EvalRunner] executing chunk: idx [{s_idx}:{e_idx}] shape={chunk.shape}")
            ok = self._execute_chunk(chunk)
            rospy.loginfo(f"[EvalRunner] chunk done ok={ok}")

            rate.sleep()

