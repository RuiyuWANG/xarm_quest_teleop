# src/eval/eval_runner.py
from __future__ import annotations

import time
import threading
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import os
import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from pynput import keyboard

from src.io.process_manager import TemporalObsBuffer, ObsPacket, SampleRing
from src.utils.conversion_utils import xyz6g_to_action_abs, pose6_to_xyz6, center_square_crop , se3_interp_xyz_rot6, rot6d_to_R, rotation_angle_between
from src.configs.eval_config import EvalConfig
from src.io.video_recorder import AsyncVideoWriter

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

    def __init__(self, cfg: EvalConfig, robot, net, sample_ring: SampleRing, result_log_dir: Optional[str] = None):
        self.cfg = cfg
        self.robot = robot
        self.net = net
        self.sample_ring = sample_ring

        self.bridge = CvBridge()
        self.obs_buf = TemporalObsBuffer(maxlen=1200)

        self.state = RunState(running=False, continue_requested=(False))  # True if gating

        # internal
        self._stop = False
        self._thread = threading.Thread(target=self._control_loop, daemon=True)
        self._kb_thread = threading.Thread(target=self._keyboard_loop, daemon=True)

        # planning/execution state
        self._last_grip_bin: Optional[int] = None
        self._plan_t_obs: float = 0.0
        self._last_exec_action10: Optional[np.ndarray] = None
        
        # logging
        # --- logging dir ---
        self.run_dir = os.path.join(
            str(self.cfg.result_log_dir),
            str(self.cfg.task_name),
            str(self.cfg.model_name),
            str(self.cfg.seed),
        )
        os.makedirs(self.run_dir, exist_ok=True)
        if self.cfg.record:
            self.video_dir = os.path.join(self.run_dir, "videos")
            os.makedirs(self.video_dir, exist_ok=True)
            
             # --- video ---
            self._video = AsyncVideoWriter(fps=cfg.video_fps)
            self._cur_rollout_video_path: Optional[str] = None
            self._recording_active = False 
            self._cur_rollout_video_tmp_path = None

        self.result_log_path = os.path.join(self.run_dir, "result.txt")

        # --- resume from existing result.txt ---
        self._load_existing_results()

        # --- horizon counters (model inference steps) ---
        self._infer_steps_in_rollout = 0
        self._cur_rollout_status: Optional[str] = None

    # ---------------- lifecycle ----------------
    def start(self):
        if int(self.state.episode_idx) < int(self.cfg.n_rollouts) and self.cfg.record:
            self._start_rollout_video()
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
                self._append_result(self.state.episode_idx, "success")
                if self.cfg.record:
                    self._finish_rollout_video("success")

                self.state.episode_idx += 1
                self.state.reset_requested = True

                rospy.loginfo(f"[EvalRunner] marked SUCCESS ({self.state.success_count}/{max(1,self.state.episode_idx)})")

            elif ch == "f":
                self.state.last_episode_status = "fail"
                self._append_result(self.state.episode_idx, "fail")
                if self.cfg.record:
                    self._finish_rollout_video("fail")

                self.state.episode_idx += 1
                self.state.reset_requested = True

                rospy.loginfo(f"[EvalRunner] marked FAIL ({self.state.success_count}/{max(1,self.state.episode_idx)})")


            elif ch == "q":
                rospy.logwarn("[EvalRunner] quit requested -> homing robot then shutting down")
                try:
                    self.robot.home()
                except Exception as e:
                    rospy.logwarn(f"[EvalRunner] home failed on quit: {e}")
                rospy.signal_shutdown("User quit requested")
                return


        keyboard.Listener(on_press=on_press).start()

        while not rospy.is_shutdown() and not self._stop and not self.state.quit_requested:
            time.sleep(0.2)
            
            
    def _load_existing_results(self):
        self.state.episode_idx = 0
        self.state.success_count = 0
        if not self.result_log_path or (not os.path.exists(self.result_log_path)):
            return
        try:
            with open(self.result_log_path, "r", encoding="utf-8") as f:
                lines = [ln.strip() for ln in f.readlines() if ln.strip()]
            # expected: "i: success" / "i: fail"
            for ln in lines:
                if ":" not in ln:
                    continue
                _, status = ln.split(":", 1)
                status = status.strip().lower()
                if status == "success":
                    self.state.success_count += 1
                if status in ("success", "fail"):
                    self.state.episode_idx += 1
            rospy.logwarn(f"[EvalRunner] resumed: episode_idx={self.state.episode_idx}, success={self.state.success_count}")
        except Exception as e:
            rospy.logwarn(f"[EvalRunner] failed to load results: {e}")


    def _append_result(self, idx: int, status: str):
        os.makedirs(os.path.dirname(self.result_log_path), exist_ok=True)
        with open(self.result_log_path, "a", encoding="utf-8") as f:
            f.write(f"{idx}: {status}\n")


    def _finalize_results_file(self):
        # create a copy named by final success rate
        n = max(1, int(self.state.episode_idx))
        rate = float(self.state.success_count) / float(n)
        name = f"success_rate_{rate:.2f}.txt"
        dst = os.path.join(self.run_dir, name)
        try:
            import shutil
            shutil.copyfile(self.result_log_path, dst)
            rospy.logwarn(f"[EvalRunner] wrote final: {dst}")
        except Exception as e:
            rospy.logwarn(f"[EvalRunner] failed to write final success-rate file: {e}")


    def _start_rollout_video(self):
        # start new video for current rollout index (status unknown yet)
        idx = int(self.state.episode_idx)
        self._cur_rollout_status = None
        self._cur_rollout_video_path = os.path.join(self.video_dir, f"{idx}_pending.mp4")
        self._video.start(self._cur_rollout_video_path)


    def _finish_rollout_video(self, status: str):
        # stop writer then rename pending to final name
        idx = int(self.state.episode_idx)
        self._video.stop()
        if self._cur_rollout_video_path is None:
            return
        final_path = os.path.join(self.video_dir, f"{idx}_{status}.mp4")
        try:
            if os.path.exists(self._cur_rollout_video_path):
                os.replace(self._cur_rollout_video_path, final_path)
        except Exception as e:
            rospy.logwarn(f"[EvalRunner] failed to rename video: {e}")
        self._cur_rollout_video_path = None
        self._cur_rollout_status = status

            
    # ---------------- sanity checks ----------------
    def _in_workspace(self, xyz: np.ndarray) -> bool:
        xyz = np.asarray(xyz, dtype=np.float32).reshape(3,)
        mn = np.asarray(self.cfg.workspace_min_xyz, dtype=np.float32).reshape(3,)
        mx = np.asarray(self.cfg.workspace_max_xyz, dtype=np.float32).reshape(3,)
        return bool(np.all(xyz >= mn) and np.all(xyz <= mx))
    

    def _step_ok(self, a_prev: np.ndarray, a_next: np.ndarray) -> bool:
        """
        a_prev, a_next: (10,) actions [xyz,rot6,grip]
        """
        a_prev = np.asarray(a_prev, dtype=np.float32).reshape(-1)
        a_next = np.asarray(a_next, dtype=np.float32).reshape(-1)

        # workspace check for absolute action
        if not self._in_workspace(a_next[0:3]):
            rospy.logwarn_throttle(1.0, f"[EvalRunner] action out of workspace xyz={a_next[0:3]}")
            return False

        # consecutive delta check
        dp = float(np.linalg.norm(a_next[0:3] - a_prev[0:3], ord=2))
        Rprev = rot6d_to_R(a_prev[3:9]).reshape(3,3)
        Rnext = rot6d_to_R(a_next[3:9]).reshape(3,3)
        dr = rotation_angle_between(Rprev, Rnext)

        if dp > float(self.cfg.max_step_trans):
            rospy.logwarn_throttle(1.0, f"[EvalRunner] step too large dp={dp:.4f} > {self.cfg.max_step_trans}")
            return False
        if dr > float(self.cfg.max_step_rot_rad):
            rospy.logwarn_throttle(1.0, f"[EvalRunner] rot step too large dr={dr:.4f} > {self.cfg.max_step_rot_rad}")
            return False

        return True

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

    # TODO: add depth
    def on_full_rgbd_set(self, t_cam: float, cams_set: Dict[str, Any]):
        """
        Optional full sync: cams_set[cam] has .rgb and .depth
        We still only use rgb cams in cfg.rgb_cams_light for now.
        """
        t_cam = float(t_cam)
        s = self.sample_ring.nearest(t_cam, float(self.cfg.robot_sync.robot_match_window_s))
        if s is None:
            return
        
        rgb_dict: Dict[str, np.ndarray] = {}
        for cam in self.cfg.rgb_cams_full:
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
    
    
    def _execute_chunk(self, chunk_actions: np.ndarray) -> bool:
        """
        chunk_actions: (Te, 10) in [xyz, rot6d, grip] predicted by policy (absolute)
        We:
        1) Build a smoothed action sequence in SE(3), including previous->current bridging
        2) Sanity-check workspace + step sizes
        3) Convert each action to robot pose6+grip and stream servo
        """
        dt = 1.0 / float(self.cfg.control_hz)
        N_interp = self.cfg.interp_steps
        grip_mode = self.cfg.gripper_interp_mode

        acts = np.asarray(chunk_actions, dtype=np.float32)
        if acts.ndim != 2 or acts.shape[1] < 10:
            rospy.logwarn(f"[EvalRunner] bad chunk_actions shape={acts.shape}")
            return False
        if acts.shape[0] == 0:
            return True

        # --- Build smoothed action sequence ---
        exec_actions = []

        first = acts[0, :10].copy()

        # bridge from last executed action to first action of this chunk
        if self._last_exec_action10 is not None:
            bridge = se3_interp_xyz_rot6(self._last_exec_action10, first, n=N_interp, grip_mode=grip_mode)
            exec_actions.append(bridge)

        # interpolate within chunk
        for i in range(acts.shape[0] - 1):
            a0 = acts[i, :10]
            a1 = acts[i + 1, :10]
            seg = se3_interp_xyz_rot6(a0, a1, n=N_interp, grip_mode=grip_mode)
            exec_actions.append(seg)

        # include final point (one step)
        exec_actions.append(acts[-1:, :10])
        exec_actions = np.concatenate(exec_actions, axis=0).astype(np.float32)  # (N,10)

        # --- Sanity checks ---
        # workspace check for all
        for j in range(exec_actions.shape[0]):
            if not self._in_workspace(exec_actions[j, 0:3]):
                rospy.logwarn(f"[EvalRunner] abort: action[{j}] xyz out of workspace: {exec_actions[j,0:3]}")
                return False

        # consecutive step checks (use last_exec if available)
        prev = self._last_exec_action10 if self._last_exec_action10 is not None else exec_actions[0]
        for j in range(exec_actions.shape[0]):
            cur = exec_actions[j]
            if not self._step_ok(prev, cur):
                rospy.logwarn(f"[EvalRunner] abort: step check failed at j={j}")
                return False
            prev = cur

        # --- Stream execution ---
        for j in range(exec_actions.shape[0]):
            if rospy.is_shutdown() or self._stop or self.state.quit_requested:
                return False
            if self.state.reset_requested or (not self.state.running):
                return False

            act10 = exec_actions[j]

            # Convert 10D action to robot command (pose6 mm+rpy + grip pulse logic)
            pose6, grip = self._convert_action_to_robot(act10)  # MAKE SURE this accepts (10,)
            self._send_step_command(pose6, grip)

            # update last executed
            self._last_exec_action10 = act10.copy()

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
        self._last_exec_action10 = None
        self._infer_steps_in_rollout = 0

        if int(self.state.episode_idx) < int(self.cfg.n_rollouts) and self.cfg.record:
            self._start_rollout_video()

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
                rate.sleep()
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
            
            if int(self.state.episode_idx) >= int(self.cfg.n_rollouts):
                rospy.logwarn("[EvalRunner] completed all rollouts -> finalizing and shutting down")
                try:
                    self.robot.home()
                except Exception as e:
                    rospy.logwarn(f"[EvalRunner] home failed at end: {e}")
                self._finalize_results_file()
                rospy.signal_shutdown("Finished all rollouts")
                return

            built = self._build_temporal_obs()
            if built is None:
                rospy.loginfo_throttle(2.0, "[EvalRunner] waiting: temporal obs not ready / stale / insufficient")
                rate.sleep()
                continue

            t_obs, temporal_obs = built
            self._plan_t_obs = float(t_obs)
            
            if self.cfg.record:
                try:
                    # temporal_obs["rgb"][cam] is list[T] of HWC RGB uint8 (in your EvalRunner)
                    # take the latest frame from a preferred camera
                    cam_pref = self.cfg.record_cam
                    cam_keys = list(temporal_obs["rgb"].keys())
                    cam = cam_pref if cam_pref in cam_keys else cam_keys[0]
                    frame = temporal_obs["rgb"][cam][-1]  # HWC RGB uint8
                    self._video.enqueue(frame)
                except Exception:
                    pass

            # rospy.loginfo_throttle(
            #     1.0,
            #     f"[EvalRunner] built obs at t_obs={t_obs:.3f} (now={self._ros_now():.3f})"
            # )

            acts = self.net.infer_action(temporal_obs)
            self._infer_steps_in_rollout += 1

            if int(self._infer_steps_in_rollout) >= int(self.cfg.horizon):
                rospy.logwarn(f"[EvalRunner] rollout horizon reached ({self.cfg.horizon}) -> auto FAIL")
                self._append_result(self.state.episode_idx, "fail")
                if self.cfg.record:
                    self._finish_rollout_video("fail")
                self.state.episode_idx += 1
                self.state.reset_requested = True
                rate.sleep()
                continue

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

