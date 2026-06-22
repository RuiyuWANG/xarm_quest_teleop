# xarm_quest_teleop/eval/eval_runner.py
from __future__ import annotations

import json
import time
import threading
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import os
from xarm_quest_teleop.ros import compat as rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from pynput import keyboard

from xarm_quest_teleop.eval.action_executor import PolicyActionExecutor
from xarm_quest_teleop.eval.live_viz import LiveVizRenderer
from xarm_quest_teleop.io.process_manager import TemporalObsBuffer, ObsPacket, SampleRing
from xarm_quest_teleop.configs.eval_config import EvalConfig
from xarm_quest_teleop.io.video_recorder import AsyncVideoWriter
from xarm_quest_teleop.policy.seeker_preprocessing import preprocess_real_rgb_image


@dataclass
class RunState:
    running: bool = False              # policy actively controlling
    reset_requested: bool = False
    quit_requested: bool = False
    continue_requested: bool = False   # optional: require "c" to start executing
    last_episode_status: Optional[str] = None
    success_count: int = 0
    fail_count: int = 0
    episode_idx: int = 0


class EvalRunner:
    """
    Merged evaluator + chunk runner + keyboard control.
    """

    def __init__(self, cfg: EvalConfig, robot, net, sample_ring: SampleRing, result_log_dir: Optional[str] = None):
        self.cfg = cfg
        self.robot = robot
        self.net = net
        self.sample_ring = sample_ring
        if result_log_dir is not None:
            self.cfg.result_log_dir = result_log_dir

        self.bridge = CvBridge()
        self.obs_buf = TemporalObsBuffer(maxlen=100)

        self.state = RunState(running=False, continue_requested=(False))  # True if gating

        self.action_executor = PolicyActionExecutor(cfg=self.cfg, robot=self.robot)
        self.action_executor.bind_runtime_state(self.state, lambda: self._stop)
        self.live_viz_renderer = LiveVizRenderer(cfg=self.cfg)

        # internal
        self._stop = False
        self._thread = threading.Thread(target=self._control_loop, daemon=True)
        self._kb_thread = threading.Thread(target=self._keyboard_loop, daemon=True)

        # planning/execution state
        self._plan_t_obs: float = 0.0
        self._live_viz_available = True

        # logging
        # --- logging dir ---
        result_root = self._resolve_repo_path(self.cfg.result_log_dir)
        self.run_dir = os.path.join(
            result_root,
            str(self.cfg.task_name),
            str(self.cfg.model_name),
            str(self.cfg.eval_name),
        )
        os.makedirs(self.run_dir, exist_ok=True)
        self.state_trace_dir: Optional[str] = None
        self._cur_state_trace_path: Optional[str] = None
        if self.cfg.record:
            self.video_dir = os.path.join(self.run_dir, "videos")
            self.attention_video_dir = os.path.join(self.run_dir, "visualization_videos")
            self.state_trace_dir = os.path.join(self.run_dir, "state_traces")
            os.makedirs(self.video_dir, exist_ok=True)
            os.makedirs(self.attention_video_dir, exist_ok=True)
            os.makedirs(self.state_trace_dir, exist_ok=True)

            # --- video (fixed-rate recording between 'c' and success/fail) ---
            fps = int(getattr(self.cfg, "record_fps", getattr(self.cfg, "video_fps", 10)))
            self._video = AsyncVideoWriter(fps=fps)
            self._attention_video = AsyncVideoWriter(fps=fps)
            self._cur_rollout_video_path: Optional[str] = None
            self._cur_attention_video_path: Optional[str] = None
            self._recording_active = False
            self._attention_recording_active = False
            self._record_thread: Optional[threading.Thread] = None
            self._record_lock = threading.Lock()
            self._latest_record_frame: Optional[np.ndarray] = None
            self._latest_attention_frame: Optional[np.ndarray] = None

        self.result_log_path = os.path.join(self.run_dir, "result.txt")

        # --- resume from existing result.txt ---
        self._load_existing_results()

        # --- horizon counters (model inference steps) ---
        self._infer_steps_in_rollout = 0
        self._cur_rollout_status: Optional[str] = None

    @staticmethod
    def _repo_root() -> str:
        return os.getcwd()

    @classmethod
    def _resolve_repo_path(cls, path: str) -> str:
        expanded = os.path.expanduser(str(path))
        if os.path.isabs(expanded):
            return expanded
        return os.path.join(cls._repo_root(), expanded)

    @staticmethod
    def _jsonable(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {str(k): EvalRunner._jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [EvalRunner._jsonable(v) for v in value]
        return value

    # ---------------- lifecycle ----------------
    def start(self):
        self._kb_thread.start()
        self._thread.start()

    def stop(self):
        self._stop = True

    # ---------------- video recording helpers ----------------
    def _update_latest_record_frame(self, rgb_dict: Dict[str, np.ndarray]):
        if not self.cfg.record:
            return
        try:
            cam_keys = list(rgb_dict.keys())
            if not cam_keys:
                return
            cam = "d435i_front" if "d435i_front" in cam_keys else cam_keys[0]
            frame = preprocess_real_rgb_image(rgb_dict[cam], camera_name=cam)
            with self._record_lock:
                self._latest_record_frame = frame.copy()
        except Exception:
            pass

    def _render_inference_viz(self, temporal_obs: Dict[str, Any], acts: np.ndarray) -> None:
        live_viz = bool(getattr(self.cfg, "live_viz", False))
        if not live_viz and not self.cfg.record:
            return
        s_idx, _ = self.action_executor.compute_exec_slice()
        focus_records = list(getattr(self.net, "last_visual_focus_records", []))

        if live_viz:
            overlay = self.live_viz_renderer.make_frame(
                temporal_obs=temporal_obs,
                acts=acts,
                current_exec_start_idx=s_idx,
                focus_records=focus_records,
            )
            if overlay is not None:
                self._show_live_viz(overlay)

        if self.cfg.record:
            attention_overlay = self.live_viz_renderer.make_method_viz_frame(
                temporal_obs=temporal_obs,
                focus_records=focus_records,
            )
            if attention_overlay is not None:
                with self._record_lock:
                    self._latest_attention_frame = attention_overlay.copy()
                self._start_attention_video_for_rollout()

    def _show_live_viz(self, overlay: np.ndarray) -> None:
        if not bool(getattr(self.cfg, "live_viz", False)):
            return
        if not self._live_viz_available:
            return
        try:
            cv2.imshow("xArm Quest Teleop Live Viz", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
        except Exception as exc:
            self._live_viz_available = False
            rospy.logwarn(
                "[EvalRunner] live visualization disabled after display error: "
                f"{exc}"
            )

    def _record_loop(self):
        fps = float(getattr(self.cfg, "record_fps", getattr(self.cfg, "video_fps", 10)))
        dt = 1.0 / max(1.0, fps)

        while (not rospy.is_shutdown()) and (not self._stop) and self._recording_active:
            t0 = time.time()

            frame = None
            attention_frame = None
            with self._record_lock:
                if self._latest_record_frame is not None:
                    frame = self._latest_record_frame.copy()
                if self._latest_attention_frame is not None:
                    attention_frame = self._latest_attention_frame.copy()

            if frame is not None:
                try:
                    self._video.enqueue(frame)
                except Exception:
                    pass
            if attention_frame is not None and self._attention_recording_active:
                try:
                    self._attention_video.enqueue(attention_frame)
                except Exception:
                    pass

            elapsed = time.time() - t0
            to_sleep = dt - elapsed
            if to_sleep > 0:
                time.sleep(to_sleep)

    def _start_recording_for_rollout(self):
        if not self.cfg.record:
            return
        if self._recording_active:
            return
        if int(self.state.episode_idx) >= int(self.cfg.n_rollouts):
            return

        self._start_rollout_video()
        self._start_state_trace_for_rollout()
        self._recording_active = True
        self._record_thread = threading.Thread(target=self._record_loop, daemon=True)
        self._record_thread.start()
        rospy.loginfo("[EvalRunner] recording started")

    def _stop_recording_thread(self):
        if not self.cfg.record:
            return
        if not self._recording_active:
            return
        self._recording_active = False
        if self._record_thread is not None:
            self._record_thread.join(timeout=1.0)
            self._record_thread = None

    def _abort_and_delete_video(self):
        if not self.cfg.record:
            return

        self._stop_recording_thread()
        try:
            self._video.stop()
        except Exception:
            pass
        try:
            self._attention_video.stop()
        except Exception:
            pass

        if self._cur_rollout_video_path is not None:
            try:
                if os.path.exists(self._cur_rollout_video_path):
                    os.remove(self._cur_rollout_video_path)
                    rospy.logwarn(f"[EvalRunner] deleted pending video: {self._cur_rollout_video_path}")
            except Exception as e:
                rospy.logwarn(f"[EvalRunner] failed to delete pending video: {e}")
        if self._cur_attention_video_path is not None:
            try:
                if os.path.exists(self._cur_attention_video_path):
                    os.remove(self._cur_attention_video_path)
                    rospy.logwarn(f"[EvalRunner] deleted pending visualization video: {self._cur_attention_video_path}")
            except Exception as e:
                rospy.logwarn(f"[EvalRunner] failed to delete pending visualization video: {e}")
        if self._cur_state_trace_path is not None:
            try:
                if os.path.exists(self._cur_state_trace_path):
                    os.remove(self._cur_state_trace_path)
                    rospy.logwarn(f"[EvalRunner] deleted pending state trace: {self._cur_state_trace_path}")
            except Exception as e:
                rospy.logwarn(f"[EvalRunner] failed to delete pending state trace: {e}")

        self._cur_rollout_video_path = None
        self._cur_attention_video_path = None
        self._cur_state_trace_path = None
        self._attention_recording_active = False
        with self._record_lock:
            self._latest_attention_frame = None
        self._cur_rollout_status = None

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

                # start fixed-rate recording window for this rollout
                if self.cfg.record:
                    self._start_recording_for_rollout()

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
                self.state.fail_count += 1
                self._append_result(self.state.episode_idx, "fail")

                if self.cfg.record:
                    self._finish_rollout_video("fail")

                self.state.episode_idx += 1
                self.state.reset_requested = True

                rospy.loginfo(f"[EvalRunner] marked FAIL ({self.state.success_count}/{max(1,self.state.episode_idx)})")

            elif ch == "q":
                rospy.logwarn("[EvalRunner] quit requested -> delete pending video then shutting down")
                if self.cfg.record:
                    self._abort_and_delete_video()
                if not bool(getattr(self.cfg, "debug_no_actuate", False)):
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
        self.state.fail_count = 0
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
                if status == "fail":
                    self.state.fail_count += 1
                if status in ("success", "fail"):
                    self.state.episode_idx += 1
            rospy.logwarn(f"[EvalRunner] resumed: episode_idx={self.state.episode_idx}, success={self.state.success_count}, fail={self.state.fail_count}")
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
        self._cur_rollout_video_path = os.path.join(self.video_dir, f"{idx}_pending.gif")
        self._video.start(self._cur_rollout_video_path)

    def _start_state_trace_for_rollout(self):
        if not self.cfg.record or self.state_trace_dir is None:
            return
        if self._cur_state_trace_path is not None:
            return
        idx = int(self.state.episode_idx)
        self._cur_state_trace_path = os.path.join(self.state_trace_dir, f"{idx}_pending.jsonl")
        header = {
            "event": "start",
            "episode_idx": idx,
            "task_name": str(self.cfg.task_name),
            "model_name": str(self.cfg.model_name),
            "eval_name": str(self.cfg.eval_name),
            "obs_horizon": int(self.cfg.obs_horizon),
            "created_wall_time": float(time.time()),
        }
        try:
            with open(self._cur_state_trace_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(header) + "\n")
        except Exception as e:
            rospy.logwarn(f"[EvalRunner] failed to start state trace: {e}")
            self._cur_state_trace_path = None

    def _append_lowdim_state_trace(self, t_obs: float, temporal_obs: Dict[str, Any]):
        if not self.cfg.record or not self._recording_active:
            return
        if self._cur_state_trace_path is None:
            self._start_state_trace_for_rollout()
        if self._cur_state_trace_path is None:
            return
        low_dim = temporal_obs.get("low_dim", {})
        payload = {
            "event": "inference_lowdim",
            "episode_idx": int(self.state.episode_idx),
            "infer_step": int(self._infer_steps_in_rollout),
            "t_obs": float(t_obs),
            "wall_time": float(time.time()),
            "low_dim": self._jsonable(low_dim),
            "latest": self._jsonable(
                {
                    key: values[-1]
                    for key, values in low_dim.items()
                    if isinstance(values, list) and values
                }
            ),
        }
        try:
            with open(self._cur_state_trace_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload) + "\n")
        except Exception as e:
            rospy.logwarn_throttle(2.0, f"[EvalRunner] failed to append state trace: {e}")

    def _start_attention_video_for_rollout(self):
        if not self.cfg.record:
            return
        if self._attention_recording_active:
            return
        if not self._recording_active:
            return
        if int(self.state.episode_idx) >= int(self.cfg.n_rollouts):
            return

        idx = int(self.state.episode_idx)
        self._cur_attention_video_path = os.path.join(self.attention_video_dir, f"{idx}_pending.gif")
        self._attention_video.start(self._cur_attention_video_path)
        self._attention_recording_active = True

    def _finish_rollout_video(self, status: str):
        # stop writer then rename pending to final name (include success/fail counts)
        idx = int(self.state.episode_idx)

        # stop recorder thread first
        self._stop_recording_thread()

        # stop writer
        self._video.stop()
        if self._attention_recording_active:
            self._attention_video.stop()
            self._attention_recording_active = False

        if self._cur_rollout_video_path is None:
            self._finish_attention_video(idx, status)
            self._finish_state_trace(idx, status)
            return

        final_path = os.path.join(self.video_dir, f"{idx}_{status}.gif")
        try:
            if os.path.exists(self._cur_rollout_video_path):
                os.replace(self._cur_rollout_video_path, final_path)
        except Exception as e:
            rospy.logwarn(f"[EvalRunner] failed to rename video: {e}")

        self._cur_rollout_video_path = None
        self._finish_attention_video(idx, status)
        self._finish_state_trace(idx, status)
        self._cur_rollout_status = status

    def _finish_attention_video(self, idx: int, status: str):
        if self._cur_attention_video_path is None:
            return

        final_path = os.path.join(self.attention_video_dir, f"{idx}_{status}.gif")
        try:
            if os.path.exists(self._cur_attention_video_path):
                os.replace(self._cur_attention_video_path, final_path)
        except Exception as e:
            rospy.logwarn(f"[EvalRunner] failed to rename visualization video: {e}")

        self._cur_attention_video_path = None
        with self._record_lock:
            self._latest_attention_frame = None

    def _finish_state_trace(self, idx: int, status: str):
        if self._cur_state_trace_path is None:
            return

        final_path = os.path.join(self.state_trace_dir, f"{idx}_{status}.jsonl")
        try:
            with open(self._cur_state_trace_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"event": "finish", "status": str(status)}) + "\n")
            if os.path.exists(self._cur_state_trace_path):
                os.replace(self._cur_state_trace_path, final_path)
        except Exception as e:
            rospy.logwarn(f"[EvalRunner] failed to finish state trace: {e}")

        self._cur_state_trace_path = None

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

    def on_rgb_set(self, t_cam: float, imgs: Dict[str, Image]):
        """
        imgs: dict cam_name -> sensor_msgs/Image
        """
        t_cam = float(t_cam)
        s = self.sample_ring.nearest(t_cam, float(self.cfg.robot_sync.robot_match_window_s))
        if s is None:
            return

        # require all configured RGB cameras
        rgb_dict: Dict[str, np.ndarray] = {}
        for cam in self.cfg.rgb_cams:
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

        # video: keep latest frame cache regardless of obs horizon usage
        if self.cfg.record:
            self._update_latest_record_frame(rgb_dict)

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

    def _do_reset(self):
        """
        Reset behavior: stop runner, clear buffers, optionally home robot.
        """
        rospy.logwarn("[EvalRunner] RESET: clearing obs buffer + internal state")

        if self.cfg.record:
            # delete any pending video if reset/abort
            self._abort_and_delete_video()

        self.obs_buf.clear()
        self.action_executor.reset()
        self.state.running = False
        self.state.continue_requested = False
        self.state.reset_requested = False
        self._infer_steps_in_rollout = 0

        if not bool(getattr(self.cfg, "debug_no_actuate", False)):
            try:
                self.robot.home()
            except Exception as e:
                rospy.logwarn(f"[EvalRunner] home_robot failed: {e}")

    def _control_loop(self):
        rate = rospy.Rate(float(self.cfg.control_hz))

        while not rospy.is_shutdown() and not self._stop:
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
                if not bool(getattr(self.cfg, "debug_no_actuate", False)):
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

            acts = self.net.infer_action(temporal_obs)
            self._infer_steps_in_rollout += 1
            self._append_lowdim_state_trace(t_obs, temporal_obs)
            self._render_inference_viz(temporal_obs, acts)

            if bool(getattr(self.cfg, "debug_no_actuate", False)):
                rospy.loginfo_throttle(
                    1.0,
                    "[EvalRunner] debug_no_actuate=True; skipping robot command execution",
                )
                rate.sleep()
                continue

            if int(self._infer_steps_in_rollout) >= int(self.cfg.horizon):
                rospy.logwarn(f"[EvalRunner] rollout horizon reached ({self.cfg.horizon}) -> auto FAIL")
                self._append_result(self.state.episode_idx, "fail")
                if self.cfg.record:
                    # auto-fail should be treated like fail: save video
                    self.state.fail_count += 1
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

            s_idx, e_idx = self.action_executor.compute_exec_slice()
            chunk = acts[s_idx:e_idx]

            rospy.loginfo(f"[EvalRunner] executing chunk: idx [{s_idx}:{e_idx}] shape={chunk.shape}")
            ok = self.action_executor.execute_chunk(chunk)
            rospy.loginfo(f"[EvalRunner] chunk done ok={ok}")
            if not ok:
                rospy.logwarn("[EvalRunner] safety abort during chunk execution; pausing policy")
                self.state.running = False
                self.state.continue_requested = False

            rate.sleep()
