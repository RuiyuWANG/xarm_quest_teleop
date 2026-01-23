# src/data_collection/colloter.py
from __future__ import annotations

import os
import time
import json
import shutil
import queue
import threading
from typing import Any, Dict, Optional, List, Tuple

import numpy as np
import rospy
import cv2
from cv_bridge import CvBridge
from pynput import keyboard

from src.utils.config_utils import DatasetInfo, next_demo_id
from src.configs.collector_config import CollectorConfig
from src.teleop.quest_xarm_teleop_sync import SyncedSample


def _ensure(p: str):
    os.makedirs(p, exist_ok=True)


def _is_finite_list(x, n_min: int) -> bool:
    if not isinstance(x, list) or len(x) < n_min:
        return False
    a = np.asarray(x, dtype=np.float64)
    return bool(np.all(np.isfinite(a)))


def _is_finite_vec6(x) -> bool:
    if not isinstance(x, list) or len(x) < 6:
        return False
    a = np.asarray(x[:6], dtype=np.float64)
    return bool(np.all(np.isfinite(a))) and a.shape[0] == 6


def _is_finite_scalar(x) -> bool:
    try:
        return bool(np.isfinite(float(x)))
    except Exception:
        return False


class TeleopDataCollector:
    """
    Two-stream collector (optional full + optional light) under same task name:

      <root>/<task_name>/
        task_meta.json
        all_sensors/
          episode_0001/
            <cam>/rgb/*.png
            <cam>/depth_rect/*.npy
            lowdim.npz
        light/
          episode_0001/
            <cam>/rgb/*.png
            lowdim.npz

    Streams enabled by config:
      - cfg.enable_full_sync  -> all_sensors stream
      - cfg.enable_light_sync -> light stream
    """

    def __init__(self, ds: DatasetInfo, cfg: CollectorConfig, robot):
        self.ds = ds
        self.cfg = cfg
        self.robot = robot
        self.bridge = CvBridge()

        # depth: always .npy uint16 for speed (your code already does that)
        self.save_depth_npy = True

        # per-stream enable
        self.enable_full = cfg.enable_full_sync
        self.enable_light = cfg.enable_light_sync
        assert self.enable_full or self.enable_light, "Collector: at least one stream must be enabled"

        self.data_root = ds.root_dir
        self.task_name = ds.task
        self.task_dir = os.path.join(self.data_root, self.task_name)
        _ensure(self.task_dir)

        self.dir_all = os.path.join(self.task_dir, "all_sensors")
        self.dir_light = os.path.join(self.task_dir, "light")
        if self.enable_full:
            _ensure(self.dir_all)
        if self.enable_light:
            _ensure(self.dir_light)

        # shared episode indexing (use all_sensors if enabled, else light)
        self.episode_prefix = "episode_"
        base_for_idx = self.dir_all if self.enable_full else self.dir_light
        existing_next = next_demo_id(base_for_idx, self.episode_prefix)
        self.current_demo_id = max(int(existing_next), int(ds.demo_id_start))
        self.demo_id_end = int(ds.demo_id_start) + int(ds.num_demos) - 1

        self.demo_in_progress = False
        self.delete_requested = False
        self.save_requested = False
        self.quit_requested = False

        self._ep_all: Optional[str] = None
        self._ep_light: Optional[str] = None

        self._frame_all = 0
        self._frame_light = 0
        self._lowdim_all: List[Dict[str, Any]] = []
        self._lowdim_light: List[Dict[str, Any]] = []

        # separate queues to maximize throughput / avoid head-of-line blocking
        qmax = int(cfg.max_queue)
        self._q_all: "queue.Queue[SyncedSample]" = queue.Queue(maxsize=qmax)
        self._q_light: "queue.Queue[SyncedSample]" = queue.Queue(maxsize=qmax)

        # worker threads (only start what we need)
        self._worker_all = None
        self._worker_light = None
        if self.enable_full:
            self._worker_all = threading.Thread(target=self._worker_loop_all, daemon=True)
            self._worker_all.start()
        if self.enable_light:
            self._worker_light = threading.Thread(target=self._worker_loop_light, daemon=True)
            self._worker_light.start()

        self._write_task_meta_once()
        self.keyboard_listener()

        rospy.loginfo(
            f"[Collector] task_dir={self.task_dir} enable_full={self.enable_full} enable_light={self.enable_light}"
        )
        rospy.loginfo(f"[Collector] episode range: {self.current_demo_id} .. {self.demo_id_end}")

        if self.current_demo_id > self.demo_id_end:
            rospy.loginfo("[Collector] All episodes already collected. Exiting.")
            rospy.signal_shutdown("done")

    # ---------------- meta ----------------
    def _write_task_meta_once(self):
        task_meta_path = os.path.join(self.task_dir, "task_meta.json")
        if os.path.exists(task_meta_path):
            return

        # Full dataset+calibration JSON (exactly like config)
        cfg_json = self.ds.as_dict()

        meta = {
            "config": cfg_json,  # <- everything from the input json
            "collection": {
                "created_wall": time.time(),
                "created_local": time.strftime("%Y-%m-%d %H:%M:%S"),
                "host": os.uname().nodename if hasattr(os, "uname") else None,
                "operator": self.ds.operator,
                "num_demos_planned": int(self.ds.num_demos),
                "demo_id_start": int(self.ds.demo_id_start),
                "demo_prefix": self.ds.demo_prefix,
            },
            "io": {
                "rgb_format": "png_bgr8",
                "depth_format": "npy_uint16",   # your choice (fast)
                "depth_units": "raw_uint16",    # and per-camera depth_scale lives in config.calibration
            },
            "software": {
                "ros_namespace": rospy.get_namespace(),
                "ros_time_wall": time.time(),
            },
        }

        with open(task_meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    # ---------------- keyboard ----------------
    def keyboard_listener(self):
        def on_press(key):
            try:
                if key.char == self.cfg.start_key:
                    if not self.demo_in_progress:
                        rospy.loginfo(f"[Collector] Starting episode {self.current_demo_id}...")
                        self.start_demo()
                elif key.char == getattr(self.cfg, "save_key", "s"):
                    rospy.loginfo("[Collector] Save requested.")
                    self.save_requested = True
                elif key.char == self.cfg.delete_key:
                    rospy.loginfo("[Collector] Delete requested.")
                    self.delete_requested = True
                elif key.char == self.cfg.quit_key:
                    rospy.loginfo("[Collector] Quit requested.")
                    self.quit_requested = True
            except AttributeError:
                pass

        keyboard.Listener(on_press=on_press).start()

    # ---------------- enqueue ----------------
    def enqueue_all(self, sample: SyncedSample):
        if not self.enable_full:
            return
        if not self.demo_in_progress:
            return
        if self.save_requested or self.quit_requested or self.delete_requested:
            return
        try:
            self._q_all.put_nowait(sample)
        except queue.Full:
            # drop to protect control
            pass

    def enqueue_light(self, sample: SyncedSample):
        if not self.enable_light:
            return
        if not self.demo_in_progress:
            return
        if self.save_requested or self.quit_requested or self.delete_requested:
            return
        try:
            self._q_light.put_nowait(sample)
        except queue.Full:
            pass

    # ---------------- workers ----------------
    def _worker_loop_all(self):
        while not rospy.is_shutdown():
            if self._handle_flags():
                continue

            try:
                s = self._q_all.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                self._on_sample_all(s)
            except Exception as e:
                rospy.logwarn_throttle(1.0, f"[Collector] all_sensors worker exception: {e}")
            finally:
                try:
                    self._q_all.task_done()
                except Exception:
                    pass

    def _worker_loop_light(self):
        while not rospy.is_shutdown():
            if self._handle_flags():
                continue

            try:
                s = self._q_light.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                self._on_sample_light(s)
            except Exception as e:
                rospy.logwarn_throttle(1.0, f"[Collector] light worker exception: {e}")
            finally:
                try:
                    self._q_light.task_done()
                except Exception:
                    pass

    def _handle_flags(self) -> bool:
        """
        Handle save/delete/quit even when queues are empty.
        Called from both worker threads.
        """
        if self.delete_requested:
            self.delete_requested = False
            self.delete_demo()
            return True

        if self.save_requested:
            self.save_requested = False
            self._drain_enabled(max_drain_s=1.0)
            self.finish_demo()
            return True

        if self.quit_requested:
            self.quit_requested = False
            self._drain_enabled(max_drain_s=1.0)
            if self.demo_in_progress:
                self.finish_demo()
            rospy.signal_shutdown("quit requested")
            return True

        return False

    def _drain_enabled(self, max_drain_s: float = 1.0):
        t0 = time.time()
        while (time.time() - t0) < float(max_drain_s) and not rospy.is_shutdown():
            did = False

            if self.enable_full:
                try:
                    s = self._q_all.get_nowait()
                    did = True
                    try:
                        self._on_sample_all(s)
                    finally:
                        self._q_all.task_done()
                except queue.Empty:
                    pass

            if self.enable_light:
                try:
                    s = self._q_light.get_nowait()
                    did = True
                    try:
                        self._on_sample_light(s)
                    finally:
                        self._q_light.task_done()
                except queue.Empty:
                    pass

            if not did:
                break

    # ---------------- episode lifecycle ----------------
    def start_demo(self):
        if self.current_demo_id > self.demo_id_end:
            rospy.loginfo("[Collector] Target episode count reached. Exiting.")
            rospy.signal_shutdown("done")
            return

        ep_name = f"{self.episode_prefix}{self.current_demo_id:04d}"

        self._ep_all = os.path.join(self.dir_all, ep_name) if self.enable_full else None
        self._ep_light = os.path.join(self.dir_light, ep_name) if self.enable_light else None

        for p in [self._ep_all, self._ep_light]:
            if p is None:
                continue
            if os.path.exists(p):
                shutil.rmtree(p)
            _ensure(p)

        self._frame_all = 0
        self._frame_light = 0
        self._lowdim_all.clear()
        self._lowdim_light.clear()

        self.demo_in_progress = True
        self.delete_requested = False
        self.save_requested = False
        self.quit_requested = False

    def delete_demo(self):
        for p in [self._ep_all, self._ep_light]:
            if p and os.path.isdir(p):
                rospy.logwarn(f"[Collector] Deleting {p}")
                shutil.rmtree(p, ignore_errors=True)

        self.demo_in_progress = False
        self._frame_all = 0
        self._frame_light = 0
        self._lowdim_all.clear()
        self._lowdim_light.clear()
        # keep current_demo_id unchanged so user can redo
        
        self.robot.home()
        rospy.loginfo(f"[Xarm] Reset robot to home position.")

    # ---------------- validation ----------------
    def _validate_lowdim_common(self, sample: SyncedSample) -> Optional[Dict[str, Any]]:
        if not bool(getattr(sample, "allow_control", False)):
            return None

        t = getattr(sample, "stamp_sync", None)
        if t is None or not _is_finite_scalar(t):
            return None
        t = float(t)

        rs = getattr(sample, "robot_state", None)
        if rs is None:
            return None

        ee_pose = getattr(rs, "ee_pose", None)
        joints = getattr(rs, "joint_angles", None)
        gripper = getattr(rs, "gripper_qpos", None)

        if ee_pose is None or not _is_finite_vec6(ee_pose):
            return None
        if joints is None or not _is_finite_list(joints, n_min=1):
            return None
        if gripper is None or not _is_finite_scalar(gripper):
            return None

        desired = getattr(sample, "desired_pose6_mm_rpy", None)
        if desired is None or not isinstance(desired, list) or len(desired) != 6 or not _is_finite_vec6(desired):
            return None

        cmd_grip = getattr(sample, "cmd_gripper", None)
        if cmd_grip is None or not _is_finite_scalar(cmd_grip):
            return None

        return {
            "timestamp": t,
            "joint_states": [float(x) for x in joints],
            "ee_pose6": [float(x) for x in ee_pose[:6]],
            "gripper_state": float(gripper),
            "pose_targets6": [float(x) for x in desired[:6]],
            "pose_targets_gripper": float(cmd_grip),
            "deadman_released": bool(getattr(sample, "deadman_released", False)),
        }

    def _validate_cams_all(self, cams: Any) -> bool:
        if not isinstance(cams, dict):
            return False
        for _, m in cams.items():
            if getattr(m, "rgb", None) is None or getattr(m, "depth", None) is None:
                return False
        return True

    def _validate_cams_light(self, cams: Any) -> bool:
        if not isinstance(cams, dict) or len(cams) != 2:
            return False
        for _, m in cams.items():
            if m is None:
                return False
            if hasattr(m, "rgb") and getattr(m, "rgb", None) is None:
                return False
        return True

    # ---------------- IO helpers ----------------
    def _write_rgb(self, out_path: str, bgr_img: np.ndarray):
        cv2.imwrite(out_path, bgr_img)

    # ---------------- per-stream saving ----------------
    def _ensure_cam_dirs_all(self, cam_name: str) -> Tuple[str, str]:
        assert self._ep_all is not None
        cam_dir = os.path.join(self._ep_all, cam_name)
        rgb_dir = os.path.join(cam_dir, "rgb")
        depth_dir = os.path.join(cam_dir, "depth_rect")
        _ensure(rgb_dir)
        _ensure(depth_dir)
        return rgb_dir, depth_dir

    def _ensure_cam_dirs_light(self, cam_name: str) -> str:
        assert self._ep_light is not None
        cam_dir = os.path.join(self._ep_light, cam_name)
        rgb_dir = os.path.join(cam_dir, "rgb")
        _ensure(rgb_dir)
        return rgb_dir

    def _on_sample_all(self, sample: SyncedSample):
        if not self.enable_full:
            return
        if not self.demo_in_progress or self._ep_all is None:
            rospy.logwarn("[Collector] Episode all is {}, skipping sample.".format(self._ep_all))
            return
        print("aaaaaaaaaaaaaa")
        lowdim = self._validate_lowdim_common(sample)
        if lowdim is None:
            return

        cams = getattr(sample, "cameras", None)
        print("cams", cams)
        if not self._validate_cams_all(cams):
            return
        print("bbbbbbbbbbbbbbb")
        idx = self._frame_all
        for cam_name, cam_msgs in cams.items():
            rgb_dir, depth_dir = self._ensure_cam_dirs_all(cam_name)

            # RGB
            if self.cfg.save_rgb_png:
                img = self.bridge.imgmsg_to_cv2(cam_msgs.rgb, desired_encoding="bgr8")
                self._write_rgb(
                    os.path.join(rgb_dir, f"{idx:06d}.png"),
                    img,
                )
            print("ddddddddddddddd")
            # Depth (always .npy for speed)
            if self.cfg.save_depth_npy:
                depth = self.bridge.imgmsg_to_cv2(cam_msgs.depth, desired_encoding="passthrough")
                np.save(os.path.join(depth_dir, f"{idx:06d}.npy"), depth.astype(np.uint16), allow_pickle=False)

        print("ccccccccccccccc")
        self._lowdim_all.append(lowdim)
        self._frame_all += 1
        rospy.loginfo_throttle(1.0, f"[Collector] all_sensors saving frame {self._frame_all}")


    def _on_sample_light(self, sample: SyncedSample):
        if not self.enable_light:
            return
        if not self.demo_in_progress or self._ep_light is None:
            return

        lowdim = self._validate_lowdim_common(sample)
        if lowdim is None:
            return

        cams = getattr(sample, "cameras", None)
        if not self._validate_cams_light(cams):
            return

        idx = self._frame_light
        for cam_name, m in cams.items():
            rgb_dir = self._ensure_cam_dirs_light(cam_name)
            rgb_msg = m.rgb if hasattr(m, "rgb") else m
            img = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
            self._write_rgb(
                os.path.join(rgb_dir, f"{idx:06d}.png"),
                img,
            )

        self._lowdim_light.append(lowdim)
        self._frame_light += 1

    # ---------------- finish ----------------
    def _check_integrity_stream(
        self,
        ep_dir: str,
        frame_n: int,
        lowdim: List[Dict[str, Any]],
        expect_depth: bool,
        rgb_ext: str,
        depth_ext: str,
    ) -> bool:
        if frame_n <= 0:
            rospy.logerr(f"[Collector] integrity fail ({os.path.basename(ep_dir)}): no frames")
            return False
        if len(lowdim) != frame_n:
            rospy.logerr(
                f"[Collector] integrity fail ({os.path.basename(ep_dir)}): lowdim={len(lowdim)} frames={frame_n}"
            )
            return False

        cam_names = [d for d in os.listdir(ep_dir) if os.path.isdir(os.path.join(ep_dir, d))]
        if not cam_names:
            rospy.logerr(f"[Collector] integrity fail ({os.path.basename(ep_dir)}): no camera folders")
            return False

        for cam_name in cam_names:
            cam_dir = os.path.join(ep_dir, cam_name)
            rgb_dir = os.path.join(cam_dir, "rgb")
            rgb_n = len([f for f in os.listdir(rgb_dir) if f.endswith(rgb_ext)]) if os.path.isdir(rgb_dir) else 0
            if rgb_n != frame_n:
                rospy.logerr(
                    f"[Collector] integrity fail ({os.path.basename(ep_dir)}): {cam_name} rgb={rgb_n} frames={frame_n}"
                )
                return False

            if expect_depth:
                dep_dir = os.path.join(cam_dir, "depth_rect")
                dep_n = len([f for f in os.listdir(dep_dir) if f.endswith(depth_ext)]) if os.path.isdir(dep_dir) else 0
                if dep_n != frame_n:
                    rospy.logerr(
                        f"[Collector] integrity fail ({os.path.basename(ep_dir)}): {cam_name} depth={dep_n} frames={frame_n}"
                    )
                    return False

        return True

    def _write_lowdim(self, out_path: str, lowdim: List[Dict[str, Any]]):
        joints_list: List[List[float]] = [s["joint_states"] for s in lowdim]
        jdim = max((len(j) for j in joints_list), default=0)

        n = len(lowdim)
        t = np.empty((n,), dtype=np.float64)
        ee = np.empty((n, 6), dtype=np.float32)
        grip = np.empty((n,), dtype=np.float32)
        tgt = np.empty((n, 6), dtype=np.float32)
        tgt_grip = np.empty((n,), dtype=np.float32)
        dmr = np.empty((n,), dtype=np.bool_)
        joints = np.empty((n, jdim), dtype=np.float32)
        joints.fill(np.nan)

        for i, s in enumerate(lowdim):
            t[i] = float(s["timestamp"])
            ee[i, :] = np.asarray(s["ee_pose6"], dtype=np.float32)
            grip[i] = float(s["gripper_state"])
            tgt[i, :] = np.asarray(s["pose_targets6"], dtype=np.float32)
            tgt_grip[i] = float(s["pose_targets_gripper"])
            dmr[i] = bool(s["deadman_released"])
            j = np.asarray(s["joint_states"], dtype=np.float32).reshape(-1)
            joints[i, : j.shape[0]] = j

        np.savez_compressed(
            out_path,
            timestamp=t,
            joint_states=joints,
            ee_pose6=ee,
            gripper_state=grip,
            pose_targets6=tgt,
            pose_targets_gripper=tgt_grip,
            deadman_released=dmr,
        )

    def finish_demo(self):
        rospy.loginfo("[Collector] Saving demo...")

        if not self.demo_in_progress:
            rospy.logerr("[Collector] No demo in progress. Nothing to save.")
            return

        # integrity per enabled stream
        if self.enable_full:
            if self._ep_all is None:
                rospy.logerr("[Collector] all_sensors enabled but ep dir is None")
                return
            rgb_ext = ".png"
            ok_all = self._check_integrity_stream(
                self._ep_all, self._frame_all, self._lowdim_all,
                expect_depth=True, rgb_ext=rgb_ext, depth_ext=".npy"
            )
            if not ok_all:
                rospy.logerr("[Collector] all_sensors integrity failed. Deleting episode for redo.")
                self.delete_demo()
                return

        if self.enable_light:
            if self._ep_light is None:
                rospy.logerr("[Collector] light enabled but ep dir is None")
                return
            rgb_ext = ".png"
            ok_light = self._check_integrity_stream(
                self._ep_light, self._frame_light, self._lowdim_light,
                expect_depth=False, rgb_ext=rgb_ext, depth_ext=".npy"
            )
            if not ok_light:
                rospy.logerr("[Collector] light integrity failed. Deleting episode for redo.")
                self.delete_demo()
                return

        # write lowdim
        if self.enable_full and self._ep_all is not None:
            self._write_lowdim(os.path.join(self._ep_all, "lowdim.npz"), self._lowdim_all)
            rospy.loginfo(f"[Collector] Saved episode all_sensors: {self._ep_all}")

        if self.enable_light and self._ep_light is not None:
            self._write_lowdim(os.path.join(self._ep_light, "lowdim.npz"), self._lowdim_light)
            rospy.loginfo(f"[Collector] Saved episode light: {self._ep_light}")

        self.demo_in_progress = False
        self.current_demo_id += 1

        if self.current_demo_id > self.demo_id_end:
            self.robot.home()
            rospy.loginfo("[Collector] All episodes collected. Exiting.")
            rospy.signal_shutdown("done")
            
        # reset robot
        self.robot.home()
        rospy.loginfo(f"[Xarm] Reset robot to home position.")