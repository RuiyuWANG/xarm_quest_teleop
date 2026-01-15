from __future__ import annotations

import os
import time
import json
import shutil
import queue
import threading
from typing import Any, Dict, Optional, List

import numpy as np
import rospy
import cv2
from cv_bridge import CvBridge
from pynput import keyboard
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2

from src.utils.config_utils import DatasetInfo, next_demo_id
from src.configs.collector_config import CollectorConfig
from src.io.camera import SyncedCameraMsgs

from src.teleop.quest_xarm_teleop_sync import SyncedSample  # adjust if needed
from src.utils.teleop_utils import wrap_to_pi


def _ensure(p: str):
    os.makedirs(p, exist_ok=True)
    
def _stack_optional_6(samples: List[Dict[str, Any]], key: str) -> np.ndarray:
    out = np.full((len(samples), 6), np.nan, dtype=np.float32)
    for i, s in enumerate(samples):
        v = s.get(key, None)
        if v is None:
            continue
        if isinstance(v, list) and len(v) == 6:
            out[i] = np.asarray(v, dtype=np.float32)
    return out

def _write_pcd_xyz_ascii(path: str, xyz: np.ndarray):
    n = int(xyz.shape[0])
    header = "\n".join([
        "# .PCD v0.7 - Point Cloud Data file format",
        "VERSION 0.7",
        "FIELDS x y z",
        "SIZE 4 4 4",
        "TYPE F F F",
        "COUNT 1 1 1",
        f"WIDTH {n}",
        "HEIGHT 1",
        "VIEWPOINT 0 0 0 1 0 0 0",
        f"POINTS {n}",
        "DATA ascii",
    ]) + "\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write(header)
        for x, y, z in xyz:
            f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")

class TeleopDataCollector:
    def __init__(self, ds: DatasetInfo, cfg: CollectorConfig, robot):
        self.ds = ds
        self.cfg = cfg
        self.robot = robot
        self.bridge = CvBridge()

        _ensure(ds.root_dir)

        existing_next = next_demo_id(ds.root_dir, ds.demo_prefix)
        self.current_demo_id = max(int(existing_next), int(ds.demo_id_start))
        self.demo_id_end = int(ds.demo_id_start) + int(ds.num_demos) - 1

        self.demo_in_progress = False
        self.continue_trajectory = False
        self.delete_requested = False
        self.quit_requested = False

        self._demo_dir: Optional[str] = None
        self._start_wall = 0.0
        self._samples: List[Dict[str, Any]] = []
        self._frame_idx = 0

        # async queue to keep teleop responsive
        self._q: "queue.Queue[SyncedSample]" = queue.Queue(maxsize=int(cfg.max_queue))
        self._last_enqueue_wall = 0.0
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

        self.keyboard_listener()

        rospy.loginfo(f"[Collector] root={ds.root_dir}")
        rospy.loginfo(f"[Collector] demo range: {self.current_demo_id} .. {self.demo_id_end}")

        if self.current_demo_id > self.demo_id_end:
            rospy.loginfo("[Collector] All demos already collected. Exiting.")
            rospy.signal_shutdown("done")

    def keyboard_listener(self):
        def on_press(key):
            try:
                if key.char == self.cfg.start_key:
                    if not self.demo_in_progress:
                        print(f"[Collector] Starting demo {self.current_demo_id}...")
                        self.start_demo()  # random start occurs here
                    else:
                        print("[Collector] Continuing trajectory execution.")
                        self.continue_trajectory = True
                elif key.char == self.cfg.delete_key:
                    print("[Collector] Delete requested.")
                    self.delete_requested = True
                elif key.char == self.cfg.quit_key:
                    print("[Collector] Quit requested.")
                    self.quit_requested = True
            except AttributeError:
                pass
        keyboard.Listener(on_press=on_press).start()

    def _worker_loop(self):
        while not rospy.is_shutdown():
            try:
                sample = self._q.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                self.on_sample(sample)  # heavy IO here
            except Exception as e:
                rospy.logwarn_throttle(1.0, f"[Collector] worker exception: {e}")
            finally:
                try:
                    self._q.task_done()
                except Exception:
                    pass

    def enqueue(self, sample: SyncedSample):
        """Fast path called from teleop thread. Does not block."""
        if not self.demo_in_progress:
            return

        # rate-limit saving independently from teleop rate
        if self.cfg.save_rate_hz > 0:
            now = time.time()
            min_dt = 1.0 / float(self.cfg.save_rate_hz)
            if (now - self._last_enqueue_wall) < min_dt:
                return
            self._last_enqueue_wall = now

        try:
            self._q.put_nowait(sample)
        except queue.Full:
            # drop to keep teleop reactive
            pass

    def _random_start_move(self):
        if not getattr(self.ds, "random_start_enabled", True):
            return

        st = self.robot.get_state()
        if st.ee_pose is None or len(st.ee_pose) < 6:
            rospy.logwarn("[Collector] random_start skipped: no ee_pose")
            return

        cur = np.array(st.ee_pose[:6], dtype=np.float32)
        pos_range = np.array(self.ds.pos_mm_xyz, dtype=np.float32)
        rot_range = np.array(self.ds.rot_rad_rpy, dtype=np.float32)

        dx = np.random.uniform(-1.0, 1.0, size=(3,)).astype(np.float32) * pos_range
        dr = np.random.uniform(-1.0, 1.0, size=(3,)).astype(np.float32) * rot_range

        tgt = cur.copy()
        tgt[:3] = cur[:3] + dx
        tgt[3:6] = wrap_to_pi(cur[3:6] + dr)

        rospy.loginfo(f"[Collector] random_start dx(mm)={dx.tolist()} dr(rad)={dr.tolist()}")
        self.robot.move_servo_cart(pose6_mm_rpy=tgt.tolist(), tool_coord=False)
        time.sleep(0.25)

    def start_demo(self):
        if self.current_demo_id > self.demo_id_end:
            rospy.loginfo("[Collector] Target demo count reached. Exiting.")
            rospy.signal_shutdown("done")
            return

        demo_name = f"{self.ds.demo_prefix}{self.current_demo_id:04d}"
        self._demo_dir = os.path.join(self.ds.root_dir, demo_name)
        if os.path.exists(self._demo_dir):
            shutil.rmtree(self._demo_dir)

        _ensure(self._demo_dir)
        _ensure(os.path.join(self._demo_dir, "frames"))
        _ensure(os.path.join(self._demo_dir, "depth"))

        # random start on 'c'
        # self._random_start_move() # no longer random start

        self._samples.clear()
        self._frame_idx = 0
        self._start_wall = time.time()

        self.demo_in_progress = True
        self.continue_trajectory = False
        self.delete_requested = False

    def finish_demo(self):
        if not self.demo_in_progress:
            return

        traj_path = os.path.join(self._demo_dir, "traj.npz")
        meta_path = os.path.join(self._demo_dir, "meta.json")

        t = np.array([s["stamp_sync"] for s in self._samples], dtype=np.float64)
        ee = np.array([s["robot_ee_pose6_mm_rpy"] for s in self._samples], dtype=np.float32)
        cmd = _stack_optional_6(self._samples, "cmd_pose6_mm_rpy")
        desired = _stack_optional_6(self._samples, "desired_pose6_mm_rpy")

        np.savez_compressed(traj_path, t=t, ee=ee, cmd=cmd, desired=desired)

        meta = {
            "dataset": {
                "name": self.ds.name,
                "task": self.ds.task,
                "operator": self.ds.operator,
            },
            "collection": {
                "demo_id": self.current_demo_id,
                "demo_id_end": self.demo_id_end,
                "num_samples": len(self._samples),
                "save_rate_hz": float(self.cfg.save_rate_hz),
                "started_wall": self._start_wall,
                "ended_wall": time.time(),
            },
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        rospy.loginfo(f"[Collector] Saved demo: {self._demo_dir}")

        self.demo_in_progress = False
        self.current_demo_id += 1

        if self.current_demo_id > self.demo_id_end:
            rospy.loginfo("[Collector] All demos collected. Exiting.")
            rospy.signal_shutdown("done")

    def delete_demo(self):
        if self._demo_dir and os.path.isdir(self._demo_dir):
            rospy.logwarn(f"[Collector] Deleting {self._demo_dir}")
            shutil.rmtree(self._demo_dir, ignore_errors=True)
        self.demo_in_progress = False
        self._samples.clear()
        self._frame_idx = 0

    def on_sample(self, sample: SyncedSample):
        # keyboard control handling
        if self.quit_requested:
            if self.demo_in_progress:
                self.finish_demo()
            rospy.signal_shutdown("quit requested")
            return

        if self.delete_requested:
            self.delete_requested = False
            self.delete_demo()
            return

        if not self.demo_in_progress:
            return

        frame_idx = self._frame_idx

        for cam_name, cam_msgs in sample.cameras.items():
            # RGB
            if self.cfg.save_rgb_png and getattr(cam_msgs, "rgb", None) is not None:
                try:
                    img = self.bridge.imgmsg_to_cv2(cam_msgs.rgb, desired_encoding="bgr8")
                except Exception:
                    img = None
                if img is not None:
                    out_dir = os.path.join(self._demo_dir, "frames", cam_name)
                    _ensure(out_dir)
                    cv2.imwrite(os.path.join(out_dir, f"{frame_idx:06d}_rgb.png"), img)

            # Depth (typically 16UC1)
            if self.cfg.save_depth_png and getattr(cam_msgs, "depth", None) is not None:
                try:
                    depth = self.bridge.imgmsg_to_cv2(cam_msgs.depth, desired_encoding="passthrough")
                except Exception:
                    depth = None
                if depth is not None:
                    out_dir = os.path.join(self._demo_dir, "frames", cam_name)
                    _ensure(out_dir)
                    # Save raw uint16 depth in millimeters (RealSense usually publishes mm in 16UC1)
                    cv2.imwrite(os.path.join(out_dir, f"{frame_idx:06d}_depth.png"), depth)


        # record minimal numeric core + cmd/desired
        ee6 = sample.robot_state.ee_pose[:6] if sample.robot_state.ee_pose else [np.nan] * 6
        cmd6 = sample.cmd_pose6_mm_rpy if sample.cmd_pose6_mm_rpy is not None else None
        des6 = sample.desired_pose6_mm_rpy if sample.desired_pose6_mm_rpy is not None else None

        self._samples.append({
            "stamp_sync": float(sample.stamp_sync),
            "robot_ee_pose6_mm_rpy": ee6,
            "cmd_pose6_mm_rpy": cmd6,
            "desired_pose6_mm_rpy": des6,
        })

        self._frame_idx += 1

