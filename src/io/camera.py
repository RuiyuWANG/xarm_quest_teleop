from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import rospy
from sensor_msgs.msg import Image


@dataclass
class SyncedCameraMsgs:
    rgb: Optional[Image]
    depth: Optional[Image]


class _Ring:
    def __init__(self, keep_s: float):
        self.keep_s = float(keep_s)
        self.buf = []

    def push(self, msg):
        self.buf.append(msg)
        now = msg.header.stamp.to_sec()
        cutoff = now - self.keep_s
        while self.buf and self.buf[0].header.stamp.to_sec() < cutoff:
            self.buf.pop(0)

    def nearest(self, t: float, window: float):
        if not self.buf:
            return None
        best, best_dt = None, 1e18
        for m in self.buf:
            dt = abs(m.header.stamp.to_sec() - t)
            if dt < best_dt:
                best, best_dt = m, dt
        return best if best_dt <= float(window) else None


class CameraSync:
    """
    Multi-camera buffer with enforced RGB->Depth sync:

    For each camera:
      1) choose RGB nearest to t_sync
      2) choose Depth nearest to RGB stamp (if RGB exists), else nearest to t_sync
    """
    def __init__(self, cameras: Dict[str, Dict[str, str]], keep_s: float, match_window_s: float, queue_size: int):
        self.match_window_s = float(match_window_s)
        self.rgb_buf: Dict[str, _Ring] = {}
        self.depth_buf: Dict[str, _Ring] = {}

        for name, spec in cameras.items():
            self.rgb_buf[name] = _Ring(keep_s)
            self.depth_buf[name] = _Ring(keep_s)

            rospy.Subscriber(
                spec["rgb_topic"], Image,
                lambda m, n=name: self.rgb_buf[n].push(m),
                queue_size=queue_size
            )
            rospy.Subscriber(
                spec["depth_topic"], Image,
                lambda m, n=name: self.depth_buf[n].push(m),
                queue_size=queue_size
            )

            rospy.loginfo(f"[CameraSync] {name} rgb={spec['rgb_topic']} depth={spec['depth_topic']}")

    def nearest(self, t_sync: float) -> Dict[str, SyncedCameraMsgs]:
        out: Dict[str, SyncedCameraMsgs] = {}
        w = self.match_window_s

        for name in self.rgb_buf.keys():
            rgb = self.rgb_buf[name].nearest(t_sync, w)
            if rgb is not None:
                t_ref = rgb.header.stamp.to_sec()
                depth = self.depth_buf[name].nearest(t_ref, w)
            else:
                depth = self.depth_buf[name].nearest(t_sync, w)

            out[name] = SyncedCameraMsgs(rgb=rgb, depth=depth)

        return out