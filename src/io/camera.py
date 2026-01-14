from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import rospy
from sensor_msgs.msg import Image, PointCloud2


@dataclass
class SyncedCameraMsgs:
    rgb: Optional[Image]
    cloud: Optional[PointCloud2]


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
    def __init__(self, cameras: Dict[str, Dict[str, str]], keep_s: float, match_window_s: float, queue_size: int):
        self.match_window_s = float(match_window_s)
        self.rgb_buf: Dict[str, _Ring] = {}
        self.cloud_buf: Dict[str, _Ring] = {}

        for name, spec in cameras.items():
            self.rgb_buf[name] = _Ring(keep_s)
            self.cloud_buf[name] = _Ring(keep_s)

            rospy.Subscriber(
                spec["rgb_topic"], Image,
                lambda m, n=name: self.rgb_buf[n].push(m),
                queue_size=queue_size
            )
            rospy.Subscriber(
                spec["cloud_topic"], PointCloud2,
                lambda m, n=name: self.cloud_buf[n].push(m),
                queue_size=queue_size
            )

            rospy.loginfo(f"[CameraSync] {name} rgb={spec['rgb_topic']} cloud={spec['cloud_topic']}")

    def nearest(self, t_sync: float) -> Dict[str, SyncedCameraMsgs]:
        out: Dict[str, SyncedCameraMsgs] = {}
        for name in self.rgb_buf.keys():
            out[name] = SyncedCameraMsgs(
                rgb=self.rgb_buf[name].nearest(t_sync, self.match_window_s),
                cloud=self.cloud_buf[name].nearest(t_sync, self.match_window_s),
            )
        return out
