import rospy
from dataclasses import dataclass
from typing import Dict, Optional
from sensor_msgs.msg import Image, PointCloud2

@dataclass
class CameraState:
    rgb: Optional[Image] = None
    depth: Optional[Image] = None
    points: Optional[PointCloud2] = None

class CameraRig:
    def __init__(self, camera_cfg_list):
        """
        camera_cfg_list: list of dicts like:
          {name: "hand", rgb: "...", depth: "...", points: "..."}
        """
        self.cams: Dict[str, CameraState] = {}
        for cfg in camera_cfg_list:
            name = cfg["name"]
            self.cams[name] = CameraState()

            rospy.Subscriber(cfg["rgb"], Image, lambda m, n=name: self._set(n, "rgb", m), queue_size=2)
            if cfg.get("depth"):
                rospy.Subscriber(cfg["depth"], Image, lambda m, n=name: self._set(n, "depth", m), queue_size=2)
            if cfg.get("points"):
                rospy.Subscriber(cfg["points"], PointCloud2, lambda m, n=name: self._set(n, "points", m), queue_size=2)

        rospy.loginfo(f"CameraRig ready with cameras: {list(self.cams.keys())}")

    def _set(self, cam_name, field, msg):
        setattr(self.cams[cam_name], field, msg)

    def get(self, cam_name: str) -> CameraState:
        return self.cams[cam_name]
