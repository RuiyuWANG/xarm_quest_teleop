#!/usr/bin/env python3
from __future__ import annotations

import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

from xarm_quest_teleop.ros import compat as rospy


class RosRgbViewer:
    def __init__(self):
        self.topic = rospy.get_param("~topic", "/camera/color/image_raw")
        self.bridge = CvBridge()
        self.win = f"ROS RGB: {self.topic}"

        rospy.loginfo(f"Subscribing to {self.topic}")
        self.sub = rospy.Subscriber(self.topic, Image, self.cb, queue_size=1)

    def cb(self, msg: Image):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            rospy.logerr(f"cv_bridge conversion failed: {exc}")
            return

        cv2.imshow(self.win, img)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            rospy.signal_shutdown("User quit")


def main():
    rospy.init_node("ros_rgb_view", anonymous=True)
    RosRgbViewer()
    rospy.on_shutdown(cv2.destroyAllWindows)
    rospy.spin()


if __name__ == "__main__":
    main()
