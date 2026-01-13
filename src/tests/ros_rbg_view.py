#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

# Usage examples:
#   rosrun <your_pkg> ros_rgb_view.py _topic:=/d435i/color/image_raw
#   rosrun <your_pkg> ros_rgb_view.py _topic:=/d405/color/image_raw
#   rosrun <your_pkg> ros_rgb_view.py _topic:=/camera/color/image_raw

class RosRgbViewer:
    def __init__(self):
        self.topic = rospy.get_param("~topic", "/camera/color/image_raw")
        self.bridge = CvBridge()
        self.win = f"ROS RGB: {self.topic}"

        rospy.loginfo(f"Subscribing to {self.topic}")
        self.sub = rospy.Subscriber(self.topic, Image, self.cb, queue_size=1)

    def cb(self, msg: Image):
        try:
            # Most RealSense color topics are already bgr8, but this handles rgb8 too.
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logerr(f"cv_bridge conversion failed: {e}")
            return

        cv2.imshow(self.win, img)
        # Press q or Esc to quit
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            rospy.signal_shutdown("User quit")

def main():
    rospy.init_node("ros_rgb_view", anonymous=True)
    RosRgbViewer()
    rospy.on_shutdown(cv2.destroyAllWindows)
    rospy.spin()

if __name__ == "__main__":
    main()
