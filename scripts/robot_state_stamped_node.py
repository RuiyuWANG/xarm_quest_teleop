#!/usr/bin/env python3
import rospy
from xarm_msgs.msg import RobotMsg
from teleop_msgs.msg import RobotMsgStamped

from src.configs.robot_config import ROBOT_TOPIC


class RobotStateStampNode:
    def __init__(self):
        self.robot_topic_in = rospy.get_param("~robot_topic_in", ROBOT_TOPIC)
        self.robot_topic_out = rospy.get_param("~robot_topic_out", "/xarm/robot_states_stamped")
        self.frame_id = rospy.get_param("~frame_id", "xarm")

        self.pub = rospy.Publisher(self.robot_topic_out, RobotMsgStamped, queue_size=50)
        rospy.Subscriber(self.robot_topic_in, RobotMsg, self._cb, queue_size=50)

        rospy.loginfo(f"[robot_state_stamp_node] {self.robot_topic_in} -> {self.robot_topic_out}")

    def _cb(self, msg: RobotMsg):
        out = RobotMsgStamped()
        out.header.stamp = rospy.Time.now()
        out.header.frame_id = self.frame_id
        out.robot = msg
        self.pub.publish(out)


def main():
    rospy.init_node("robot_state_stamp_node", anonymous=False)
    RobotStateStampNode()
    rospy.spin()


if __name__ == "__main__":
    main()
