#!/usr/bin/env python3
import rospy
from xarm_msgs.msg import RobotMsg
from teleop_msgs.msg import RobotMsgStamped

class RobotStateStampNode:
    def __init__(self):
        # Default matches xarm namespace
        topic_in  = rospy.get_param("~robot_topic_in", "/xarm/robot_states")
        topic_out = rospy.get_param("~robot_topic_out", "/xarm/robot_states_stamped")
        frame_id  = rospy.get_param("~frame_id", "xarm")

        self.pub = rospy.Publisher(topic_out, RobotMsgStamped, queue_size=50)
        rospy.Subscriber(topic_in, RobotMsg, self._cb, queue_size=100)

        self.frame_id = frame_id
        rospy.loginfo(f"[robot_state_stamp_node] {topic_in} -> {topic_out}")

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
