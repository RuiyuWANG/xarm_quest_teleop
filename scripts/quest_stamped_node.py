#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist, TwistStamped
from quest2ros.msg import OVR2ROSInputs
from teleop_msgs.msg import OVR2ROSInputsStamped


class QuestStampNode:
    def __init__(self):
        # Topics (can be remapped)
        self.right_twist_in = rospy.get_param("~right_twist_in", "/q2r_right_hand_twist")
        self.left_twist_in  = rospy.get_param("~left_twist_in",  "/q2r_left_hand_twist")
        self.right_inputs_in = rospy.get_param("~right_inputs_in", "/q2r_right_hand_inputs")
        self.left_inputs_in  = rospy.get_param("~left_inputs_in",  "/q2r_left_hand_inputs")

        self.right_twist_out = rospy.get_param("~right_twist_out", "/q2r_right_hand_twist_stamped")
        self.left_twist_out  = rospy.get_param("~left_twist_out",  "/q2r_left_hand_twist_stamped")
        self.right_inputs_out = rospy.get_param("~right_inputs_out", "/q2r_right_hand_inputs_stamped")
        self.left_inputs_out  = rospy.get_param("~left_inputs_out",  "/q2r_left_hand_inputs_stamped")

        self.frame_id = rospy.get_param("~frame_id", "quest")

        self.pub_rt = rospy.Publisher(self.right_twist_out, TwistStamped, queue_size=20)
        self.pub_lt = rospy.Publisher(self.left_twist_out, TwistStamped, queue_size=20)
        self.pub_ri = rospy.Publisher(self.right_inputs_out, OVR2ROSInputsStamped, queue_size=20)
        self.pub_li = rospy.Publisher(self.left_inputs_out, OVR2ROSInputsStamped, queue_size=20)

        rospy.Subscriber(self.right_twist_in, Twist, self._cb_right_twist, queue_size=50)
        rospy.Subscriber(self.left_twist_in, Twist, self._cb_left_twist, queue_size=50)
        rospy.Subscriber(self.right_inputs_in, OVR2ROSInputs, self._cb_right_inputs, queue_size=50)
        rospy.Subscriber(self.left_inputs_in, OVR2ROSInputs, self._cb_left_inputs, queue_size=50)

        rospy.loginfo("[quest_stamp_node] started")

    def _stamp(self):
        return rospy.Time.now()

    def _cb_right_twist(self, msg: Twist):
        out = TwistStamped()
        out.header.stamp = self._stamp()
        out.header.frame_id = self.frame_id
        out.twist = msg
        self.pub_rt.publish(out)

    def _cb_left_twist(self, msg: Twist):
        out = TwistStamped()
        out.header.stamp = self._stamp()
        out.header.frame_id = self.frame_id
        out.twist = msg
        self.pub_lt.publish(out)

    def _cb_right_inputs(self, msg: OVR2ROSInputs):
        out = OVR2ROSInputsStamped()
        out.header.stamp = self._stamp()
        out.header.frame_id = self.frame_id
        out.inputs = msg
        self.pub_ri.publish(out)

    def _cb_left_inputs(self, msg: OVR2ROSInputs):
        out = OVR2ROSInputsStamped()
        out.header.stamp = self._stamp()
        out.header.frame_id = self.frame_id
        out.inputs = msg
        self.pub_li.publish(out)


def main():
    rospy.init_node("quest_stamp_node", anonymous=False)
    QuestStampNode()
    rospy.spin()


if __name__ == "__main__":
    main()
