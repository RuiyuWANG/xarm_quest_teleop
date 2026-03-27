#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist, TwistStamped, PoseStamped
from quest2ros.msg import OVR2ROSInputs
from teleop_msgs.msg import OVR2ROSInputsStamped


class QuestStampNode:
    def __init__(self):
        # Topics (can be remapped)
        self.right_twist_in = rospy.get_param("~right_twist_in", "/q2r_right_hand_twist")
        self.left_twist_in  = rospy.get_param("~left_twist_in",  "/q2r_left_hand_twist")
        self.right_inputs_in = rospy.get_param("~right_inputs_in", "/q2r_right_hand_inputs")
        self.left_inputs_in  = rospy.get_param("~left_inputs_in",  "/q2r_left_hand_inputs")

        # NEW: pose inputs
        self.right_pose_in = rospy.get_param("~right_pose_in", "/q2r_right_hand_pose")
        self.left_pose_in  = rospy.get_param("~left_pose_in",  "/q2r_left_hand_pose")

        self.right_twist_out = rospy.get_param("~right_twist_out", "/q2r_right_hand_twist_stamped")
        self.left_twist_out  = rospy.get_param("~left_twist_out",  "/q2r_left_hand_twist_stamped")
        self.right_inputs_out = rospy.get_param("~right_inputs_out", "/q2r_right_hand_inputs_stamped")
        self.left_inputs_out  = rospy.get_param("~left_inputs_out",  "/q2r_left_hand_inputs_stamped")

        # NEW: pose outputs
        self.right_pose_out = rospy.get_param("~right_pose_out", "/q2r_right_hand_pose_stamped")
        self.left_pose_out  = rospy.get_param("~left_pose_out",  "/q2r_left_hand_pose_stamped")

        # Frame IDs
        self.frame_id = rospy.get_param("~frame_id", "quest")
        self.frame_id_pose = rospy.get_param("~frame_id_pose", self.frame_id)

        # Pose stamp behavior:
        # - if True: always overwrite pose header.stamp (even if present)
        # - if False: only stamp if incoming stamp is 0 / unset
        self.always_overwrite_pose_stamp = rospy.get_param("~always_overwrite_pose_stamp", True)

        # Publishers
        self.pub_rt = rospy.Publisher(self.right_twist_out, TwistStamped, queue_size=20)
        self.pub_lt = rospy.Publisher(self.left_twist_out, TwistStamped, queue_size=20)
        self.pub_ri = rospy.Publisher(self.right_inputs_out, OVR2ROSInputsStamped, queue_size=20)
        self.pub_li = rospy.Publisher(self.left_inputs_out, OVR2ROSInputsStamped, queue_size=20)

        # NEW: pose publishers
        self.pub_rp = rospy.Publisher(self.right_pose_out, PoseStamped, queue_size=20)
        self.pub_lp = rospy.Publisher(self.left_pose_out, PoseStamped, queue_size=20)

        # Subscribers
        rospy.Subscriber(self.right_twist_in, Twist, self._cb_right_twist, queue_size=50)
        rospy.Subscriber(self.left_twist_in, Twist, self._cb_left_twist, queue_size=50)
        rospy.Subscriber(self.right_inputs_in, OVR2ROSInputs, self._cb_right_inputs, queue_size=50)
        rospy.Subscriber(self.left_inputs_in, OVR2ROSInputs, self._cb_left_inputs, queue_size=50)

        # NEW: pose subscribers (assumes PoseStamped input, but with bad/empty stamps)
        rospy.Subscriber(self.right_pose_in, PoseStamped, self._cb_right_pose, queue_size=50)
        rospy.Subscriber(self.left_pose_in, PoseStamped, self._cb_left_pose, queue_size=50)

        rospy.loginfo("[quest_stamp_node] started")
        rospy.loginfo(f"  twist:  {self.left_twist_in}, {self.right_twist_in}")
        rospy.loginfo(f"  inputs: {self.left_inputs_in}, {self.right_inputs_in}")
        rospy.loginfo(f"  pose:   {self.left_pose_in}, {self.right_pose_in}")

    def _stamp(self):
        return rospy.Time.now()

    # -------- twist --------
    def _cb_right_twist(self, msg: Twist):
        out = TwistStamped()
        out.header.stamp = self._stamp()
        out.header.frame_id = self.frame_id
        out.twist = msg
        self.pub_rt.publish(out)

    def _cb_left_twist(self, msg: Twist):
        out = TwistStamped
        out = TwistStamped()
        out.header.stamp = self._stamp()
        out.header.frame_id = self.frame_id
        out.twist = msg
        self.pub_lt.publish(out)

    # -------- inputs --------
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

    # -------- pose (NEW) --------
    def _stamp_pose_header(self, msg: PoseStamped) -> PoseStamped:
        out = PoseStamped()
        out.pose = msg.pose
        out.header = msg.header

        if self.always_overwrite_pose_stamp or out.header.stamp.to_sec() <= 0.0:
            out.header.stamp = self._stamp()

        if not out.header.frame_id:
            out.header.frame_id = self.frame_id_pose

        return out

    def _cb_right_pose(self, msg: PoseStamped):
        out = self._stamp_pose_header(msg)
        self.pub_rp.publish(out)

    def _cb_left_pose(self, msg: PoseStamped):
        out = self._stamp_pose_header(msg)
        self.pub_lp.publish(out)


def main():
    rospy.init_node("quest_stamp_node", anonymous=False)
    QuestStampNode()
    rospy.spin()


if __name__ == "__main__":
    main()
