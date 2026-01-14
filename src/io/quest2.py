"""
Quest2Interface for Quest2ROS messages.

This class:
  - stores latest left/right pose/twist/inputs
  - provides helper methods: deadman(), triggers, buttons
  - provides haptic publishers: vibrate_left/right()
"""

from dataclasses import dataclass
from typing import Optional, Literal

import rospy
from geometry_msgs.msg import PoseStamped, Twist
from quest2ros.msg import OVR2ROSInputs, OVR2ROSHapticFeedback

from src.configs.quest2_config import (
    Q2R_RIGHT_HAND_POSE, Q2R_RIGHT_HAND_TWIST, Q2R_RIGHT_HAND_INPUTS, Q2R_RIGHT_HAND_HAPTIC,
    Q2R_LEFT_HAND_POSE, Q2R_LEFT_HAND_TWIST, Q2R_LEFT_HAND_INPUTS, Q2R_LEFT_HAND_HAPTIC,
)


Hand = Literal["left", "right"]


@dataclass
class QuestHandState:
    pose: Optional[PoseStamped] = None
    twist: Optional[Twist] = None
    inputs: Optional[OVR2ROSInputs] = None


class Quest2Interface:
    def __init__(
        self,
        right_pose_topic: str = Q2R_RIGHT_HAND_POSE,
        right_twist_topic: str = Q2R_RIGHT_HAND_TWIST,
        right_inputs_topic: str = Q2R_RIGHT_HAND_INPUTS,
        left_pose_topic: str = Q2R_LEFT_HAND_POSE,
        left_twist_topic: str = Q2R_LEFT_HAND_TWIST,
        left_inputs_topic: str = Q2R_LEFT_HAND_INPUTS,
        right_haptic_topic: str = Q2R_RIGHT_HAND_HAPTIC,
        left_haptic_topic: str = Q2R_LEFT_HAND_HAPTIC,
        queue_size: int = 10,
        debug: bool = False,
    ):
        self.debug = debug

        self.left = QuestHandState()
        self.right = QuestHandState()

        # subs
        rospy.Subscriber(right_pose_topic, PoseStamped, self._right_pose_cb, queue_size=queue_size)
        rospy.Subscriber(right_twist_topic, Twist, self._right_twist_cb, queue_size=queue_size)
        rospy.Subscriber(right_inputs_topic, OVR2ROSInputs, self._right_inputs_cb, queue_size=queue_size)

        rospy.Subscriber(left_pose_topic, PoseStamped, self._left_pose_cb, queue_size=queue_size)
        rospy.Subscriber(left_twist_topic, Twist, self._left_twist_cb, queue_size=queue_size)
        rospy.Subscriber(left_inputs_topic, OVR2ROSInputs, self._left_inputs_cb, queue_size=queue_size)

        # pubs (haptics)
        self._pub_haptic_right = rospy.Publisher(right_haptic_topic, OVR2ROSHapticFeedback, queue_size=1)
        self._pub_haptic_left = rospy.Publisher(left_haptic_topic, OVR2ROSHapticFeedback, queue_size=1)

        rospy.loginfo("[Quest2Interface] connected to Quest2ROS topics")

    # ---------------- callbacks ----------------
    def _right_pose_cb(self, msg: PoseStamped):
        self.right.pose = msg

    def _right_twist_cb(self, msg: Twist):
        self.right.twist = msg

    def _right_inputs_cb(self, msg: OVR2ROSInputs):
        self.right.inputs = msg
        if self.debug:
            rospy.loginfo_throttle(
                0.5,
                f"[Quest2Interface] right lower={getattr(msg,'button_lower',None)} "
                f"idx={getattr(msg,'press_index',None):.3f} mid={getattr(msg,'press_middle',None):.3f}"
            )

    def _left_pose_cb(self, msg: PoseStamped):
        self.left.pose = msg

    def _left_twist_cb(self, msg: Twist):
        self.left.twist = msg

    def _left_inputs_cb(self, msg: OVR2ROSInputs):
        self.left.inputs = msg
        if self.debug:
            rospy.loginfo_throttle(
                0.5,
                f"[Quest2Interface] left  lower={getattr(msg,'button_lower',None)} "
                f"idx={getattr(msg,'press_index',None):.3f} mid={getattr(msg,'press_middle',None):.3f}"
            )

    # ---------------- convenience getters ----------------
    def hand(self, which: Hand) -> QuestHandState:
        return self.left if which == "left" else self.right

    def has_data(self, which: Hand) -> bool:
        h = self.hand(which)
        return (h.pose is not None) and (h.twist is not None) and (h.inputs is not None)

    # Buttons / triggers are field-based in your msg, so expose them cleanly:
    def button_lower(self, which: Hand) -> bool:
        h = self.hand(which)
        return bool(getattr(h.inputs, "button_lower", False)) if h.inputs is not None else False

    def press_index(self, which: Hand) -> float:
        h = self.hand(which)
        return float(getattr(h.inputs, "press_index", 0.0)) if h.inputs is not None else 0.0

    def press_middle(self, which: Hand) -> float:
        h = self.hand(which)
        return float(getattr(h.inputs, "press_middle", 0.0)) if h.inputs is not None else 0.0

    # ---------------- haptics ----------------
    def vibrate(self, which: Hand, frequency: float, amplitude: float):
        """
        Publish a single haptic command. You can call this at your teleop rate.
        """
        msg = OVR2ROSHapticFeedback()
        msg.frequency = float(frequency)
        msg.amplitude = float(amplitude)
        if which == "left":
            self._pub_haptic_left.publish(msg)
        else:
            self._pub_haptic_right.publish(msg)

    def vibrate_both(self, frequency: float, amplitude: float):
        self.vibrate("left", frequency, amplitude)
        self.vibrate("right", frequency, amplitude)
