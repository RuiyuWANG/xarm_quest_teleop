import rospy
from dataclasses import dataclass
from geometry_msgs.msg import PoseStamped, Twist

# Replace with your actual msg package names
# from quest2ros_msgs.msg import HandInputs, HapticFeedback

@dataclass
class HandState:
    pose: PoseStamped = None
    twist: Twist = None
    inputs = None  # HandInputs

class Quest2Client:
    def __init__(self,
                 left_pose, left_twist, left_inputs, left_haptics_pub,
                 right_pose, right_twist, right_inputs, right_haptics_pub,
                 inputs_msg_type, haptics_msg_type):
        """
        inputs_msg_type / haptics_msg_type: pass the actual Python classes for your custom msgs.
        e.g. from quest2ros_msgs.msg import HandInputs, HapticFeedback
        """
        self.HandInputs = inputs_msg_type
        self.HapticFeedback = haptics_msg_type

        self.left = HandState()
        self.right = HandState()

        self.left_haptics_pub = rospy.Publisher(left_haptics_pub, self.HapticFeedback, queue_size=10)
        self.right_haptics_pub = rospy.Publisher(right_haptics_pub, self.HapticFeedback, queue_size=10)

        rospy.Subscriber(left_pose, PoseStamped, lambda m: setattr(self.left, "pose", m), queue_size=10)
        rospy.Subscriber(left_twist, Twist,      lambda m: setattr(self.left, "twist", m), queue_size=10)
        rospy.Subscriber(left_inputs, self.HandInputs, lambda m: setattr(self.left, "inputs", m), queue_size=10)

        rospy.Subscriber(right_pose, PoseStamped, lambda m: setattr(self.right, "pose", m), queue_size=10)
        rospy.Subscriber(right_twist, Twist,      lambda m: setattr(self.right, "twist", m), queue_size=10)
        rospy.Subscriber(right_inputs, self.HandInputs, lambda m: setattr(self.right, "inputs", m), queue_size=10)

    def send_haptics(self, hand: str, frequency: float, amplitude: float):
        msg = self.HapticFeedback()
        msg.frequency = float(frequency)
        msg.amplitude = float(amplitude)
        if hand == "left":
            self.left_haptics_pub.publish(msg)
        elif hand == "right":
            self.right_haptics_pub.publish(msg)
        else:
            raise ValueError("hand must be 'left' or 'right'")
