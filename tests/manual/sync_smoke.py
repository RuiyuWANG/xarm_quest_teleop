#!/usr/bin/env python3
from xarm_quest_teleop.ros import compat as rospy
from xarm_msgs.msg import RobotMsg
from xarm_quest_teleop_msgs.msg import OVR2ROSInputsStamped

# Global variables to store last timestamps
last_quest_time = None
last_robot_time = None

def stamp_to_sec(stamp):
    if hasattr(stamp, "to_sec"):
        return stamp.to_sec()
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9

def quest_cb(msg):
    global last_quest_time
    last_quest_time = stamp_to_sec(msg.header.stamp)
    compare()

def robot_cb(msg):
    global last_robot_time
    last_robot_time = stamp_to_sec(msg.header.stamp)
    compare()

def compare():
    if last_quest_time and last_robot_time:
        diff = abs(last_quest_time - last_robot_time)
        # Use color to highlight issues
        color = "\033[92m" if diff < 0.1 else "\033[91m"
        reset = "\033[0m"
        print(f"Time Difference: {color}{diff:.4f}s{reset} | Quest: {last_quest_time:.2f} | Robot: {last_robot_time:.2f}")

rospy.init_node("sync_checker")

rospy.Subscriber("/q2r_right_hand_inputs_stamped", OVR2ROSInputsStamped, quest_cb)
rospy.Subscriber("/xarm/robot_states", RobotMsg, robot_cb)

print("Monitoring timestamp difference... (Goal: < 0.05s)")
rospy.spin()
