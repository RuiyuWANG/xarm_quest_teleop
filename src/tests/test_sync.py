#!/usr/bin/env python3
import rospy
from xarm_msgs.msg import RobotMsg
from teleop_msgs.msg import OVR2ROSInputsStamped

# Global variables to store last timestamps
last_quest_time = None
last_robot_time = None

def quest_cb(msg):
    global last_quest_time
    last_quest_time = msg.header.stamp.to_sec()
    compare()

def robot_cb(msg):
    global last_robot_time
    last_robot_time = msg.header.stamp.to_sec()
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
rospy.Subscriber("/xarm/xarm_states", RobotMsg, robot_cb)

print("Monitoring timestamp difference... (Goal: < 0.05s)")
rospy.spin()