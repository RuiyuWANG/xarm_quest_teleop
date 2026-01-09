#!/usr/bin/env python3
# scripts/run_quest_xarm_teleop.py
"""
Launch script (ROS node) for QuestXArmTeleop.

This keeps "main" separate from the library class so your data collection pipeline
can import and reuse QuestXArmTeleop directly.
"""

import rospy
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.io.quest2 import Quest2Interface
from src.robots.xarm import XArmRobot  # adjust import
from src.configs.teleop_config import TeleopConfig
from src.teleop.quest_xarm_teleop import QuestXArmTeleop


def main():
    rospy.init_node("quest_xarm_teleop", anonymous=False)

    quest = Quest2Interface(debug=False)
    robot = XArmRobot(auto_init=True, debug=False)

    cfg = TeleopConfig()

    teleop = QuestXArmTeleop(quest=quest, robot=robot, cfg=cfg)

    # Example hook for later data collection:
    # def collector_hook(sample):
    #     pass
    # teleop.register_hook(collector_hook)

    teleop.spin()


if __name__ == "__main__":
    main()
