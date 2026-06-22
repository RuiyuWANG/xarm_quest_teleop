#!/usr/bin/env python3
# Usage examples:
#   ros2 run xarm_quest_teleop ros_rgb_view --ros-args -p topic:=/d435i/color/image_raw
#   ros2 run xarm_quest_teleop ros_rgb_view --ros-args -p topic:=/d405/color/image_raw
#   ros2 run xarm_quest_teleop ros_rgb_view --ros-args -p topic:=/camera/color/image_raw

from xarm_quest_teleop.scripts.ros_rgb_view import main

if __name__ == "__main__":
    main()
