#!/usr/bin/env bash
set -euo pipefail

ROS_DISTRO="${ROS_DISTRO:-humble}"
ROS2_WS="${ROS2_WS:-$HOME/ros2_ws}"

source "/opt/ros/${ROS_DISTRO}/setup.bash"
cd "${ROS2_WS}"
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install

echo "Built ${ROS2_WS}. Source install/setup.bash before running xArm Quest Teleop."
