#!/usr/bin/env bash
set -euo pipefail

ROS_DISTRO="${ROS_DISTRO:-humble}"
ROS2_WS="${ROS2_WS:-$HOME/ros2_ws}"

source "/opt/ros/${ROS_DISTRO}/setup.bash"
if [[ -f "${ROS2_WS}/install/setup.bash" ]]; then
  source "${ROS2_WS}/install/setup.bash"
fi

echo "[check] package imports"
python3 - <<'PY'
import xarm_quest_teleop
from xarm_quest_teleop.policy.registry import available_policies
print("xarm_quest_teleop:", xarm_quest_teleop.__file__)
print("policies:", sorted(available_policies()))
PY

echo "[check] interfaces"
ros2 interface show xarm_quest_teleop_msgs/msg/OVR2ROSInputsStamped

echo "[check] topics"
ros2 topic list | grep -E '(/q2r_|/xarm/robot_states|/d405/color/image_raw|/d435i_front/color/image_raw)' || true

echo "[check] required xArm services"
for srv in \
  /xarm/set_mode \
  /xarm/set_state \
  /xarm/motion_enable \
  /xarm/set_servo_angle \
  /xarm/set_position \
  /xarm/set_servo_cartesian \
  /xarm/set_gripper_position \
  /xarm/get_gripper_position
do
  if ros2 service list | grep -qx "${srv}"; then
    echo "  ok ${srv}"
  else
    echo "  missing ${srv}"
  fi
done
