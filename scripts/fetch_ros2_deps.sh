#!/usr/bin/env bash
set -euo pipefail

ROS2_WS="${ROS2_WS:-$HOME/ros2_ws}"
SRC_DIR="${ROS2_WS}/src"
mkdir -p "${SRC_DIR}"

clone_or_update() {
  local url="$1"
  local branch="$2"
  local dir="$3"
  if [[ -d "${dir}/.git" ]]; then
    git -C "${dir}" fetch --depth 1 origin "${branch}"
    git -C "${dir}" checkout "${branch}"
    git -C "${dir}" reset --hard "origin/${branch}"
  else
    git clone --depth 1 --branch "${branch}" "${url}" "${dir}"
  fi
}

clone_or_update "https://github.com/Quest2ROS/quest2ros.git" "ros2" "${SRC_DIR}/quest2ros"
clone_or_update "https://github.com/MWelle77/ROS-TCP-Endpoint.git" "main-ros2" "${SRC_DIR}/ROS-TCP-Endpoint"
clone_or_update "https://github.com/xArm-Developer/xarm_ros2.git" "humble" "${SRC_DIR}/xarm_ros2"
clone_or_update "https://github.com/realsenseai/realsense-ros.git" "ros2-master" "${SRC_DIR}/realsense-ros"

THIS_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${SRC_DIR}/xarm_quest_teleop"
if [[ "${THIS_REPO}" != "${TARGET}" ]]; then
  if [[ -e "${TARGET}" && ! -L "${TARGET}" ]]; then
    echo "${TARGET} exists and is not a symlink; leaving it untouched." >&2
  elif [[ ! -e "${TARGET}" ]]; then
    ln -s "${THIS_REPO}" "${TARGET}"
  fi
fi

echo "ROS2 dependencies are available under ${SRC_DIR}."
