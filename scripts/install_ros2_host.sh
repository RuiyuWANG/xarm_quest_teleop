#!/usr/bin/env bash
set -euo pipefail

ROS_DISTRO="${ROS_DISTRO:-humble}"

if [[ "$(lsb_release -cs)" != "jammy" ]]; then
  echo "This installer targets Ubuntu 22.04 Jammy + ROS Humble." >&2
  exit 1
fi

sudo apt update
sudo apt install -y \
  build-essential \
  cmake \
  curl \
  git \
  python3-colcon-common-extensions \
  python3-pip \
  python3-rosdep \
  python3-vcstool \
  ros-"${ROS_DISTRO}"-ament-cmake \
  ros-"${ROS_DISTRO}"-ament-cmake-python \
  ros-"${ROS_DISTRO}"-cv-bridge \
  ros-"${ROS_DISTRO}"-geometry-msgs \
  ros-"${ROS_DISTRO}"-image-transport \
  ros-"${ROS_DISTRO}"-launch \
  ros-"${ROS_DISTRO}"-launch-ros \
  ros-"${ROS_DISTRO}"-message-filters \
  ros-"${ROS_DISTRO}"-rclpy \
  ros-"${ROS_DISTRO}"-sensor-msgs \
  ros-"${ROS_DISTRO}"-std-msgs \
  ros-"${ROS_DISTRO}"-tf2-ros \
  librealsense2-dev \
  librealsense2-utils

if ! rosdep db >/dev/null 2>&1; then
  sudo rosdep init || true
  rosdep update
fi

python3 -m pip install --user -r requirement.txt

echo "Host dependencies installed for ROS ${ROS_DISTRO}."
