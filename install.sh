#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CATKIN_WS="${HOME}/catkin_ws"
SKIP_APT=0
SKIP_PIP=0
SKIP_ROSDEP=0
SKIP_CATKIN_BUILD=0

usage() {
  cat <<'EOF'
Usage: ./install.sh [options]

Options:
  --catkin-ws <path>     Target catkin workspace (default: ~/catkin_ws)
  --skip-apt             Skip apt package installation
  --skip-pip             Skip pip installation from requirement.txt
  --skip-rosdep          Skip rosdep install
  --skip-catkin-build    Skip catkin build
  -h, --help             Show this help

Examples:
  ./install.sh
  ./install.sh --catkin-ws /root/catkin_ws
  ./install.sh --skip-apt --skip-rosdep
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --catkin-ws)
      CATKIN_WS="$2"
      shift 2
      ;;
    --skip-apt)
      SKIP_APT=1
      shift
      ;;
    --skip-pip)
      SKIP_PIP=1
      shift
      ;;
    --skip-rosdep)
      SKIP_ROSDEP=1
      shift
      ;;
    --skip-catkin-build)
      SKIP_CATKIN_BUILD=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ ! -d "${REPO_ROOT}/ros_link/teleop_msgs" || ! -d "${REPO_ROOT}/ros_link/cloudgripper_teleop" ]]; then
  echo "Could not find ros_link packages under: ${REPO_ROOT}" >&2
  exit 1
fi

mkdir -p "${CATKIN_WS}/src"

if [[ ! -f "${CATKIN_WS}/src/CMakeLists.txt" ]]; then
  echo "[install] Initializing catkin workspace at ${CATKIN_WS}"
  (cd "${CATKIN_WS}/src" && catkin_init_workspace)
fi

link_pkg() {
  local src="$1"
  local dst="$2"
  rm -rf "${dst}"
  ln -s "${src}" "${dst}"
}

echo "[install] Symlinking ROS packages into ${CATKIN_WS}/src"
link_pkg "${REPO_ROOT}/ros_link/teleop_msgs" "${CATKIN_WS}/src/teleop_msgs"
link_pkg "${REPO_ROOT}/ros_link/cloudgripper_teleop" "${CATKIN_WS}/src/cloudgripper_teleop"

if [[ "${SKIP_APT}" -eq 0 ]]; then
  echo "[install] Installing system dependencies"
  sudo apt-get update
  sudo apt-get install -y python3-pip python3-catkin-tools python3-rosdep
fi

if [[ "${SKIP_ROSDEP}" -eq 0 ]]; then
  if [[ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]]; then
    echo "[install] Initializing rosdep"
    sudo rosdep init || true
  fi
  rosdep update
  rosdep install --from-paths "${CATKIN_WS}/src" --ignore-src -r -y
fi

if [[ "${SKIP_PIP}" -eq 0 ]]; then
  if [[ -f "${REPO_ROOT}/requirement.txt" ]]; then
    echo "[install] Installing Python dependencies from requirement.txt"
    python3 -m pip install -r "${REPO_ROOT}/requirement.txt"
  elif [[ -f "${REPO_ROOT}/requirements.txt" ]]; then
    echo "[install] Installing Python dependencies from requirements.txt"
    python3 -m pip install -r "${REPO_ROOT}/requirements.txt"
  else
    echo "[install] No requirement.txt/requirements.txt found, skipping pip install"
  fi
fi

if [[ "${SKIP_CATKIN_BUILD}" -eq 0 ]]; then
  echo "[install] Building ROS packages"
  (
    cd "${CATKIN_WS}"
    source /opt/ros/noetic/setup.bash
    catkin build teleop_msgs cloudgripper_teleop
  )
fi

echo
echo "[install] Completed."
echo "Source your workspace before running scripts:"
echo "  source /opt/ros/noetic/setup.bash"
echo "  source ${CATKIN_WS}/devel/setup.bash"
