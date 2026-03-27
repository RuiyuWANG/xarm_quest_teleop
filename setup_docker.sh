#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-ros1_noetic}"
DOCKER_IMAGE="${DOCKER_IMAGE:-ros:noetic}"
CATKIN_MOUNT="${CATKIN_MOUNT:-$HOME/docker_shared/catkin_ws}"

usage() {
  cat <<EOF
Usage: ./setup_docker.sh <command>

Commands:
  create     Create a persistent ROS1 Noetic container with /dev and X11 access
  start      Start and attach to the container
  exec       Open a new shell in a running container
  bootstrap  Install shell/dev tooling inside the container (run as root in container)
  help       Show this message

Environment overrides:
  CONTAINER_NAME   (default: ros1_noetic)
  DOCKER_IMAGE     (default: ros:noetic)
  CATKIN_MOUNT     (default: \$HOME/docker_shared/catkin_ws)
EOF
}

create_container() {
  mkdir -p "${CATKIN_MOUNT}"
  xhost +local:docker

  docker run -it \
    --name "${CONTAINER_NAME}" \
    --network host \
    --privileged \
    --gpus all \
    -v /dev:/dev \
    -v /run/udev:/run/udev:ro \
    -v "${CATKIN_MOUNT}:/root/catkin_ws" \
    -e DISPLAY="${DISPLAY}" \
    -e QT_X11_NO_MITSHM=1 \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    "${DOCKER_IMAGE}" \
    bash
}

start_container() {
  xhost +local:docker
  docker start -ai "${CONTAINER_NAME}"
}

exec_container() {
  docker exec -it "${CONTAINER_NAME}" zsh
}

bootstrap_container() {
  echo "[bootstrap] Installing system packages..."
  apt-get update
  apt-get install -y \
    curl wget git build-essential \
    tmux htop vim nano \
    unzip zip \
    ca-certificates \
    software-properties-common \
    libgl1-mesa-glx libglfw3 libosmesa6-dev patchelf \
    zsh

  echo "[bootstrap] Installing Miniconda..."
  CONDA_DIR=/opt/conda
  if [[ ! -d "${CONDA_DIR}" ]]; then
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "${CONDA_DIR}"
    rm -f /tmp/miniconda.sh
  fi

  "${CONDA_DIR}/bin/conda" init bash
  "${CONDA_DIR}/bin/conda" init zsh

  echo "[bootstrap] Installing Oh My Zsh..."
  export RUNZSH=no
  export CHSH=no
  sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"

  echo "[bootstrap] Installing Zsh plugins..."
  ZSH_CUSTOM="${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}"
  git clone https://github.com/zsh-users/zsh-autosuggestions "${ZSH_CUSTOM}/plugins/zsh-autosuggestions" || true
  git clone https://github.com/zsh-users/zsh-syntax-highlighting "${ZSH_CUSTOM}/plugins/zsh-syntax-highlighting" || true
  git clone https://github.com/agkozak/zsh-z.git "${HOME}/.zsh/zsh-z" || true

  echo "[bootstrap] Configuring ~/.zshrc..."
  sed -i 's/^plugins=.*/plugins=(git conda zsh-autosuggestions zsh-syntax-highlighting)/' ~/.zshrc || true

  chsh -s /usr/bin/zsh root
  echo "[bootstrap] Completed. Restart shell with: exec zsh"
}

cmd="${1:-help}"
case "${cmd}" in
  create) create_container ;;
  start) start_container ;;
  exec) exec_container ;;
  bootstrap) bootstrap_container ;;
  help|-h|--help) usage ;;
  *)
    echo "Unknown command: ${cmd}" >&2
    usage
    exit 1
    ;;
esac
