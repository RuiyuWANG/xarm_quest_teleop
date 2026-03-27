# CloudGripper Manipulation

CloudGripper real-world teleoperation, data collection, and policy evaluation with:
- xArm + gripper
- Meta Quest 2 via `quest2ros`
- Intel RealSense cameras
- ROS Noetic

This repo now includes two ROS packages under [`ros_link/`](./ros_link):
- `teleop_msgs`: stamped message definitions
- `cloudgripper_teleop`: ROS Python nodes/scripts used by this project

## 1. Prerequisites

- Ubuntu + ROS Noetic installed
- A catkin workspace (default expected path: `~/catkin_ws`)
- `xarm_ros`, `quest2ros`, `ros_tcp_endpoint`, and `realsense2_camera` available in your ROS environment
- Python 3

Optional: use [`setup_docker.sh`](./setup_docker.sh) inside your ROS Docker container for shell/dev tooling.

## 2. One-command installation

From repository root:

```bash
./install.sh
```

What this does:
1. Registers `ros_link/teleop_msgs` and `ros_link/cloudgripper_teleop` into your catkin workspace.
2. Installs base system tools (`python3-pip`, `python3-catkin-tools`, `python3-rosdep`).
3. Runs `rosdep install` for ROS dependencies.
4. Installs Python packages from `requirement.txt`.
5. Builds ROS packages: `teleop_msgs`, `cloudgripper_teleop`.

### Useful install options

```bash
./install.sh --catkin-ws /root/catkin_ws
./install.sh --skip-apt --skip-rosdep
```

- `--skip-*`: skip parts you already manage manually.
- ROS packages are symlinked (not copied) so ROS entrypoints stay aligned with `src/` logic.

## 3. Environment setup (every new shell)

```bash
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash
```

## 4. Hardware/network quick checklist

1. xArm connected, reachable from host (same network/subnet).
2. Quest 2 app running and data bridged through `quest2ros` / `ros_tcp_endpoint`.
3. RealSense devices connected and visible.
4. ROS topics/services available before running scripts.

## 5. Run workflows

### Teleoperation

```bash
python src/scripts/run_quest_xarm_teleop_sync.py
```

### Data collection

```bash
python src/scripts/run_data_collection.py
```

Choose dataset config via ROS param if needed:

```bash
python src/scripts/run_data_collection.py _dataset_json:=three_piece_toy_d1.json
```

### Policy evaluation

```bash
python src/scripts/run_policy_eval.py
```

With checkpoint override:

```bash
python src/scripts/run_policy_eval.py _model_ckpt:=/path/to/latest.ckpt
```

## 6. Repo structure (high level)

- `src/`: core Python implementation (teleop, robots, io, data_collection, eval)
- `ros_link/teleop_msgs/`: custom ROS messages
- `ros_link/cloudgripper_teleop/`: ROS wrapper nodes/scripts
- `config/`: dataset/task json configs
- `install.sh`: installation automation

## 7. Notes

- `ros_link/cloudgripper_teleop/scripts/run_quest_xarm_teleop_sync.py` is now a thin wrapper that reuses `src/scripts/run_quest_xarm_teleop_sync.py` to avoid duplicated logic.
- Generated Python cache files (`__pycache__`, `*.pyc`) are ignored and should not be committed.
