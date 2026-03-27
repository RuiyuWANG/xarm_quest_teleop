# XArm ROS1 Quest Teleoperation

This repository supports UFactory xArm7 + xArm gripper teleoperation through Meta Quest 2, a data collection pipeline for VR human demonstrations, and policy evaluation stack for:
- xArm + gripper
- Meta Quest 2 (`quest2ros`)
- Intel RealSense
- ROS Noetic

The repo includes two ROS packages in `ros_link/`:
- `teleop_msgs`
- `cloudgripper_teleop`

Key features:
- VR teleoperation for xArm7 + xArm gripper
- Position-control teleoperation
- ROS1 + Docker support
- Plug-and-play data collection and policy evaluation pipeline
- Synchronized multi-camera support for both RGB and RGB-D  
  Note: synchronization is software-side and can still be challenging in multi-camera systems; hardware synchronization may be needed for best results.
- Automated camera calibration (intrinsics + extrinsics) for both third-person and eye-in-hand cameras

## 1. Prerequisites

- Ubuntu + ROS Noetic installed
- A catkin workspace (default expected path: `~/catkin_ws`)
- `xarm_ros`, `quest2ros`, `ros_tcp_endpoint`, and `realsense2_camera` available in your ROS environment
- Python 3

## 2. Quick start

```bash
./install.sh
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash
```

Then run (these scripts automatically launch required ROS nodes based on config, such as Quest bridge, xArm bringup, RealSense, and stamped wrapper nodes):

```bash
python src/scripts/run_quest_xarm_teleop_sync.py
python src/scripts/run_data_collection.py
python src/scripts/run_policy_eval.py
```

Camera calibration script:
```bash
python src/scripts/camera_calibration.py _camera_name:=d405 _setup:=eye_to_hand
```

Example for wrist camera calibration:
```bash
python src/scripts/camera_calibration.py _camera_name:=d435i_front _setup:=eye_in_hand
```

## 3. Installation automation

Use:

```bash
./install.sh
```

Useful options:

```bash
./install.sh --catkin-ws /root/catkin_ws
./install.sh --skip-apt --skip-rosdep
```

What it automates:
1. Symlinks `ros_link/teleop_msgs` and `ros_link/cloudgripper_teleop` into your catkin workspace.
2. Installs base system dependencies.
3. Runs `rosdep install`.
4. Installs Python dependencies from `requirement.txt`.
5. Builds `teleop_msgs` and `cloudgripper_teleop`.

## 4. Docker helpers

Docker startup commands are now kept in [`setup_docker.sh`](./setup_docker.sh):

```bash
./setup_docker.sh create     # one-time create persistent container
./setup_docker.sh start      # start + attach
./setup_docker.sh exec       # new shell in running container
./setup_docker.sh bootstrap  # install shell/dev tooling inside container
```

You can override defaults with env vars:
- `CONTAINER_NAME`
- `DOCKER_IMAGE`
- `CATKIN_MOUNT`

## 5. Where configs live

All runtime settings are config-driven:
- Dataset/task configs: `config/*.json`
- Teleop/data/eval python configs: `src/configs/*.py`
- ROS package-level settings/scripts: `ros_link/cloudgripper_teleop/`

If you need to change IPs, camera topics, launch commands, sync windows, or evaluation behavior, start in `src/configs/`.

### Common customizations

1. Robot IP / xArm launch arguments  
Edit [`src/configs/teleop_config.py`](/home/ruiyuw/Codes/CloudGripper_Manipulation/src/configs/teleop_config.py) in `ROBOT_LAUNCH_CMD` (e.g. `robot_ip:=...`, `add_gripper:=true`).

2. Quest bridge host IP and port  
Edit [`src/configs/teleop_config.py`](/home/ruiyuw/Codes/CloudGripper_Manipulation/src/configs/teleop_config.py) in `QUEST_LAUNCH_CMD` (e.g. `tcp_ip:=...`, `tcp_port:=10000`).

3. Camera serial numbers / camera launch setup  
Edit [`src/configs/collector_config.py`](/home/ruiyuw/Codes/CloudGripper_Manipulation/src/configs/collector_config.py) in `realsense_all_launch_cmds`, `realsense_light_launch_cmds`, and `cameras_all` / `cameras_light` topic mappings.

4. Data collection behavior (sync windows, save modes, keyboard controls)  
Edit [`src/configs/collector_config.py`](/home/ruiyuw/Codes/CloudGripper_Manipulation/src/configs/collector_config.py) (`CameraSyncConfig`, `CollectorConfig`, `RobotSyncConfig`).

5. Policy checkpoint + evaluation settings  
Edit [`src/configs/eval_config.py`](/home/ruiyuw/Codes/CloudGripper_Manipulation/src/configs/eval_config.py) for `model_ckpt_path`, horizons, control frequency, camera selection, rollout counts, and logging.

6. Task/dataset definitions  
Edit JSON task files under `config/` (for example `three_piece_toy_d1.json`) and pass with:
```bash
python src/scripts/run_data_collection.py _dataset_json:=three_piece_toy_d1.json
```

## 6. Repo map

- `src/`: core code (teleop, robots, IO, data collection, eval)
- `ros_link/teleop_msgs/`: custom stamped ROS messages
- `ros_link/cloudgripper_teleop/`: ROS wrapper nodes/scripts
- `config/`: task + dataset JSON configs
- `install.sh`: installation automation
- `setup_docker.sh`: docker lifecycle + container bootstrap helper

## 7. Optional: Full hardware + ROS setup tutorial

This section is intentionally detailed for first-time full hardware bring-up.

### Step A. Hardware

1. xArm setup
- Follow UFactory manual: https://www.ufactory.cc/wp-content/uploads/2023/05/xArm-User-Manual-V2.0.0.pdf
- Connect robot + gripper and power on.
- Put PC and xArm on the same LAN (example subnet `192.168.1.x`).
- Verify connection in UFactory Studio.
- Set initial pose / TCP payload / TCP offset correctly.

2. Meta Quest 2 setup
- Install Meta Horizon app and enable Developer Mode.
- Connect Quest and PC to the same network.
- Install and configure quest2ros app: https://quest2ros.github.io/

3. RealSense camera setup
- Connect cameras with USB 3.0 cables/ports.
- Check serial numbers:
```bash
rs-enumerate-devices
```

### Step B. ROS Docker

Before creating or starting the container, allow Docker to access your X display:
```bash
xhost +local:docker
```

On host:
```bash
./setup_docker.sh create
```

Later sessions:
```bash
./setup_docker.sh start
./setup_docker.sh exec
```

Inside container (one-time bootstrap):
```bash
./setup_docker.sh bootstrap
```

### Step C. Install ROS dependencies

Install/clone and build these packages in your ROS environment:
- `xarm_ros`: https://github.com/xArm-Developer/xarm_ros
- `quest2ros`: https://quest2ros.github.io/
- `ros_tcp_endpoint`: https://github.com/Unity-Technologies/ROS-TCP-Endpoint
- `realsense2_camera` (build from source as below)

Camera (RealSense) Will need to build from source, to get a 2.55 + librealsense for D405 camera

```bash
cd /root
git clone https://github.com/IntelRealSense/librealsense.git
cd librealsense
git checkout v2.56.3
mkdir -p build && cd build
cmake .. -DBUILD_EXAMPLES=false -DBUILD_GRAPHICAL_EXAMPLES=false
make -j"$(nproc)"
make install
ldconfig
```

And ROS1 version of realsense-ros

```bash
cd ~/catkin_ws/src
git clone https://github.com/IntelRealSense/realsense-ros.git
cd realsense-ros
git checkout noetic-development
```

### Step D. Test installation

Source ROS and workspace:
```bash
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash
```

Bring up and verify dependencies manually:
```bash
roscore
```

```bash
roslaunch ros_tcp_endpoint endpoint.launch tcp_ip:=<HOST_IP> tcp_port:=10000
rosrun quest2ros ros2quest.py
```

```bash
roslaunch xarm_bringup xarm7_server.launch robot_ip:=<ROBOT_IP> report_type:=dev add_gripper:=true
```

```bash
rostopic list
rostopic echo /xarm/xarm_states
rosservice call /xarm/move_joint "{pose: [0,0,0,0,0,0,0], mvvelo: 0.35, mvacc: 7, mvtime: 0, wait: 0}"
rosservice call /xarm/gripper_move 500
```

### Step E. End manual ROS nodes, then run CloudGripper workflows

First stop all manually started ROS nodes/processes from Step D.  
Then test teleop first:

```bash
python src/scripts/run_quest_xarm_teleop_sync.py
```

Then run:

```bash
python src/scripts/run_data_collection.py
```

The Python entry scripts auto-launch required ROS dependencies based on your config files.
