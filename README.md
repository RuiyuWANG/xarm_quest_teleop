# XArm Quest Teleop ROS1

ROS1 teleoperation, data collection, and policy evaluation for an xArm7 with a Meta Quest controller and RealSense cameras.

This branch is the ROS1 open-source release. The ROS2 migration work lives on the `ros2` branch and keeps a similar workflow shape, but uses ROS Humble, `ament`, and ROS2 launch files.

## Branches And ROS Versions

- `open_source`: ROS1 Noetic release for teleoperation, data collection, and policy evaluation.
- `teleop` / `main`: historical ROS1 development branches.
- `ros2`: ROS2 Humble migration branch.

Keep ROS1 and ROS2 workspaces separate. This branch is intended for ROS1 Noetic and catkin.

## Hardware

Default hardware assumptions:

- xArm7 with xArm gripper.
- Meta Quest 2 running the Quest2ROS app.
- Intel RealSense D405, default serial `230322271104`.
- Intel RealSense D435i front camera, default serial `335522071488`.
- Optional Intel RealSense D435i shoulder camera, default serial `233522073481`.

## Install

### Normal Host Install

Use Ubuntu 20.04 with ROS Noetic.

```bash
./install.sh
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash
```

`install.sh` installs ROS/Python dependencies, creates or reuses a catkin workspace, links this repo into `~/catkin_ws/src`, and builds the local ROS wrapper packages.

The host setup expects these ROS1 dependencies in the catkin workspace:

- `xarm_ros`
- `quest2ros`
- `ROS-TCP-Endpoint`
- `realsense-ros`

### Docker Install

Docker is optional for this ROS1 branch.

```bash
xhost +local:docker
./setup_docker.sh create
./setup_docker.sh start
./setup_docker.sh exec
./install.sh --catkin-ws /root/catkin_ws
source /opt/ros/noetic/setup.bash
source /root/catkin_ws/devel/setup.bash
```

On hosts without NVIDIA Docker support, create the container without GPU flags:

```bash
DOCKER_GPU_ARGS="" ./setup_docker.sh create
```

Common Docker commands:

```bash
./setup_docker.sh help
./setup_docker.sh exec
./setup_docker.sh stop
```

## Build Checks

From a sourced ROS1 shell:

```bash
catkin build teleop_msgs cloudgripper_teleop
rosmsg show teleop_msgs/OVR2ROSInputsStamped
rosmsg show teleop_msgs/RobotMsgStamped
rosrun cloudgripper_teleop quest_stamped_node.py
```

Useful runtime checks:

```bash
rostopic list | grep q2r
rostopic echo /xarm/xarm_states
rosservice list | grep /xarm
```

## Important Configuration

Most users only need these files:

- `src/configs/teleop_config.py`: robot IP launch command, Quest launch command, active hand, deadman button, movement scaling, roll lock, and servo limits.
- `src/configs/collector_config.py`: RealSense launch commands, camera topic groups, light/full sync mode, and sync timing.
- `src/configs/eval_config.py`: checkpoint path, rollout timing, action units, gripper settings, logging, and camera sync.
- `config/*.json`: task-level dataset settings such as task name, output directory, number of demos, collection frequency, and calibration file.

For a new setup, start by changing the xArm IP in `TeleopConfig.ROBOT_LAUNCH_CMD`, then update RealSense serial numbers in `CollectorConfig`.

## Bringup

The teleop and collection scripts can launch the required ROS nodes automatically from their config commands. For manual bringup and debugging, use separate terminals.

Start ROS and the Quest bridge:

```bash
roscore
roslaunch ros_tcp_endpoint endpoint.launch tcp_ip:=<HOST_IP> tcp_port:=10000
rosrun quest2ros ros2quest.py
rosrun cloudgripper_teleop quest_stamped_node.py
```

Start the xArm driver:

```bash
roslaunch xarm_bringup xarm7_server.launch robot_ip:=<ROBOT_IP> report_type:=dev add_gripper:=true
```

Check the expected robot state and service interfaces:

```bash
rostopic echo /xarm/xarm_states
rosservice list | grep /xarm
```

## Teleoperation

Run synchronized Quest-to-xArm teleoperation:

```bash
python src/scripts/run_quest_xarm_teleop_sync.py
```

The node waits for stamped Quest pose/input topics, xArm state, and xArm services before enabling control. By default, roll is locked for safer Cartesian teleop; adjust `lock_roll` and `locked_roll_rad` in `src/configs/teleop_config.py` when a task needs full orientation control.

## Data Collection

Collect demonstrations with the synchronized teleop collector:

```bash
python src/scripts/run_data_collection.py _dataset_json:=three_piece_toy_d1.json
```

Keyboard controls:

- `c`: start or continue recording.
- `s`: save the current episode.
- `d`: delete the current episode.
- `q`: quit.

Raw episodes are written under:

```text
data/<task>/
  task_meta.json
  light/episode_*/
  all_sensors/episode_*/
```

Collection modes are controlled in `src/configs/collector_config.py`:

- Light mode records the configured RGB cameras in `cameras_light`.
- Full mode records the configured RGB-D cameras in `cameras_all`.

The stored action contract is absolute robot action:

```text
[x_mm, y_mm, z_mm, rot6d_first_two_columns, gripper]
```

## Multi-Camera Setup

List connected cameras:

```bash
rs-enumerate-devices
```

Then update the serial numbers in the RealSense launch commands in `src/configs/collector_config.py`.

Default topic names expected by the collector and evaluator:

- `/d405/color/image_raw`
- `/d435i_front/color/image_raw`
- `/d435i_front/depth/image_rect_raw`
- `/d435i_shoulder/color/image_raw`

Manual RealSense launch example:

```bash
roslaunch realsense2_camera rs_camera.launch \
  serial_no:=230322271104 \
  camera:=d405 \
  enable_color:=true \
  enable_depth:=false
```

For multiple cameras, launch one driver instance per camera name and serial number.

## Calibration

Run calibration utilities from a sourced environment:

```bash
python src/scripts/camera_calibration.py _camera_name:=d405 _setup:=eye_to_hand
python src/scripts/camera_calibration.py _camera_name:=d435i_front _setup:=eye_in_hand
```

Keep task JSON calibration paths compatible with existing datasets and cache conversion tools.

## Policy Evaluation

Run policy evaluation:

```bash
python src/scripts/run_policy_eval.py
```

Override the checkpoint or skip Quest launch from ROS params:

```bash
python src/scripts/run_policy_eval.py _model_ckpt:=/path/to/model.ckpt _launch_quest:=false
```

Keyboard controls:

- `c`: start or continue rollout.
- `p`: pause.
- `r`: reset episode.
- `s`: mark success.
- `f`: mark failure.
- `q`: quit.

Policy evaluation reads camera observations, builds temporal RGB observations, predicts action chunks, and executes absolute xArm actions. If a policy outputs XYZ in meters, set `xyz_unit = "m"` in `src/configs/eval_config.py`; otherwise the default is millimeters.

## Add A Policy

Add a new policy behind the same small interface used by `src/eval/eval_runner.py`:

- Load the model in a policy module under `src/policy/`.
- Expose an inference method that accepts the current observation dict.
- Return action chunks with shape `[T, action_dim]`.
- Keep actions compatible with `[xyz_mm, rot6d, gripper]`, or set `xyz_unit` in eval config when conversion is needed.

Use `src/policy/seeker_policy.py` as the current reference implementation.

## Custom Data Conversion

Raw collection data is intentionally simple: synchronized images, low-dimensional robot state, actions, and task metadata. For a custom converter:

- Read task metadata from `data/<task>/task_meta.json`.
- Load episode folders from `light/` or `all_sensors/`.
- Preserve timestamps when resampling observations and actions.
- Keep the action convention absolute unless the downstream policy explicitly documents another format.
- Write converted caches outside the raw dataset directory so raw demonstrations stay immutable.

## Repo Map

```text
config/                         Task and dataset JSON files
ros_link/cloudgripper_teleop/    ROS1 wrapper nodes for stamped Quest messages
ros_link/teleop_msgs/            ROS1 custom stamped message definitions
src/configs/                     Runtime Python config objects
src/data_collection/             Dataset collector and writers
src/eval/                        Policy rollout runner
src/policy/                      Policy wrappers
src/robots/                      xArm robot adapter
src/scripts/                     CLI entry points
src/teleop/                      Quest-to-xArm teleop loop
src/utils/                       Shared transforms, ROS helpers, and utilities
```
