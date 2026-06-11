# VR Teleop Hardware
1. Set up the robot

   Ufactory xArm, please refer to the official manual, https://www.ufactory.cc/wp-content/uploads/2023/05/xArm-User-Manual-V2.0.0.pdf
   1. Set up the robot, connect it to the adapter, and then to the electrical source. Connect the gripper.
   2. Connect your computer and robot to the same router. Set PC IPV4 ip to be in 192.168.1.0 - 192.168.1.255.
   Get PC access to internet with Wifi, perferrably seted by another router.
   3. Test your connection at UFactory Studio, IP:18333.
   4. Reset the initial position, TCP payload, TCP height offset (see manual). These are important for adjusting IK sover and safety constraints.

2. Set up Meta Quest 2

   1. Download Meta Horizon app on your mobile phone. Login with email. Pay attention to enter true age when register, otherwise some repos are not accessible to people under 18.
   2. Reboot Quest to sign in as owner. Press open and volume - at the same time. Connect Quest to the same WiFi of PC, and connect to app.
   3. In app, enable developer mode.
   4. Check Quest network IP and ping IP on PC to check connectivity.

3. Setup cameras.

   Connect the cameras to the PC, perferrably one to the motherboard and one to the front panel. Connect with USE 3.0 wire & port if it is necessary.

# ROS Set-up
1. Docker
   1. Start a persistent ROS1 docker container with full /dev access for USB and RealSense connection, X11 enabled for RViz, --gpus all to allow docker access to GPU
   ```
   docker run -it \
   --name ros1_noetic \
   --network host \
   --privileged \
   --gpus all \
   -v /dev:/dev \
   -v /run/udev:/run/udev:ro \
   -v ~/docker_shared/catkin_ws:/root/catkin_ws \
   -e DISPLAY=$DISPLAY \
   -e QT_X11_NO_MITSHM=1 \
   -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
   ros:noetic \
   bash -c "apt update && apt install -y zsh && chsh -s /usr/bin/zsh && zsh"
   ```

   2. Start container
   Allow the docker access to display (IMPORTANT for visualization like rviz and rqt)
   ```
   xhost +local:docker

   ```
   ```
   sudo docker start -ai ros1_noetic
   ```

   open new shell in the running docker
   ```
   sudo docker exec -it ros1_noetic zsh
   ```
   
   source:
   ```
   source /opt/ros/noetic/setup.zsh 2>/dev/null || source /opt/ros/noetic/setup.bash
   source ~/catkin_ws/devel/setup.zsh 2>/dev/null || source ~/catkin_ws/devel/setup.bash
   ```

2. Quest
   Follow:
   ```
   roslaunch ros_tcp_endpoint endpoint.launch tcp_ip:=192.168.0.182 tcp_port:=10000
   ```

   ```
   rosrun quest2ros ros2quest.py
   ```

   check connection
   ```
   rostopic list
   ```

3. Robot
   Follow:

   Test robot connection:
   ```
   roslaunch xarm_bringup xarm7_server.launch robot_ip:=192.168.1.241 report_type:=dev add_gripper:=true
   ```

   check Topic Register:
   ```
   rostopic list
   ```

   check connection
   ```
   rostopic echo /xarm/xarm_states
   ```

   check robot movement
   ```
   rosservice call /xarm/move_joint "{pose: [0,0,0,0,0,0,0], mvvelo: 0.35, mvacc: 7, mvtime: 0, wait: 0}"
   ```

   check gripper
   ```
   rosservice call /xarm/gripper_move 500

   ```

4. Camera (RealSense)
   Will need to build from source, to get a 2.55 + librealsense for D405 camera
   ```
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
   ```
   cd ~/catkin_ws/src
   git clone https://github.com/IntelRealSense/realsense-ros.git
   cd realsense-ros
   git checkout noetic-development
   ```
   Launch camera with name to differentiate cameras, replace it with your serial numbers

   ```
   # D405
   roslaunch realsense2_camera rs_camera.launch \
   serial_no:=230322271104 camera:=d405 color_width:=848 color_height:=480 color_fps:=30 align_depth:=True enable_sync:=True 

   # D435i front
   roslaunch realsense2_camera rs_camera.launch \
   serial_no:=335522071488 camera:=d435i_front color_width:=848 color_height:=480 color_fps:=30 align_depth:=True enable_sync:=True 

   # D435i shoulder
   roslaunch realsense2_camera rs_camera.launch \
   serial_no:=233522073481 camera:=d435i_shoulder color_width:=848 color_height:=480 color_fps:=30 align_depth:=True enable_sync:=True 
   ```

   Check serial numbers
   ```
   rs-enumerate-devices
   ```

   [Optional] Get RViz for visualization.
   ```
   apt update
   apt install -y ros-noetic-rviz
   ```


# Install Teleop
1. Warp robot and quest info to get timestamped topic for sychronization, build teleop_msgs
   ```
   cd ~/catkin_ws
   catkin build teleop_msgs
   source ~/catkin_ws/devel/setup.zsh

   ```

   Test 
   ```
   rosmsg show teleop_msgs/OVR2ROSInputsStamped
   rosmsg show teleop_msgs/RobotMsgStamped
   ```

2. Add scripts to launch nodes for timestamped topics, build cloudgripper_teleop
   ```
   chmod +x ~/catkin_ws/src/cloudgripper_teleop/scripts/robot_state_stamp_node.py
   cd ~/catkin_ws
   catkin build cloudgripper_teleop
   source ~/catkin_ws/devel/setup.zsh
   ```


# Run Teleop / Data collection / Policy eval
1. Initialize docker, source
   ```
   sudo docker start -ai ros1_noetic
   ```
2. Start ROS core
   ```
   roscore
   ```
3. Run Scripts

   The sections above are the hardware and ROS interface reference. Keep them as
   the source of truth for Docker, Quest, xArm, RealSense, and ROS topic/service
   bringup. The commands below are the current RGB-only real experiment loop.

   When launching Python `rospy` scripts directly, pass private params as
   `_param:=value`.

## Current Real Experiment Loop

### 1. Collect Demonstrations

From the ROS container, after sourcing ROS and the catkin workspace:

```bash
cd /root/catkin_ws/src/real_ws/CloudGripper_Manipulation
python3 src/scripts/run_data_collection.py _dataset_json:=cleanup_table_d2.json
```

This writes raw collector data under:

```text
/root/catkin_ws/src/real_ws/data/cleanup_table_d2/
```

Key files/directories:

- `task_meta.json`: collection metadata and calibration snapshot
- `rgb/episode_*/`: RGB-only episodes used by the current real policy path
- `all_sensors/episode_*/`: optional full sensor stream, currently not used by eval

### 2. Convert Raw Data To Cache

```bash
cd /root/catkin_ws/src/real_ws
python3 CloudGripper_Manipulation/src/scripts/convert_raw_data_to_cache.py --task-dir data/cleanup_table_d2 --out-dir data/cleanup_table_d2/rgb_lmdb --stream rgb --overwrite
```

The current cache contract is:

```text
data/cleanup_table_d2/rgb_lmdb/
  images.lmdb
  arrays.npz
  meta.json
  build_done.flag
```

Current action convention:

```text
action/absolute_action: [x_mm,y_mm,z_mm,rot6d_first_two_columns,gripper]
```

Delta actions are intentionally disabled in the converter and are not part of
the current real rollout path.

### 3. Train RVT2 Visual Focus

```bash
cd /root/catkin_ws/src/real_ws/seeker-dev
PYTHONPATH=$PWD seeker train --config-name=train_visual_focus_rvt2_real task_name=cleanup_table_d2 dataset_path=../data/cleanup_table_d2/rgb_lmdb seed=0 background_overlay.enabled=true background_overlay.background_path=.weights/backgrounds_224.pt
```

The RVT2 focus checkpoint is written to:

```text
seeker-dev/experiments/cleanup_table_d2/rvt2_real/demos-all_seed-0/latest.pt
```

Copy it to the path expected by the real RVT2 policy config:

```bash
cp experiments/cleanup_table_d2/rvt2_real/demos-all_seed-0/latest.pt .weights/rvt2_real_cleanup_table_d2.pt
```

### 4. Train Real Policy

```bash
cd /root/catkin_ws/src/real_ws/seeker-dev
PYTHONPATH=$PWD seeker train --config-name=train_focus_policy_rvt2_real task_name=cleanup_table_d2 dataset_path=../data/cleanup_table_d2/rgb_lmdb seed=0
```

The default policy checkpoint path is:

```text
seeker-dev/experiments/cleanup_table_d2/real_policy/real_rvt2_cleanup_table_d2_seed_0/checkpoints/latest.ckpt
```

### 5. Dry-Run Eval With Live Viz

Start with no actuation:

```bash
cd /root/catkin_ws/src/real_ws/CloudGripper_Manipulation
python3 src/scripts/run_policy_eval.py
```

Press `c` to start inference. The dry-run config enables live inference
visualization with `live_viz: true`. Disable that field when running headless.
Eval outputs are grouped by `eval_name`, for example
`evaluation/cleanup_table_d2/real_rvt2_dry_run/in-domain/`.

### 6. Real Rollout

After dry-run overlays and action sanity checks look correct, run with actuation:

```bash
cd /root/catkin_ws/src/real_ws/CloudGripper_Manipulation
python3 src/scripts/run_policy_eval.py _config:=cleanup_table_d2_rollout.yaml
```

Eval configs are resolved from `config/eval/` by default. The default config is
`cleanup_table_d2_dry_run.yaml`, so the bare command is safe and does not
actuate. Edit the eval YAML to switch checkpoints, task names, model names,
calibration files, rollout count, or logging location. `run_policy_eval.py`
only reads `_config:=...` from the terminal; individual eval fields should live
in the YAML.

Eval profiles:

- `profile: dry_run`: no actuation, debug overlays on, recording off
- `profile: rollout`: actuation on, debug overlays on, recording on
- `profile: manual`: use explicit YAML values without profile defaults

Keyboard controls:

- `c`: start/continue
- `p`: pause
- `r`: reset
- `s`: mark success
- `f`: mark fail
- `q`: quit

## Runtime Ownership

- The existing ROS setup sections own Docker, Quest, xArm, RealSense, topics,
  services, and manual bringup checks.
- Current collection/eval scripts auto-launch RGB-only cameras from
  `src/configs/collector_config.py` unless auto-launch is disabled.
- `CloudGripper_Manipulation/config/*.json` seeds task collection metadata.
- `CloudGripper_Manipulation/all_cams_calib.json` is the preferred eval/debug
  projection calibration source.
- The policy checkpoint owns policy architecture, image size, observation
  horizon, action horizon, task embedding, and focus source.
- Real eval owns ROS topics, camera launch, robot sync, safety thresholds,
  interpolation, gripper behavior, logging, and debug overlays.
