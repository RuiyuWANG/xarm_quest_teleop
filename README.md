# xArm Quest Teleop ROS2

xArm Quest Teleop is a ROS2 Humble pipeline for VR teleoperation, real-robot demonstration collection, cache conversion, and policy evaluation on a UFactory xArm7 with RealSense cameras.

This branch is ROS2-only. ROS1/Noetic and Docker instructions are preserved in the older `teleop`/`main` history, not in this branch.

## Hardware

- UFactory xArm7 with xArm gripper.
- Meta Quest 2 running the Quest2ROS app: https://mcwelle.com/q2r/
- Intel RealSense cameras:
  - D405 wrist camera, default serial `230322271104`.
  - D435i front camera, default serial `335522071488`.
  - Optional D435i shoulder camera, default serial `233522073481`.

## Install

```bash
cd ~/ros2_ws/src/xarm_quest_teleop
./scripts/install_ros2_host.sh
./scripts/fetch_ros2_deps.sh
./scripts/build_ros2_ws.sh
source ~/ros2_ws/install/setup.bash
```

The dependency script fetches:

- `Quest2ROS/quest2ros`, branch `ros2`
- `MWelle77/ROS-TCP-Endpoint`, branch `main-ros2`
- `xArm-Developer/xarm_ros2`, branch `humble`
- `realsenseai/realsense-ros`, branch `ros2-master`

## Build Checks

```bash
colcon test --packages-select xarm_quest_teleop_msgs xarm_quest_teleop
ros2 interface show xarm_quest_teleop_msgs/msg/OVR2ROSInputsStamped
./scripts/check_ros2_stack.sh
```

## Important Configuration

Most hardware settings are in `ros2_packages/xarm_quest_teleop/config/ros2/hardware.yaml`.

Edit these first:

- `robot.robot_ip`: xArm controller IP.
- `quest.tcp_ip`: host IP visible to the Quest app.
- `quest.tcp_port`: ROS-TCP endpoint port, normally `10000`.
- `cameras.<name>.serial_no`: RealSense serial numbers from `rs-enumerate-devices`.
- `profiles.rgb.cameras`: cameras launched for teleop, collection, and eval RGB streams.
- `profiles.all.cameras`: all configured cameras for RGB-D collection.

Runtime behavior lives in Python dataclasses:

- `xarm_quest_teleop/configs/teleop_config.py`: active hand, deadman/reset buttons, pose scaling, gripper mapping, servo rate, safety clamps.
- `xarm_quest_teleop/configs/collector_config.py`: RGB vs RGB-D sync, camera topic maps, queue sizes, collection keys.
- `xarm_quest_teleop/configs/eval_config.py`: policy name, checkpoint path, rollout profile, action safety, logging, observation cameras.

Installed package configs are also available through the ROS2 share path after build, so launch files can load them without depending on the source tree location.

## Bringup

Start the normal RGB stack:

```bash
ros2 launch xarm_quest_teleop xarm_quest_teleop_bringup.launch.py camera_profile:=rgb
```

Launch all configured cameras:

```bash
ros2 launch xarm_quest_teleop xarm_quest_teleop_bringup.launch.py camera_profile:=all
```

Useful checks:

```bash
ros2 topic list
ros2 topic echo /xarm/robot_states
ros2 service list | grep /xarm
ros2 topic hz /q2r_right_hand_pose_stamped
ros2 topic hz /d405/color/image_raw
ros2 topic hz /d435i_front/color/image_raw
```

The xArm driver is launched with `config/ros2/xarm_api_services.yaml`, which enables the ROS2 services used by xArm Quest Teleop: servo Cartesian control, gripper position, gripper state, and gripper setup.

## Teleoperation

Teleoperation streams Quest pose and button input into xArm Cartesian servo commands. It does not save demonstrations.

The Quest app publishes unstamped `/q2r_*` topics. xArm Quest Teleop stamps pose, twist, and input streams with:

```bash
ros2 run xarm_quest_teleop quest_stamped_node
```

For the full teleop-only loop:

```bash
ros2 run xarm_quest_teleop run_quest_xarm_teleop_sync
```

Important teleop settings:

- `TeleopConfig.active_hand`: `right` or `left`.
- `TeleopConfig.require_deadman`: require a held button before moving.
- `TeleopConfig.deadman_field`: default `button_lower`.
- `TeleopConfig.pos_scale` and `rot_scale`: Quest motion gain.
- `TeleopConfig.lock_roll` and `locked_roll_rad`: optional fixed roll angle for the tool.
- `TeleopConfig.servo_max_step_mm` and `servo_max_step_rot_rad`: per-command safety clamps.
- `TeleopConfig.servo_rate_hz`: fixed-rate servo stream.
- `TeleopConfig.grip_close_from_index` and `grip_open_from_middle`: trigger-to-gripper mapping.

Startup waits for:

- `/q2r_<hand>_pose_stamped`
- `/q2r_<hand>_inputs_stamped`
- `/xarm/robot_states`
- xArm mode/state, servo Cartesian, gripper move, and gripper state services.

## Data Collection

Data collection runs the teleop loop and writes synchronized demonstrations.

```bash
ros2 run xarm_quest_teleop run_data_collection --ros-args -p dataset_json:=cleanup_table_d2.json
```

Dataset/task JSON files live in `ros2_packages/xarm_quest_teleop/config/`. The important fields are:

- `dataset.root_dir`: root output directory, usually `../../data` relative to package share/source config.
- `dataset.task`: task folder name.
- `dataset.num_demos`: target number of episodes.
- `dataset.collection_freq_hz`: fixed-rate sampling frequency.
- `dataset.demo_id_start`: first episode id.
- `calibration`: camera intrinsics/extrinsics used by downstream conversion and visualization.

Keyboard controls:

- `c`: start collection for the current episode.
- `s`: save the current episode.
- `d`: delete the current episode.
- `q`: quit.

Raw data is written under the task root from the dataset JSON:

```text
data/<task>/
  task_meta.json
  rgb/episode_*/
  all_sensors/episode_*/
```

Collection modes:

- RGB-only mode: `CollectorConfig.enable_rgb_sync = True`, `enable_full_sync = False`. Uses `CameraSyncConfig.cameras_rgb`.
- RGB-D mode: `CollectorConfig.enable_full_sync = True`. Uses `CameraSyncConfig.cameras_all` and saves depth when enabled.

Actions remain absolute:

```text
action/absolute_action: [x_mm,y_mm,z_mm,rot6d_first_two_columns,gripper]
```

## Multi-Camera Setup

1. Find camera serial numbers:

```bash
rs-enumerate-devices
```

2. Edit `ros2_packages/xarm_quest_teleop/config/ros2/hardware.yaml`.

Each camera needs:

- `serial_no`: exact device serial.
- `camera_name`: topic root, such as `d405` or `d435i_front`.
- `camera_namespace`: keep empty to preserve topics like `/d405/color/image_raw`.
- `rgb_camera.color_profile`: `width,height,fps`.
- `depth_module.depth_profile`: `width,height,fps`.
- `enable_depth`: `true` for RGB-D cameras, `false` for RGB-only wrist RGB.
- `enable_sync`, `align_depth.enable`, `initial_reset`: RealSense launch behavior.

3. Choose launch profiles:

```yaml
profiles:
  rgb:
    cameras: ["d405", "d435i_front"]
  all:
    cameras: ["d405", "d435i_front", "d435i_shoulder"]
```

4. Launch and verify:

```bash
ros2 launch xarm_quest_teleop xarm_quest_teleop_bringup.launch.py camera_profile:=all
ros2 topic hz /d405/color/image_raw
ros2 topic hz /d435i_front/color/image_raw
ros2 topic hz /d435i_front/depth/image_rect_raw
ros2 topic hz /d435i_shoulder/color/image_raw
```

5. Match collection topic maps if camera names change.

Update `CameraSyncConfig.cameras_rgb` and `CameraSyncConfig.cameras_all` in `collector_config.py` if you change camera names or topic shapes. The default topic shape is:

```text
/<camera_name>/color/image_raw
/<camera_name>/depth/image_rect_raw
```

For best reliability, put high-bandwidth cameras on separate USB controllers when possible. If frames drop or timestamps drift, lower FPS/resolution, disable unused streams, or collect RGB-only first.

## Cache Conversion

The RGB cache contract is:

```text
rgb_lmdb/
  images.lmdb
  arrays.npz
  meta.json
  build_done.flag
```

Convert raw demonstrations:

```bash
ros2 run xarm_quest_teleop convert_raw_data_to_cache \
  --task-dir data/cleanup_table_d2 \
  --out-dir data/cleanup_table_d2/rgb_lmdb \
  --stream rgb \
  --overwrite
```

Customized conversion:

```bash
ros2 run xarm_quest_teleop convert_raw_data_to_cache \
  --task-dir data/cleanup_table_d2 \
  --out-dir data/cleanup_table_d2/rgb_lmdb_debug \
  --stream rgb \
  --cams d405 d435i_front \
  --n-demo 3 \
  --start-index 0 \
  --image-size 256 \
  --jpeg-quality 95 \
  --trans-thresh-mm 3.0 \
  --rot-thresh-rad 0.02 \
  --grip-thresh 2.0 \
  --image-workers 8 \
  --lmdb-map-size-gb 16 \
  --overwrite
```

Useful conversion options:

- `--stream rgb` converts `data/<task>/rgb`; `--stream all_sensors` converts RGB frames from the RGB-D stream.
- `--cams ...` selects cameras. Camera names must exist in `REAL_RGB_TO_POLICY_KEY` in `xarm_quest_teleop/policy/seeker_preprocessing.py`.
- `--n-demo` and `--start-index` convert a smaller subset for fast debugging.
- Still-frame filtering is enabled by default. Tune `--trans-thresh-mm`, `--rot-thresh-rad`, and `--grip-thresh`, or pass `--no-filter-still-frames` to keep every frame.
- `--image-size` and `--jpeg-quality` control policy image preprocessing and LMDB storage size.
- `--image-workers`, `--lmdb-map-size-gb`, and `--commit-every` are throughput/storage knobs for large datasets.

When adding a custom converter, keep this output contract unless the policy loader is changed at the same time:

```text
<cache_dir>/
  images.lmdb
  arrays.npz
  meta.json
  build_done.flag
```

Required cache contents:

- LMDB image keys from `REAL_RGB_TO_POLICY_KEY`, for example `agentview_image/00000000` and `robot0_eye_in_hand_image/00000000`.
- `arrays.npz` lowdim keys: `lowdim/robot0_eef_pos`, `lowdim/robot0_eef_rot`, `lowdim/robot0_gripper_qpos`, `lowdim/joint_states`, `lowdim/robot0_joint_vel`, and `lowdim/timestamp`.
- `arrays.npz` action key: `action/absolute_action` with shape `[N, 10]` and convention `[x_mm,y_mm,z_mm,rot6d_first_two_columns,gripper]`.
- `meta.json` with `episode_lengths`, `camera_names`, `rgb_keys`, `action_convention`, `image_size`, `n_demo`, and `n_samples`.

Custom camera/model conversion usually touches these files:

- `xarm_quest_teleop/policy/seeker_preprocessing.py`: camera-to-policy key map, task instruction map, policy image size, live observation preprocessing.
- `xarm_quest_teleop/utils/conversion_utils.py`: camera crop rules, intrinsics adjustment, rotation/action conversion helpers.
- `xarm_quest_teleop/scripts/convert_raw_data_to_cache.py`: episode validation, frame filtering, LMDB writing, `arrays.npz`, and `meta.json`.

## Policy Evaluation

Dry-run eval is the default profile and does not actuate:

```bash
ros2 run xarm_quest_teleop run_policy_eval --ros-args -p config:=cleanup_table_d2_dry_run.yaml
```

Real rollout:

```bash
ros2 run xarm_quest_teleop run_policy_eval --ros-args -p config:=cleanup_table_d2_rollout.yaml
```

Keyboard controls:

- `c`: start/continue
- `p`: pause
- `r`: reset
- `s`: mark success
- `f`: mark fail
- `q`: quit

Important eval config keys:

- `profile`: `dry_run`, `rollout`, or `manual`.
- `policy`: policy registry name.
- `ckpt`: model checkpoint path.
- `task`: task name used by logs and policy setup.
- `name`: model/run label.
- `eval_name`: condition label.
- `calib`: calibration JSON path.
- `result_log_dir`: output directory.
- `n_rollouts`, `horizon`, `exec_horizon`: rollout limits and action chunking.
- `exec_start_offset`: first predicted action index to execute from each policy chunk.
- `xyz_unit`: policy action position unit, `mm` by default and `m` when the policy outputs meters.
- `gripper_command_eps`: minimum non-binary gripper target change before sending another gripper service call.
- `live_viz`: enable projection/attention overlays when supported.

## Adding A Policy To Eval

Policies implement:

```python
class MyPolicy:
    n_obs_steps = 2
    n_action_steps = 8
    task_name = "cleanup_table_d2"
    last_visual_focus_records = []

    def infer_action(self, obs):
        ...
```

`infer_action(obs)` must return absolute actions shaped like:

```text
[x_mm, y_mm, z_mm, rot6d_first_two_columns, gripper]
```

There are two reliable ways to make a policy available to `run_policy_eval`.

In-repo path:

1. Add the policy implementation, for example `xarm_quest_teleop/policy/my_policy.py`.
2. Add a built-in branch in `xarm_quest_teleop/policy/registry.py`:

```python
if name == "my_policy":
    from xarm_quest_teleop.policy.my_policy import MyPolicy

    return MyPolicy
```

Package/plugin path:

```toml
[project.entry-points."xarm_quest_teleop.policies"]
my_policy = "my_policy_pkg.policy:MyPolicy"
```

Then use it in an eval YAML:

```yaml
profile: dry_run
policy: my_policy
task: cleanup_table_d2
name: my_policy_debug
eval_name: smoke
policy_kwargs:
  ckpt_path: /path/to/checkpoint.pt
```

Built-ins:

- `seeker`: current Seeker/RVT2 policy wrapper.
- `cache_replay`: replays cached absolute actions through the normal executor.

Use `cache_replay` for actuation-path smoke tests without a learned model:

```yaml
profile: dry_run
policy: cache_replay
replay_cache: data/cleanup_table_d2/rgb_lmdb
replay_episode: 0
replay_start: 0
```
