from dataclasses import dataclass, field
from typing import Dict, List, Optional

from xarm_quest_teleop.configs.ros2_config import camera_launch_cmds

@dataclass
class RGBDCameraSpec:
    rgb_topic: str
    depth_topic: str
    
@dataclass
class RGBCameraSpec:
    rgb_topic: str
    
@dataclass
class CameraSyncConfig:
    # For full RGBD sync
    pair_slop_s: float = 0.01       # rgb<->depth within a camera
    tri_slop_s: float = 0.02        # cam0<->cam1/cam2 alignment window
    pair_queue: int = 60
    pair_buf_len: int = 120
    keep_s: float = 1.2
    max_wait_s: float = 0.08       # bounded latency: wait up to 50ms
    sub_queue_size: int = 20
    ref_camera: str = "d435i_front"
    
    # For RGB-only sync
    rgb_slop_s: float = 0.02
    rgb_queue_size: int = 60

    cameras_all: Dict[str, RGBDCameraSpec] = field(default_factory=lambda: {
        # "d405": RGBDCameraSpec(
        #     rgb_topic="/d405/color/image_raw",
        #     depth_topic="/d405/aligned_depth_to_color/image_raw",
        # ),
        "d435i_front": RGBDCameraSpec(
            rgb_topic="/d435i_front/color/image_raw",
            # depth_topic="/d435i_front/aligned_depth_to_color/image_raw",
            depth_topic="/d435i_front/depth/image_rect_raw"
        ),
        "d435i_shoulder": RGBDCameraSpec(
            rgb_topic="/d435i_shoulder/color/image_raw",
            # depth_topic="/d435i_shoulder/aligned_depth_to_color/image_raw",
            depth_topic="/d435i_shoulder/depth/image_rect_raw"
        ),
    })
    
    cameras_rgb: Dict[str, RGBCameraSpec] = field(default_factory=lambda: {
        "d405": RGBCameraSpec(
            rgb_topic="/d405/color/image_raw",
        ),
        "d435i_front": RGBCameraSpec(
            rgb_topic="/d435i_front/color/image_raw",
        ),
    })
    
class RobotSyncConfig:
    robot_match_window_s: float = 0.05
    keep_s: float = 1.0
    queue_maxlen: int = 5000

@dataclass
class AutoLaunchConfig:
    enabled: bool = True
    workdir: Optional[str] = None
    pipe_output: bool = True

    auto_launch_quest: bool = True
    auto_launch_robot: bool = True
    auto_launch_realsense: bool = True
    auto_launch_stamp_node: bool = True

    # Commands are lists (subprocess argv), overridden in run script from TeleopConfig
    quest_cmd: List[str] = field(default_factory=list)
    robot_cmd: List[str] = field(default_factory=list)
    stamp_cmd: List[str] = field(default_factory=list)

    # List of launch commands for RealSense ROS2 cameras.
    realsense_all_launch_cmds: List[List[str]] = field(default_factory=lambda: camera_launch_cmds("all"))
    realsense_rgb_launch_cmds: List[List[str]] = field(default_factory=lambda: camera_launch_cmds("rgb"))

# TODO: to support subtask point tracking from quest input
@dataclass
class CollectorConfig:
    # saving
    save_rgb_png: bool = True
    save_depth_npy: bool = False
    enable_full_sync: bool = False
    enable_rgb_sync: bool = True
    max_queue: int = 200
    disable_gripper: bool = False
    fixed_gripper: float = 0.0
    
    # keyboard
    start_key: str = "c"
    delete_key: str = "d"
    quit_key: str = "q"
    save_key: str = "s"

    # autolaunch
    launch: AutoLaunchConfig = field(default_factory=AutoLaunchConfig)
    cam_sync: CameraSyncConfig = field(default_factory=CameraSyncConfig)
    robot_sync: RobotSyncConfig = field(default_factory=RobotSyncConfig)
