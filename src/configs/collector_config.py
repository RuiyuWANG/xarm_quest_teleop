from dataclasses import dataclass, field
from typing import Dict, List, Optional

# TODO: Update this for RGBD topics
@dataclass
class CameraSpec:
    rgb_topic: str
    depth_topic: str



@dataclass
class CameraSyncConfig:
    keep_s: float = 2.0
    queue_size: int = 20
    match_window_s: float = 0.05

    cameras: Dict[str, CameraSpec] = field(default_factory=lambda: {
        "d405": CameraSpec(
            rgb_topic="/d405/color/image_raw",
            depth_topic="/d405/depth/image_rect_raw",
        ),
        "d435i_front": CameraSpec(
            rgb_topic="/d435i_front/color/image_raw",
            depth_topic="/d435i_front/depth/image_rect_raw",
        ),
        "d435i_shoulder": CameraSpec(
            rgb_topic="/d435i_shoulder/color/image_raw",
            depth_topic="/d435i_shoulder/depth/image_rect_raw",
        ),
    })


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

    # List of launch commands for realsense cameras
    realsense_cmds: List[List[str]] = field(default_factory=lambda: [
        ["roslaunch", "realsense2_camera", "rs_camera.launch",
        "serial_no:=230322271104", "camera:=d405",
        "enable_color:=true", "enable_depth:=true",
        "filters:=",
        ],
        ["roslaunch", "realsense2_camera", "rs_camera.launch",
        "serial_no:=335522071488", "camera:=d435i_front",
        "enable_color:=true", "enable_depth:=true",
        "enable_gyro:=false", "enable_accel:=false",
        "filters:=",
        ],
        ["roslaunch", "realsense2_camera", "rs_camera.launch",
        "serial_no:=233522073481", "camera:=d435i_shoulder",
        "enable_color:=true", "enable_depth:=true",
        "enable_gyro:=false", "enable_accel:=false",
        "filters:=",
        ],
    ])


# TODO: figure out the queue size the save rate, check sychronization
@dataclass
class CollectorConfig:
    # saving
    save_rgb_png: bool = True
    save_cloud_pcd: bool = True
    subsample: int = 1
    cloud_point_stride: int = 4
    save_rate_hz: float = 10.0
    max_queue: int = 20
    
    save_depth_png: bool = True
    depth_scale: float = 0.001  # if you want meters later; for PNG we just store raw uint16

    
    # keyboard
    # TODO: add save key and save command for keyboard listener
    start_key: str = "c"
    delete_key: str = "d"
    quit_key: str = "q"
    save_key: str = "s"

    # autolaunch
    launch: AutoLaunchConfig = field(default_factory=AutoLaunchConfig)
    cam_sync: CameraSyncConfig = field(default_factory=CameraSyncConfig)
