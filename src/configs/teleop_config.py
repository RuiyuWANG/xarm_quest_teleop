from dataclasses import dataclass, field
from typing import List, Literal, Optional
import numpy as np

Hand = Literal["left", "right"]

@dataclass
class TeleopConfig:
    # Autolaunch (subprocess roslaunch)
    auto_launch_quest: bool = True
    auto_launch_robot: bool = True

    QUEST_LAUNCH_CMD: List[str] = field(default_factory=lambda: [
        "roslaunch",
        "ros_tcp_endpoint",
        "endpoint.launch",
        "tcp_ip:=192.168.0.181",
        "tcp_port:=10000",
    ])

    ROBOT_LAUNCH_CMD: List[str] = field(default_factory=lambda: [
        "roslaunch",
        "xarm_bringup",
        "xarm7_server.launch",
        "robot_ip:=192.168.1.241",
        "report_type:=dev",
        "add_gripper:=true",
    ])

    # rosrun
    scripts_package: str = "cloudgripper_teleop"

    launch_workdir: Optional[str] = None
    pipe_launch_output: bool = True
    startup_timeout_s: float = 20.0

    # Active hand
    active_hand: Hand = "right"

    # Topics for stamped sync
    right_pose_stamped_topic: str = "/q2r_right_hand_pose_stamped"
    left_pose_stamped_topic: str = "/q2r_left_hand_pose_stamped"
    right_twist_stamped_topic: str = "/q2r_right_hand_twist_stamped"
    left_twist_stamped_topic: str = "/q2r_left_hand_twist_stamped"
    right_inputs_stamped_topic: str = "/q2r_right_hand_inputs_stamped"
    left_inputs_stamped_topic: str = "/q2r_left_hand_inputs_stamped"

    # Robot state topic
    robot_state: str = "/xarm/xarm_states"

    # message_filters sync params
    sync_queue_size: int = 30
    sync_slop_s: float = 0.03
    sync_allow_headerless: bool = False  # strictly require header

    # Control gating / buttons
    require_deadman: bool = True
    deadman_field: str = "button_lower"
    enable_reset: bool = False    # Recenter reference
    reset_field: str = "button_upper"
    enable_subtask: bool = True  # Subtask control
    subtask_field: str = "button_lower"  # The other hand

    # If True: releasing deadman clears reference; next press recenters; If False: releasing deadman pauses motion but keeps reference.
    clear_reference_on_deadman_release: bool = True

    # Pose-delta teleop mapping
    pos_scale: float = 0.40         # multiplier on quest delta position
    rot_scale: float = 0.30         # multiplier on quest delta rotation (axis-angle)
    
    # Map Quest delta axes into robot base/tool axes.
    R_pos_map: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float32))   # dp_robot = R_pos_map @ dp_quest
    R_rot_map: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float32))   # daa_robot = R_rot_map @ daa_quest

    # Apply orientation deltas
    enable_orientation: bool = True

    # Filtering
    delta_filter_alpha: float = 0.25  # (higher = snappier, lower = smoother)
    max_delta_pos_m: float = 0.50     # Safety clamp on how far target can move from reference (in Quest-delta space)
    max_delta_rot_rad: float = 1.20

    # Gripper mapping
    enable_gripper: bool = True
    grip_rate_limit_s: float = 0.12
    grip_change_eps: float = 5.0
    grip_continuous: bool = True
    grip_speed: float = 0.2
    grip_close_from_index: bool = True
    grip_open_from_middle: bool = True

    # Haptics
    enable_haptics: bool = True
    deadman_haptic_freq: float = 80.0
    deadman_haptic_amp: float = 0.08
    err_haptic_freq: float = 140.0 
    err_haptic_amp: float = 0.35   # harder vibration for error

    # Servo_Cartesian
    servo_auto_configure: bool = True     # run mode/state setup at startup
    servo_tool_coord: bool = False        # base coord recommended
    servo_max_step_mm: float = 8.0        # keep below 10mm
    servo_max_step_rot_rad: float = 0.05  # ~3 degrees
    servo_rate_hz: float = 100.0          # Servo loop rate (fixed-rate streaming)

    # Adaptive filter (Vive-style)
    pose_ema_alpha_slow: float = 0.35
    pose_ema_bypass_mm: float = 15.0
    pose_ema_bypass_rot_rad: float = 0.14   # ~8 deg
    pose_clamp_mm: float = 30.0
    pose_clamp_rot_rad: float = 0.21        # ~12 deg

    # Smooth re-engagement
    reengage_steps: int = 30