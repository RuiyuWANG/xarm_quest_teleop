from dataclasses import dataclass, field
from typing import List, Literal, Optional
import numpy as np

Hand = Literal["left", "right"]


@dataclass
class TeleopConfig:
    # ==============================
    # Autolaunch (subprocess roslaunch)
    # ==============================
    auto_launch_quest: bool = True
    auto_launch_robot: bool = True

    QUEST_LAUNCH_CMD: List[str] = field(default_factory=lambda: [
        "roslaunch",
        "ros_tcp_endpoint",
        "endpoint.launch",
        "tcp_ip:=192.168.0.182",
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

    # Package that contains your executable scripts (rosrun uses this)
    scripts_package: str = "cloudgripper_teleop"

    launch_workdir: Optional[str] = None
    pipe_launch_output: bool = True
    startup_timeout_s: float = 45.0

    # ==============================
    # Topics for stamped sync
    # ==============================
    active_hand: Hand = "right"

    # Raw (unstamped) pose topics from q2r
    right_pose_topic: str = "/q2r_right_hand_pose"
    left_pose_topic: str = "/q2r_left_hand_pose"

    # Stamped pose topics (from your quest_stamp_node wrapper)
    right_pose_stamped_topic: str = "/q2r_right_hand_pose_stamped"
    left_pose_stamped_topic: str = "/q2r_left_hand_pose_stamped"

    # TwistStamped topics (from wrapper; may still be useful for logging)
    right_twist_stamped_topic: str = "/q2r_right_hand_twist_stamped"
    left_twist_stamped_topic: str = "/q2r_left_hand_twist_stamped"

    # InputsStamped topics (from wrapper)
    right_inputs_stamped_topic: str = "/q2r_right_hand_inputs_stamped"
    left_inputs_stamped_topic: str = "/q2r_left_hand_inputs_stamped"

    # Robot state topic (RobotMsg)
    robot_state: str = "/xarm/xarm_states"

    # ==============================
    # message_filters sync params
    # ==============================
    sync_queue_size: int = 30
    sync_slop_s: float = 0.03
    sync_allow_headerless: bool = False  # we stamp everything

    # ==============================
    # Control gating / buttons
    # ==============================
    require_deadman: bool = True  # uses inputs.inputs.<deadman_field>
    deadman_field: str = "button_lower"

    # Recenter / relatch reference (pose-delta teleop)
    enable_reset: bool = True
    reset_field: str = "button_upper"

    # If True: releasing deadman clears reference; next press recenters.
    # If False: releasing deadman pauses motion but keeps reference.
    clear_reference_on_deadman_release: bool = True

    # ==============================
    # Pose-delta teleop mapping (Quest pose -> robot target pose)
    # ==============================
    # Quest delta position is in meters; robot TCP position is in millimeters.
    pos_scale: float = 0.35          # dimensionless multiplier on quest delta position
    rot_scale: float = 0.25         # dimensionless multiplier on quest delta rotation (axis-angle)

    # Map Quest delta axes into robot base/tool axes.
    # dp_robot = R_pos_map @ dp_quest
    # daa_robot = R_rot_map @ daa_quest
    R_pos_map: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float32))
    R_rot_map: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float32))

    # Apply orientation deltas (recommended True; set False for position-only demo)
    enable_orientation: bool = True

    # ==============================
    # Filtering / robustness
    # ==============================
    # EMA on deltas: 0..1 (higher = snappier, lower = smoother)
    delta_filter_alpha: float = 0.25

    # Safety clamp on how far target can move from reference (in Quest-delta space)
    max_delta_pos_m: float = 0.50
    max_delta_rot_rad: float = 1.20

    # ==============================
    # Gripper mapping
    # ==============================
    enable_gripper: bool = True
    grip_rate_limit_s: float = 0.12
    grip_change_eps: float = 5.0
    grip_continuous: bool = True
    grip_speed: float = 0.2

    # close amount = press_index - press_middle (clamped 0..1)
    grip_close_from_index: bool = True
    grip_open_from_middle: bool = True

    # ==============================
    # Haptics
    # ==============================
    enable_haptics: bool = True
    deadman_haptic_freq: float = 80.0
    deadman_haptic_amp: float = 0.08
    err_haptic_freq: float = 140.0
    err_haptic_amp: float = 0.35

    # ==============================
    # Camera handling for data collection
    # ==============================
    camera_image_topics: List[str] = field(default_factory=lambda: [])
    camera_buffer_seconds: float = 2.0
    camera_match_window_s: float = 0.05

    # Servo_Cartesian
    servo_auto_configure: bool = True   # run mode/state setup at startup
    servo_tool_coord: bool = False      # base coord recommended

    # MUST be < 10mm per update
    servo_max_step_mm: float = 8.0      # keep below 10mm
    servo_max_step_rot_rad: float = 0.05  # ~3 degrees
    
    # Servo loop rate (fixed-rate streaming)
    servo_rate_hz: float = 100.0   # 50~100 recommended

    # Step limits (xArm requirement: <10mm)
    servo_max_step_mm: float = 6.0
    servo_max_step_rot_rad: float = 0.05

    # Adaptive filter (Vive-style)
    pose_ema_alpha_slow: float = 0.35
    pose_ema_bypass_mm: float = 15.0
    pose_ema_bypass_rot_rad: float = 0.14   # ~8 deg
    pose_clamp_mm: float = 30.0
    pose_clamp_rot_rad: float = 0.21        # ~12 deg

    # Smooth re-engagement
    reengage_steps: int = 30

