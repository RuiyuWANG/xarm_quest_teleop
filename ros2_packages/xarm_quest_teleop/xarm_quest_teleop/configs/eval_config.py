# xarm_quest_teleop/configs/eval_config.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, List, Any

from xarm_quest_teleop.configs.collector_config import CameraSyncConfig, RobotSyncConfig, AutoLaunchConfig


@dataclass
class EvalConfig:
    # ---------------- model ----------------
    model_ckpt_path: str = "~/ros2_ws/src/seeker-dev/experiments/cleanup_table_d2/real_policy/real_rvt2_cleanup_table_d2_seed_0/checkpoints/latest.ckpt"
    result_log_dir: str = "evaluation/"
    device: str = "cuda"  # "cuda" | "cpu"
    seed: int = 0
    n_rollouts: int = 20
    horizon: int = 1000
    task_name: str = "cleanup_table_d2"
    model_name: str = "real_rvt2_policy"
    policy_name: str = "seeker"
    policy_kwargs: Dict[str, Any] = field(default_factory=dict)
    replay_cache_dir: str = ""
    replay_episode: int = 0
    replay_start: int = 0
    eval_name: str = "default"
    eval_config_path: Optional[str] = "cleanup_table_d2_dry_run.yaml"
    eval_profile: str = "dry_run"  # "rollout" | "dry_run" | "manual"
    record: bool = True
    video_fps: int = 5  
    record_cam: str = "d435i_front"
    debug_no_actuate: bool = False
    live_viz: bool = False
    calibration_path: str = "all_cams_calib.json"

    # ---------------- observation keys ----------------
    rgb_cams: List[str] = field(default_factory=lambda: ["d405", "d435i_front"])
    lowdim_keys: List[str] = field(default_factory=lambda: ["ee_pose6", "gripper_state"])

    # ---------------- temporal policy spec ----------------
    obs_horizon: int = 2     # To
    pred_horizon: int = 16    # Ta
    exec_horizon: int = 8    # Te (1..Ta)
    exec_start_offset: int = 1
    dt_ctrl: float = 1.0 / 30.0
    use_delay_comp: bool = True
    
    # ---------------- smooth interpolation ----------------
    interp_steps: int = 3
    gripper_interp_mode: str = "hold" # "hold" | "linear"
    max_step_trans: float = 20.0  # mm
    max_step_rot_rad: float = 0.15  # rad
    workspace_min_xyz: List[float] = (50.0, -200.0, 0.0)  # mm
    workspace_max_xyz: List[float] = (500.0, 200.0, 400.0)  # mm

    # ---------------- execution spec ----------------
    control_hz: float = 30.0
    obs_stale_s: float = 0.20

    pos_tol_mm: float = 2.0
    rot_tol_rad: float = 0.05
    step_timeout_s: float = 0.35
    chunk_timeout_s: float = 6.0

    servo_tool_coord: bool = False

    xyz_unit: str = "mm"   # model output units (xyz) | "mm" | "m"
    
    gripper_binary: bool = False    # True if gripper prediction is binary
    gripper_open_pulse: float = 850.0
    gripper_close_pulse: float = 100.0
    gripper_deadband: float = 0.5
    gripper_command_eps: float = 5.0

    # ---------------- sync ----------------
    enable_rgb_sync: bool = True
    enable_full_sync: bool = False  # eval is RGB-only; depth/full-sync is intentionally disabled

    # ---------------- launch ----------------
    cam_sync: CameraSyncConfig = field(default_factory=CameraSyncConfig)
    robot_sync: RobotSyncConfig = field(default_factory=RobotSyncConfig)
    launch: AutoLaunchConfig = field(default_factory=AutoLaunchConfig)
