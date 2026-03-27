# src/eval/eval_config.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, List

from src.configs.collector_config import RGBDCameraSpec, RGBCameraSpec, CameraSyncConfig, RobotSyncConfig, AutoLaunchConfig


@dataclass
class EvalConfig:
    # ---------------- model ----------------
    model_ckpt_path: str = "./"
    result_log_dir: str = "../evaluation/"
    device: str = "cuda"  # "cuda" | "cpu"
    seed: int = 0
    n_rollouts: int = 10
    horizon: int = 1000
    task_name: str = "three_piece_toy_d1"
    model_name: str = "baseline_background"
    record: bool = True
    video_fps: int = 5  
    record_cam: str = "d435i_front"

    # ---------------- observation keys ----------------
    rgb_cams_light: List[str] = field(default_factory=lambda: ["d405", "d435i_front"])
    rgb_cams_full: List[str] = field(default_factory=lambda: ["d435i_front", "d435i_shoulder"])
    lowdim_keys: List[str] = field(default_factory=lambda: ["ee_pose6", "gripper_state"])

    # ---------------- temporal policy spec ----------------
    obs_horizon: int = 1     # To
    pred_horizon: int = 16    # Ta
    exec_horizon: int = 8    # Te (1..Ta)
    dt_ctrl: float = 1.0 / 30.0
    use_delay_comp: bool = True
    
    # ---------------- smooth interpolation ----------------
    interp_steps: int = 3
    gripper_interp_mode: str = "hold" # "hold" | "linear"
    max_step_trans: float = 10.0  # mm
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

    # ---------------- sync ----------------
    enable_light_sync: bool = True
    enable_full_sync: bool = False

    # ---------------- launch ----------------
    cam_sync: CameraSyncConfig = field(default_factory=CameraSyncConfig)
    robot_sync: RobotSyncConfig = field(default_factory=RobotSyncConfig)
    launch: AutoLaunchConfig = field(default_factory=AutoLaunchConfig)

    # ---------------- quest ----------------
    launch_quest: bool = False
    quest_cmd: Optional[List[str]] = None
