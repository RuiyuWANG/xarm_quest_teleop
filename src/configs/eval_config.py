# src/eval/eval_config.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, List

from src.configs.collector_config import RGBDCameraSpec, RGBCameraSpec, CameraSyncConfig, RobotSyncConfig, AutoLaunchConfig


@dataclass
class EvalConfig:
    """
    All evaluation knobs in one place.
    """

    # ---------------- model ----------------
    model_ckpt_path: str = "/root/catkin_ws/src/experiment/demos-50_agentview-pass_through_eye_in_hand-pass_through_seed-0/checkpoints/latest.ckpt"
    result_log_path: str = "../evaluation/test_pick_and_place.txt"
    device: str = "cuda"  # "cuda" | "cpu"
    seed: int = 0    

    # ---------------- observation keys ----------------
    rgb_cams_light: List[str] = field(default_factory=lambda: ["d405", "d435i_front"])
    lowdim_keys: List[str] = field(default_factory=lambda: ["ee_pose6", "gripper_state"])

    # ---------------- temporal policy spec ----------------
    # You will later load these from model config; keep them here now.
    obs_horizon: int = 1     # To
    pred_horizon: int = 8    # Ta
    exec_horizon: int = 4    # Te (1..Ta)
    dt_ctrl: float = 1.0 / 30.0
    use_delay_comp: bool = True
    interp_steps: int = 10

    # ---------------- execution spec ----------------
    control_hz: float = 30.0
    obs_stale_s: float = 0.20

    pos_tol_mm: float = 2.0
    rot_tol_rad: float = 0.05
    step_timeout_s: float = 0.35
    chunk_timeout_s: float = 6.0

    servo_tool_coord: bool = False

    # model output units (xyz)
    xyz_unit: str = "mm"
    
    # gripper mapping
    gripper_binary: bool = False
    gripper_open_pulse: float = 850.0
    gripper_close_pulse: float = 100.0
    gripper_deadband: float = 0.5

    # ---------------- sync ----------------
    enable_light_sync: bool = False
    enable_full_sync: bool = True   # optional: if you also want 3 cams + depth later

    # keep same as your collector configs, reuse your existing dataclasses
    cam_sync: CameraSyncConfig = field(default_factory=CameraSyncConfig)
    robot_sync: RobotSyncConfig = field(default_factory=RobotSyncConfig)
    launch: AutoLaunchConfig = field(default_factory=AutoLaunchConfig)

    # ---------------- optional quest ----------------
    launch_quest: bool = False
    quest_cmd: Optional[List[str]] = None
