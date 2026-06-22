from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from xarm_quest_teleop.utils.conversion_utils import (
    REAL_FRONT_CROP_RATIO,
    center_square_crop,
    pose6_to_xyz6,
    rpy_to_R_xyz,
)


REAL_RGB_TO_POLICY_KEY = {
    "d435i_front": "agentview_image",
    "d405": "robot0_eye_in_hand_image",
}
REAL_POLICY_IMAGE_SIZE = 256
REAL_POLICY_ROT_DIM = 9

TASK_DICT = {
    "real_pick_place_veggis_d1": "Pick up he purple toy eggplant and place it in the brown box.",
    "cleanup_table_d2": "Clean up the table.",
    "three_piece_toy_d2": "Insert the three stamps in the holder.",
    "coffee_transport_d1": "Transport the coffee beans from the bowl to the cup.",
    "three_piece_toy_d1": "Insert the three stamps in the holder.",
}


def task_name_to_instruction(task_name: str, default: Optional[str] = None) -> str:
    if task_name in TASK_DICT:
        return TASK_DICT[task_name]
    if default is not None:
        return str(default)
    return str(task_name)


def preprocess_real_rgb_image(
    image_rgb: np.ndarray,
    camera_name: str,
    target_size: int = REAL_POLICY_IMAGE_SIZE,
    front_crop_ratio: float = REAL_FRONT_CROP_RATIO,
) -> np.ndarray:
    image_rgb = np.asarray(image_rgb)
    if image_rgb.ndim != 3 or image_rgb.shape[-1] != 3:
        raise ValueError(f"{camera_name}: expected HWC RGB image, got {image_rgb.shape}")

    image_rgb = center_square_crop(
        image_rgb,
        camera_name=camera_name,
        front_crop_ratio=float(front_crop_ratio),
    )
    if image_rgb.shape[0] != int(target_size) or image_rgb.shape[1] != int(target_size):
        image_rgb = cv2.resize(
            image_rgb,
            (int(target_size), int(target_size)),
            interpolation=cv2.INTER_AREA,
        )
    return image_rgb.astype(np.uint8)


def preprocess_real_rgb_sequence(
    rgb_seq: List[np.ndarray],
    camera_name: str,
    target_size: int = REAL_POLICY_IMAGE_SIZE,
    front_crop_ratio: float = REAL_FRONT_CROP_RATIO,
) -> np.ndarray:
    return np.stack(
        [
            preprocess_real_rgb_image(
                img,
                camera_name,
                target_size,
                front_crop_ratio=front_crop_ratio,
            )
            for img in rgb_seq
        ],
        axis=0,
    ).astype(np.uint8)


def pose6_to_xyz_rot(pose6: np.ndarray, rot_dim: int = REAL_POLICY_ROT_DIM) -> Tuple[np.ndarray, np.ndarray]:
    ee_pose6 = np.asarray(pose6, dtype=np.float32)
    if ee_pose6.ndim == 1:
        ee_pose6 = ee_pose6[None, :]

    if int(rot_dim) == 6:
        return pose6_to_xyz6(ee_pose6)
    if int(rot_dim) != 9:
        raise ValueError(f"rot_dim must be 6 or 9, got {rot_dim}")

    n = ee_pose6.shape[0]
    xyz = ee_pose6[:, :3].astype(np.float32)
    rot9 = np.stack([rpy_to_R_xyz(ee_pose6[i, 3:6]) for i in range(n)], axis=0)
    return xyz, rot9.reshape(n, 9).astype(np.float32)


def convert_real_low_dim(
    low_dim: Dict[str, Any],
    rot_dim: int = REAL_POLICY_ROT_DIM,
) -> Dict[str, np.ndarray]:
    ee_pose6 = np.asarray(low_dim["ee_pose6"], dtype=np.float32)
    if ee_pose6.ndim == 1:
        ee_pose6 = ee_pose6[None, :]

    eef_pos, eef_rot = pose6_to_xyz_rot(ee_pose6, rot_dim=rot_dim)

    gripper = np.asarray(low_dim["gripper_state"], dtype=np.float32).reshape(-1, 1)
    # Match convert_raw_data_to_cache.py and RealWorldDataset: two opposing
    # gripper fingers represented from the measured gripper opening.
    gripper_qpos = np.concatenate([gripper / 2.0, -gripper / 2.0], axis=-1).astype(np.float32)

    return {
        "robot0_eef_pos": eef_pos.astype(np.float32),
        "robot0_eef_rot": eef_rot.astype(np.float32),
        "robot0_gripper_qpos": gripper_qpos,
    }


def preprocess_real_obs_for_policy(
    obs_dict: Dict[str, Dict[str, Any]],
    target_size: int = REAL_POLICY_IMAGE_SIZE,
    rot_dim: int = REAL_POLICY_ROT_DIM,
    camera_map: Optional[Dict[str, str]] = None,
) -> Dict[str, np.ndarray]:
    if "rgb" not in obs_dict or "low_dim" not in obs_dict:
        raise ValueError(f"Unknown obs_dict format. Keys={list(obs_dict.keys())}")

    rgb = obs_dict["rgb"]
    low = obs_dict["low_dim"]
    camera_map = dict(REAL_RGB_TO_POLICY_KEY if camera_map is None else camera_map)

    out: Dict[str, np.ndarray] = {}
    for camera_name, seq in rgb.items():
        if camera_name not in camera_map:
            continue

        imgs_np = preprocess_real_rgb_sequence(
            rgb_seq=list(seq),
            camera_name=camera_name,
            target_size=int(target_size),
        )
        out[camera_map[camera_name]] = np.moveaxis(imgs_np, -1, 1)[None, ...].astype(np.uint8)

    missing = [v for v in camera_map.values() if v not in out]
    if missing:
        raise KeyError(f"Missing required image keys after mapping: {missing}")

    low_proc = convert_real_low_dim(low, rot_dim=rot_dim)
    for k, arr in low_proc.items():
        out[k] = arr[None, ...].astype(np.float32)

    return out
