# src/utils/lowdim_utils.py
from __future__ import annotations

from dataclasses import dataclass
from scipy.spatial.transform import Rotation as R
from typing import Any, Dict, List, Tuple, Optional, Callable
import cv2
import torch
import numpy as np

# --------------------------- pose6 <-> SE(3) ---------------------------

def rot6d_to_R(R6: np.ndarray) -> np.ndarray:
    """
    R6: [N,6] continuous rotation representation
    Returns: [N,3,3] rotation matrices
    """
    a1 = R6[:, 0:3]
    a2 = R6[:, 3:6]

    # normalize first vector
    b1 = a1 / np.linalg.norm(a1, axis=1, keepdims=True)

    # make second vector orthogonal to first
    dot = np.sum(b1 * a2, axis=1, keepdims=True)
    b2 = a2 - dot * b1
    b2 = b2 / np.linalg.norm(b2, axis=1, keepdims=True)

    # third vector by right-hand rule
    b3 = np.cross(b1, b2, axis=1)

    Rm = np.stack([b1, b2, b3], axis=-1)  # [N,3,3]
    return Rm

def R_to_rpy_xyz(Rm: np.ndarray) -> np.ndarray:
    """
    Rm: [N,3,3]
    Returns: [N,3] roll, pitch, yaw (xyz intrinsic, radians)
    """
    return R.from_matrix(Rm).as_euler("xyz", degrees=False)


def rpy_to_R_xyz(rpy: np.ndarray) -> np.ndarray:
    # rpy = [roll, pitch, yaw] in rad, xyz convention
    return R.from_euler("xyz", rpy, degrees=False).as_matrix()

def pose6_to_xyz6(pose6: np.ndarray) -> np.ndarray:
    """
    pose6: [T,6] = [x(mm),y(mm),z(mm), roll, pitch, yaw]
    return: [T,12] = [x,y,z, R_flat(9)]
    """
    T = pose6.shape[0]
    xyz = pose6[:, :3]
    Rm = np.stack([rpy_to_R_xyz(pose6[i, 3:6]) for i in range(T)], axis=0)  # [T,3,3]
    R6 = Rm.reshape(T, 9)[:, :6]
    return xyz, R6

def xyz6_to_pose6(xyz6: np.ndarray) -> np.ndarray:
    """
    xyz6: [N,9] = [x,y,z, rot6d]
    Returns: [N,6] = [x,y,z, roll, pitch, yaw]
    """
    xyz = xyz6[:, :3]
    R6 = xyz6[:, 3:9]

    Rm = rot6d_to_R(R6)
    rpy = R_to_rpy_xyz(Rm)
    return np.concatenate([xyz, rpy], axis=-1)


def action_abs_to_xyz6g(tgt_pose6: np.ndarray, tgt_grip: np.ndarray) -> np.ndarray:
    """
    tgt_pose6: [T,6], tgt_grip: [T]
    return: [T,13] = [x,y,z, R_flat(9), grip]
    """
    xyz, R6 = pose6_to_xyz6(tgt_pose6)  # [T,12]
    xyz6 = np.concatenate([xyz, R6], axis=1)
    g = tgt_grip.reshape(-1, 1)
    return np.concatenate([xyz6, g], axis=1)

def xyz6g_to_action_abs(xyz6g: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    xyz6, g = xyz6g[:, :-1], xyz6g[:, -1:]
    pose6 = xyz6_to_pose6(xyz6)
    return pose6, g


# ------------------------ image preprocessing ------------------------

def center_square_crop(img_bgr: np.ndarray) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    s = min(h, w)
    y0 = (h - s) // 2
    x0 = (w - s) // 2
    return img_bgr[y0:y0 + s, x0:x0 + s]

# ------------------------ cuda ------------------------
def dict_apply(
        x: Dict[str, torch.Tensor], 
        func: Callable[[torch.Tensor], torch.Tensor]
        ) -> Dict[str, torch.Tensor]:
    result = dict()
    for key, value in x.items():
        if isinstance(value, dict):
            result[key] = dict_apply(value, func)
        else:
            result[key] = func(value)
    return result