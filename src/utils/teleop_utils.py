# src/utils/teleop_utils.py
from __future__ import annotations

import math
from dataclasses import dataclass
import numpy as np
from typing import Optional, Any

def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def wrap_to_pi(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2 * np.pi) - np.pi


def vec_clamp_norm(v: np.ndarray, max_norm: float) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= 1e-12 or n <= max_norm:
        return v
    return v * (max_norm / n)


def ema_vec(prev: np.ndarray, x: np.ndarray, alpha: float) -> np.ndarray:
    a = float(alpha)
    return (1.0 - a) * prev + a * x


def quat_xyzw_to_rot(q_xyzw: np.ndarray) -> np.ndarray:
    """Quaternion [x,y,z,w] -> 3x3 rotation matrix."""
    x, y, z, w = [float(q_xyzw[i]) for i in range(4)]
    n = math.sqrt(x*x + y*y + z*z + w*w)
    if n < 1e-12:
        return np.eye(3, dtype=np.float32)
    x, y, z, w = x/n, y/n, z/n, w/n

    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z

    R = np.array([
        [1 - 2*(yy+zz),     2*(xy - wz),     2*(xz + wy)],
        [    2*(xy + wz), 1 - 2*(xx+zz),     2*(yz - wx)],
        [    2*(xz - wy),     2*(yz + wx), 1 - 2*(xx+yy)],
    ], dtype=np.float32)
    return R


def rot_to_axis_angle(R: np.ndarray) -> np.ndarray:
    """
    Rotation matrix -> axis-angle vector (axis * angle), 3-vector in rad.
    Stable for small angles.
    """
    tr = float(np.trace(R))
    c = max(-1.0, min(1.0, (tr - 1.0) * 0.5))
    angle = math.acos(c)
    if angle < 1e-6:
        return np.zeros(3, dtype=np.float32)

    axis = np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1],
    ], dtype=np.float32)
    n = float(np.linalg.norm(axis))
    if n < 1e-12:
        return np.zeros(3, dtype=np.float32)
    axis = axis / n
    return axis * float(angle)

def _msg_stamp_sec(msg: Any) -> Optional[float]:
    try:
        return float(msg.header.stamp.to_sec())
    except Exception:
        return None
    
def _stamp_sec(msg: Any) -> Optional[float]:
    try:
        return float(msg.header.stamp.to_sec())
    except Exception:
        return None

@dataclass
class LatchedReference:
    quest_pos_m: np.ndarray          # (3,)
    quest_rot: np.ndarray            # (3,3)
    robot_pose6_mm_rpy: np.ndarray   # (6,) [mm,mm,mm,rad,rad,rad]


def _smoothstep01(t: float) -> float:
    t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else t
    return t * t * (3.0 - 2.0 * t)

class AdaptivePoseFilter:
    """
    Vive-style filter:
      - clamp per-frame spikes (pos, rot)
      - bypass EMA on big moves (no lag)
      - EMA only for micro-jitter
    Units:
      - position in mm
      - rotation in rad (RPY)
    """
    def __init__(
        self,
        ema_alpha_slow: float,
        ema_bypass_mm: float,
        ema_bypass_rot_rad: float,
        clamp_mm: float,
        clamp_rot_rad: float,
    ):
        self.ema_alpha_slow = float(ema_alpha_slow)
        self.ema_bypass_mm = float(ema_bypass_mm)
        self.ema_bypass_rot_rad = float(ema_bypass_rot_rad)
        self.clamp_mm = float(clamp_mm)
        self.clamp_rot_rad = float(clamp_rot_rad)
        self._last: Optional[np.ndarray] = None  # 6

    def reset(self):
        self._last = None

    def _clamp_step(self, dpos_mm: np.ndarray, drot_rad: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # clamp by norm, separately
        dpos_mm = vec_clamp_norm(dpos_mm.astype(np.float32), self.clamp_mm)
        drot_rad = vec_clamp_norm(drot_rad.astype(np.float32), self.clamp_rot_rad)
        return dpos_mm, drot_rad

    def apply(self, pose6_mm_rpy: np.ndarray, bypass: bool = False) -> np.ndarray:
        if self._last is None:
            self._last = pose6_mm_rpy.copy()
            return pose6_mm_rpy

        d = pose6_mm_rpy - self._last
        d[:3], d[3:6] = self._clamp_step(d[:3], wrap_to_pi(d[3:6]))

        clamped = self._last + d
        clamped[3:6] = wrap_to_pi(clamped[3:6])

        pos_step = float(np.linalg.norm(d[:3]))
        rot_step = float(np.linalg.norm(d[3:6]))

        if bypass or (pos_step > self.ema_bypass_mm) or (rot_step > self.ema_bypass_rot_rad):
            self._last = clamped
            return clamped

        a = max(0.0, min(1.0, self.ema_alpha_slow))
        out = (1.0 - a) * self._last + a * clamped
        out[3:6] = wrap_to_pi(out[3:6])
        self._last = out
        return out