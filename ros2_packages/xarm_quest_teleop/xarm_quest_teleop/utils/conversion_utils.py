# xarm_quest_teleop/utils/conversion_utils.py
from __future__ import annotations

from dataclasses import dataclass
from scipy.spatial.transform import Rotation as R, Slerp
from typing import Tuple
import cv2
import numpy as np

# --------------------------- pose6 <-> SE(3) ---------------------------
def rot6d_to_R(R6: np.ndarray) -> np.ndarray:
    """
    R6: [N,6] continuous rotation representation
    Returns: [N,3,3] rotation matrices
    """
    no_batch = len(R6.shape) == 1
    if no_batch:
        R6 = R6.reshape(1, -1)
        
    a1 = R6[:, 0:3]
    a2 = R6[:, 3:6]

    # normalize first vector
    b1 = a1 / np.linalg.norm(a1, axis=1, keepdims=True)

    # make second vector orthogonal to first
    dot = np.sum(b1 * a2, axis=1, keepdims=True)
    b2 = a2 - dot * b1
    b2 = b2 / np.linalg.norm(b2, axis=1, keepdims=True)

    b3 = np.cross(b1, b2, axis=1)

    Rm = np.stack([b1, b2, b3], axis=-1)  # [N,3,3]
    return Rm[0] if no_batch else Rm

def R_to_rot6d(Rm: np.ndarray) -> np.ndarray:
    """
    Rm: (...,3,3) -> (...,6) using first two columns
    """
    no_batch = len(Rm.shape) == 2
    if no_batch:
        Rm = Rm.reshape(1, *Rm.shape)
        
    Rm = np.asarray(Rm, dtype=np.float32)
    c1 = Rm[..., :, 0]
    c2 = Rm[..., :, 1]
    rot6 = np.concatenate([c1, c2], axis=-1).astype(np.float32)
    return rot6[0] if no_batch else rot6

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
    return: xyz [T,3], rot6d [T,6] using first two rotation columns
    """
    T = pose6.shape[0]
    xyz = pose6[:, :3]
    Rm = np.stack([rpy_to_R_xyz(pose6[i, 3:6]) for i in range(T)], axis=0)  # [T,3,3]
    R6 = R_to_rot6d(Rm)
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
    return: [T,10] = [x,y,z, rot6d(6), grip]
    """
    xyz, R6 = pose6_to_xyz6(tgt_pose6)
    xyz6 = np.concatenate([xyz, R6], axis=1)
    g = tgt_grip.reshape(-1, 1)
    return np.concatenate([xyz6, g], axis=1)


def xyz6g_to_action_abs(xyz6g: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    xyz6, g = xyz6g[:, :-1], xyz6g[:, -1:]
    pose6 = xyz6_to_pose6(xyz6)
    return pose6, g


# --------------------------- SE(3) interpolation ---------------------------
def se3_interp_xyz_rot6(a0: np.ndarray, a1: np.ndarray, n: int, grip_mode: str = "hold") -> np.ndarray:
    """
    Interpolate between two 10D actions in SE(3):
      action: [x,y,z, rot6d(6), grip]
    Returns: (n, 10) with endpoint excluded (good for streaming)
    grip_mode: "hold" or "linear"
    """
    a0 = np.asarray(a0, dtype=np.float32).reshape(-1)
    a1 = np.asarray(a1, dtype=np.float32).reshape(-1)
    assert a0.shape[0] >= 10 and a1.shape[0] >= 10

    n = int(max(1, n))
    ts = np.linspace(0.0, 1.0, n, endpoint=False, dtype=np.float32)

    # xyz linear
    xyz = (1.0 - ts[:, None]) * a0[0:3][None, :] + ts[:, None] * a1[0:3][None, :]

    # rotation slerp
    R0 = rot6d_to_R(a0[3:9]).reshape(3, 3)
    R1 = rot6d_to_R(a1[3:9]).reshape(3, 3)
    r0 = R.from_matrix(R0)
    r1 = R.from_matrix(R1)

    slerp = Slerp([0.0, 1.0], R.concatenate([r0, r1]))
    Rts = slerp(ts).as_matrix().astype(np.float32)  # (n,3,3)
    rot6 = R_to_rot6d(Rts)                        # (n,6)

    # gripper
    g0 = float(a0[9])
    g1 = float(a1[9])
    if grip_mode == "linear":
        g = (1.0 - ts) * g0 + ts * g1
    else:
        # hold target grip for this segment (less chattering)
        g = np.full((n,), g1, dtype=np.float32)

    out = np.concatenate([xyz.astype(np.float32), rot6.astype(np.float32), g[:, None]], axis=1)
    return out.astype(np.float32)


def rotation_angle_between(Ra: np.ndarray, Rb: np.ndarray) -> float:
    """
    Ra, Rb: (3,3) rotation matrices -> angle in radians
    """
    rrel = R.from_matrix(Ra).inv() * R.from_matrix(Rb)
    return float(rrel.magnitude())

# ------------------------ image preprocessing ------------------------

REAL_FRONT_CROP_RATIO = 0.8


@dataclass(frozen=True)
class CameraImageGeometry:
    crop_x: int
    crop_y: int
    crop_width: int
    crop_height: int
    pad_left: int = 0
    pad_top: int = 0
    pad_right: int = 0
    pad_bottom: int = 0


def camera_image_geometry(
    camera_name: str,
    src_width: int,
    src_height: int,
    *,
    front_crop_ratio: float = REAL_FRONT_CROP_RATIO,
    square_crop: bool = True,
) -> CameraImageGeometry:
    src_width = int(src_width)
    src_height = int(src_height)
    if src_width <= 0 or src_height <= 0:
        raise ValueError(f"invalid image size: width={src_width}, height={src_height}")

    camera_name = str(camera_name)
    if camera_name == "d435i_front":
        crop_size = max(1, int(min(src_width, src_height) * float(front_crop_ratio)))
        x0 = int(round((float(src_width) - float(crop_size)) * 0.5))
        y0 = int(src_height - crop_size)
        return CameraImageGeometry(
            crop_x=max(0, x0),
            crop_y=max(0, y0),
            crop_width=crop_size,
            crop_height=crop_size,
        )

    if camera_name == "d405":
        return CameraImageGeometry(
            crop_x=0,
            crop_y=0,
            crop_width=src_width,
            crop_height=src_height,
            pad_top=max(0, src_width - src_height),
        )

    if square_crop:
        crop_size = min(src_width, src_height)
        return CameraImageGeometry(
            crop_x=max(0, (src_width - crop_size) // 2),
            crop_y=max(0, (src_height - crop_size) // 2),
            crop_width=crop_size,
            crop_height=crop_size,
        )

    return CameraImageGeometry(
        crop_x=0,
        crop_y=0,
        crop_width=src_width,
        crop_height=src_height,
    )


def center_square_crop(
    img_bgr: np.ndarray,
    camera_name: str,
    front_crop_ratio: float = REAL_FRONT_CROP_RATIO,
) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    geom = camera_image_geometry(
        camera_name=camera_name,
        src_width=w,
        src_height=h,
        front_crop_ratio=front_crop_ratio,
        square_crop=True,
    )
    cropped = img_bgr[
        geom.crop_y : geom.crop_y + geom.crop_height,
        geom.crop_x : geom.crop_x + geom.crop_width,
    ]
    if geom.pad_left or geom.pad_top or geom.pad_right or geom.pad_bottom:
        cropped = cv2.copyMakeBorder(
            cropped,
            top=geom.pad_top,
            bottom=geom.pad_bottom,
            left=geom.pad_left,
            right=geom.pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )
    return cropped


def adjust_intrinsics_for_image_geometry(
    K: np.ndarray,
    *,
    camera_name: str,
    src_width: int,
    src_height: int,
    image_size: int,
    square_crop: bool = True,
    front_crop_ratio: float = REAL_FRONT_CROP_RATIO,
) -> np.ndarray:
    geom = camera_image_geometry(
        camera_name=camera_name,
        src_width=int(src_width),
        src_height=int(src_height),
        front_crop_ratio=float(front_crop_ratio),
        square_crop=bool(square_crop),
    )
    K = np.asarray(K, dtype=np.float32).copy()

    K[0, 2] = K[0, 2] - float(geom.crop_x) + float(geom.pad_left)
    K[1, 2] = K[1, 2] - float(geom.crop_y) + float(geom.pad_top)

    padded_width = geom.crop_width + geom.pad_left + geom.pad_right
    padded_height = geom.crop_height + geom.pad_top + geom.pad_bottom
    sx = float(image_size) / float(padded_width)
    sy = float(image_size) / float(padded_height)

    K[0, 0] *= sx
    K[0, 2] *= sx
    K[1, 1] *= sy
    K[1, 2] *= sy
    return K
