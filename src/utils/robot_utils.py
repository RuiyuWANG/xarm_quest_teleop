from typing import List, Tuple
import math
import numpy as np

def _is_finite(x: float) -> bool:
    return not (math.isinf(x) or math.isnan(x))

def _wrap_to_pi(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2 * np.pi) - np.pi

def speeds6_mps_to_xarm_units(
    speeds6_mps_rads: List[float],
    max_lin_m_s: float,
    max_ang_rad_s: float,
    abs_sanity_lin_m_s: float = 2.0,
    abs_sanity_ang_rad_s: float = 10.0,
) -> Tuple[List[float], List[str]]:
    """
    Convert [vx, vy, vz, wx, wy, wz] from m/s + rad/s
    to xArm units: [vx, vy, vz] in mm/s and [wx, wy, wz] in rad/s.

    Returns:
      converted_speeds6
    Raises:
      ValueError on invalid length / non-finite / gross unit mistakes.
    """
    if not isinstance(speeds6_mps_rads, list) or len(speeds6_mps_rads) != 6:
        raise ValueError(f"speeds6 must be list of len 6, got {type(speeds6_mps_rads)} len={len(speeds6_mps_rads) if isinstance(speeds6_mps_rads, list) else 'n/a'}")

    v = [float(x) for x in speeds6_mps_rads]
    if any(not _is_finite(x) for x in v):
        raise ValueError(f"speeds6 contains NaN/Inf: {v}")

    vx, vy, vz, wx, wy, wz = v

    # Absolute sanity: catch obvious unit bugs early (e.g., mm/s passed as m/s)
    if max(abs(vx), abs(vy), abs(vz)) > abs_sanity_lin_m_s:
        raise ValueError(
            f"Linear speed looks too large ({max(abs(vx),abs(vy),abs(vz)):.3f} m/s). "
            f"Did you accidentally pass mm/s as m/s?"
        )
    if max(abs(wx), abs(wy), abs(wz)) > abs_sanity_ang_rad_s:
        raise ValueError(
            f"Angular speed looks too large ({max(abs(wx),abs(wy),abs(wz)):.3f} rad/s). "
            f"Did you accidentally pass deg/s or an unscaled value?"
        )

    # Clamp to configured max
    def clamp(x: float, lo: float, hi: float) -> float:
        return lo if x < lo else hi if x > hi else x

    lin_max = float(max_lin_m_s)
    ang_max = float(max_ang_rad_s)

    v_lin = [vx, vy, vz]
    v_ang = [wx, wy, wz]

    v_lin_clamped = [clamp(x, -lin_max, lin_max) for x in v_lin]
    v_ang_clamped = [clamp(x, -ang_max, ang_max) for x in v_ang]
    
    # Convert m/s -> mm/s for xArm linear speeds
    v_lin_mm_s = [x * 1000.0 for x in v_lin_clamped]

    out = [v_lin_mm_s[0], v_lin_mm_s[1], v_lin_mm_s[2],
           v_ang_clamped[0], v_ang_clamped[1], v_ang_clamped[2]]
    return out
