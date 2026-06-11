from __future__ import annotations

import time
from typing import Optional, Tuple

import numpy as np
import rospy

from src.utils.conversion_utils import (
    pose6_to_xyz6,
    rot6d_to_R,
    rotation_angle_between,
    se3_interp_xyz_rot6,
    xyz6g_to_action_abs,
)


class PolicyActionExecutor:
    """
    Executes absolute policy action chunks on the robot.

    This is a behavior-preserving extraction from EvalRunner. Keep existing
    action slicing, interpolation, safety checks, gripper handling, and robot
    service calls unchanged.
    """

    def __init__(self, cfg, robot):
        self.cfg = cfg
        self.robot = robot
        self.state = None
        self.stop_requested = lambda: False
        self.last_exec_action10: Optional[np.ndarray] = None
        self._last_grip_bin: Optional[int] = None

    def bind_runtime_state(self, state, stop_requested) -> None:
        self.state = state
        self.stop_requested = stop_requested

    def reset(self) -> None:
        self._last_grip_bin = None
        self.last_exec_action10 = None

    # HARDCODE
    def compute_exec_slice(self) -> Tuple[int, int]:
        Ta = int(self.cfg.pred_horizon)
        Te = int(self.cfg.exec_horizon)

        start = 1
        end = min(start + Te, Ta)
        return start, end

    def _ros_now(self) -> float:
        return rospy.Time.now().to_sec()

    def _state_should_stop(self) -> bool:
        if self.state is None:
            return False
        return bool(self.state.quit_requested)

    def _state_blocks_execution(self) -> bool:
        if self.state is None:
            return False
        return bool(self.state.reset_requested) or (not bool(self.state.running))

    def _in_workspace(self, xyz: np.ndarray) -> bool:
        xyz = np.asarray(xyz, dtype=np.float32).reshape(3,)
        mn = np.asarray(self.cfg.workspace_min_xyz, dtype=np.float32).reshape(3,)
        mx = np.asarray(self.cfg.workspace_max_xyz, dtype=np.float32).reshape(3,)
        return bool(np.all(xyz >= mn) and np.all(xyz <= mx))

    def _step_ok(self, a_prev: np.ndarray, a_next: np.ndarray) -> bool:
        """
        a_prev, a_next: (10,) actions [xyz,rot6,grip]
        """
        a_prev = np.asarray(a_prev, dtype=np.float32).reshape(-1)
        a_next = np.asarray(a_next, dtype=np.float32).reshape(-1)

        # workspace check for absolute action
        if not self._in_workspace(a_next[0:3]):
            rospy.logwarn_throttle(1.0, f"[EvalRunner] action out of workspace xyz={a_next[0:3]}")
            return False

        # consecutive delta check
        dp = float(np.linalg.norm(a_next[0:3] - a_prev[0:3], ord=2))
        Rprev = rot6d_to_R(a_prev[3:9]).reshape(3, 3)
        Rnext = rot6d_to_R(a_next[3:9]).reshape(3, 3)
        dr = rotation_angle_between(Rprev, Rnext)

        if dp > float(self.cfg.max_step_trans):
            rospy.logwarn_throttle(1.0, f"[EvalRunner] step too large dp={dp:.4f} > {self.cfg.max_step_trans}")
            return False
        if dr > float(self.cfg.max_step_rot_rad):
            rospy.logwarn_throttle(1.0, f"[EvalRunner] rot step too large dr={dr:.4f} > {self.cfg.max_step_rot_rad}")
            return False

        return True

    def _pose_error(self, cur6: np.ndarray, tgt6: np.ndarray) -> Tuple[float, float]:
        dp = float(np.linalg.norm(cur6[:3] - tgt6[:3], ord=2))
        dr = float(np.linalg.norm(cur6[3:6] - tgt6[3:6], ord=2))
        return dp, dr

    def _is_step_done(self, tgt6: np.ndarray) -> bool:
        st = self.robot.get_state()
        if st is None or getattr(st, "ee_pose", None) is None:
            return False
        cur = np.asarray(st.ee_pose[:6], dtype=np.float32)
        dp, dr = self._pose_error(cur, tgt6)
        return (dp <= float(self.cfg.pos_tol_mm)) and (dr <= float(self.cfg.rot_tol_rad))

    def convert_action_to_robot(self, act: np.ndarray) -> Tuple[np.ndarray, Optional[float]]:
        """
        act: [x,y,z, rot6d(6), grip?]
        xyz unit per cfg.xyz_unit.
        returns pose6_mm_rpy + optional gripper scalar
        """
        act = np.asarray(act, dtype=np.float32).reshape(1, -1)
        if act.shape[-1] < 9:
            raise ValueError(f"Expected >=9 dims (xyz+rot6d), got {act.shape[0]}")

        pose6, grip = xyz6g_to_action_abs(act)

        pose6 = np.asarray(pose6, dtype=np.float32).flatten()
        grip = float(grip.flatten()[0]) if grip is not None else None
        return pose6.astype(np.float32), grip

    def _current_action10(self, grip: float) -> Optional[np.ndarray]:
        st = self.robot.get_state()
        ee = getattr(st, "ee_pose", None)
        if ee is None or len(ee) < 6:
            rospy.logwarn("[EvalRunner] cannot safety-check first action: missing current ee_pose")
            return None
        cur6 = np.asarray(ee[:6], dtype=np.float32).reshape(1, 6)
        xyz, rot6 = pose6_to_xyz6(cur6)
        return np.concatenate(
            [xyz[0], rot6[0], np.asarray([float(grip)], dtype=np.float32)],
            axis=0,
        ).astype(np.float32)

    def _send_step_command(self, pose6: np.ndarray, grip: Optional[np.ndarray]):
        self.robot.move_servo_cart(
            pose6_mm_rpy=pose6.tolist(),
            tool_coord=bool(self.cfg.servo_tool_coord),
        )

        if grip is None:
            return

        if self.cfg.gripper_binary:
            gb = 1 if grip > float(self.cfg.gripper_deadband) else 0
            if self._last_grip_bin is None or gb != self._last_grip_bin:
                if gb == 1:
                    self.robot.move_gripper(float(self.cfg.gripper_open_pulse))
                else:
                    self.robot.move_gripper(float(self.cfg.gripper_close_pulse))
                self._last_grip_bin = gb
        else:
            self.robot.move_gripper(grip)

    def _execute_step_until_done(self, pose6: np.ndarray, grip: Optional[float]) -> bool:
        """
        Stream servo command until reached or timeout.
        """
        t0 = self._ros_now()
        dt = 1.0 / float(self.cfg.control_hz)

        while (self._ros_now() - t0) < float(self.cfg.step_timeout_s):
            if rospy.is_shutdown() or self.stop_requested() or self._state_should_stop():
                return False
            if self._state_blocks_execution():
                return False

            self._send_step_command(pose6, grip)

            if self._is_step_done(pose6):
                return True

            time.sleep(dt)

        return True

    def execute_chunk(self, chunk_actions: np.ndarray) -> bool:
        """
        chunk_actions: (Te, 10) in [xyz, rot6d, grip] predicted by policy (absolute)
        """
        dt = 1.0 / float(self.cfg.control_hz)
        N_interp = self.cfg.interp_steps
        grip_mode = self.cfg.gripper_interp_mode

        acts = np.asarray(chunk_actions, dtype=np.float32)
        if acts.ndim != 2 or acts.shape[1] < 10:
            rospy.logwarn(f"[EvalRunner] bad chunk_actions shape={acts.shape}")
            return False
        if acts.shape[0] == 0:
            return True

        # --- Build smoothed action sequence ---
        exec_actions = []

        first = acts[0, :10].copy()

        # bridge from last executed action to first action of this chunk
        if self.last_exec_action10 is not None:
            bridge = se3_interp_xyz_rot6(self.last_exec_action10, first, n=N_interp, grip_mode=grip_mode)
            exec_actions.append(bridge)

        # interpolate within chunk
        for i in range(acts.shape[0] - 1):
            a0 = acts[i, :10]
            a1 = acts[i + 1, :10]
            seg = se3_interp_xyz_rot6(a0, a1, n=N_interp, grip_mode=grip_mode)
            exec_actions.append(seg)

        # include final point (one step)
        exec_actions.append(acts[-1:, :10])
        exec_actions = np.concatenate(exec_actions, axis=0).astype(np.float32)  # (N,10)

        # --- Sanity checks ---
        for j in range(exec_actions.shape[0]):
            if not self._in_workspace(exec_actions[j, 0:3]):
                rospy.logwarn(f"[EvalRunner] abort: action[{j}] xyz out of workspace: {exec_actions[j,0:3]}")
                return False

        prev = self.last_exec_action10
        if prev is None:
            prev = self._current_action10(float(exec_actions[0, 9]))
            if prev is None:
                return False
        for j in range(exec_actions.shape[0]):
            cur = exec_actions[j]
            if not self._step_ok(prev, cur):
                rospy.logwarn(f"[EvalRunner] abort: step check failed at j={j}")
                return False
            prev = cur

        # --- Stream execution ---
        for j in range(exec_actions.shape[0]):
            if rospy.is_shutdown() or self.stop_requested() or self._state_should_stop():
                return False
            if self._state_blocks_execution():
                return False

            act10 = exec_actions[j]
            pose6, grip = self.convert_action_to_robot(act10)
            # # HARDCODE: avoid frequent gripper command, fix this with better handling
            if grip is not None and grip <= 100:
                grip = None

            self._send_step_command(pose6, grip)

            self.last_exec_action10 = act10.copy()
            time.sleep(dt)

        return True
