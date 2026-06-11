from __future__ import annotations

import sys
import types
from dataclasses import dataclass

import numpy as np


class _FakeTime:
    @staticmethod
    def now():
        return types.SimpleNamespace(to_sec=lambda: 0.0)


sys.modules.setdefault(
    "rospy",
    types.SimpleNamespace(
        Time=_FakeTime,
        is_shutdown=lambda: False,
        logwarn=lambda *args, **kwargs: None,
        logwarn_throttle=lambda *args, **kwargs: None,
    ),
)

from src.eval import action_executor as action_executor_module
from src.eval.action_executor import PolicyActionExecutor


def _force_ros_running(monkeypatch):
    monkeypatch.setattr(action_executor_module.rospy, "is_shutdown", lambda: False)


@dataclass
class _Cfg:
    pred_horizon: int = 16
    exec_horizon: int = 8
    workspace_min_xyz: tuple = (0.0, -200.0, 0.0)
    workspace_max_xyz: tuple = (500.0, 200.0, 400.0)
    max_step_trans: float = 1000.0
    max_step_rot_rad: float = 10.0
    pos_tol_mm: float = 2.0
    rot_tol_rad: float = 0.05
    control_hz: float = 10.0
    step_timeout_s: float = 0.35
    interp_steps: int = 1
    gripper_interp_mode: str = "hold"
    servo_tool_coord: bool = False
    gripper_binary: bool = False
    gripper_open_pulse: float = 850.0
    gripper_close_pulse: float = 100.0
    gripper_deadband: float = 0.5


class _Robot:
    def __init__(self, ee_pose=None):
        self.ee_pose = [100.0, 0.0, 100.0, 0.0, 0.0, 0.0] if ee_pose is None else list(ee_pose)
        self.servo_calls = []
        self.gripper_calls = []

    def get_state(self):
        return types.SimpleNamespace(ee_pose=self.ee_pose)

    def move_servo_cart(self, pose6_mm_rpy, tool_coord):
        self.servo_calls.append((list(pose6_mm_rpy), bool(tool_coord)))

    def move_gripper(self, value):
        self.gripper_calls.append(float(value))


def _action(xyz=(100.0, 0.0, 100.0), grip=100.0):
    return np.asarray([*xyz, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, grip], dtype=np.float32)


def test_compute_exec_slice_keeps_existing_start_offset():
    executor = PolicyActionExecutor(_Cfg(), _Robot())

    assert executor.compute_exec_slice() == (1, 9)


def test_reset_clears_executor_state():
    executor = PolicyActionExecutor(_Cfg(), _Robot())
    executor.last_exec_action10 = _action()
    executor._last_grip_bin = 1

    executor.reset()

    assert executor.last_exec_action10 is None
    assert executor._last_grip_bin is None


def test_execute_chunk_sends_servo_and_suppresses_small_gripper(monkeypatch):
    _force_ros_running(monkeypatch)
    monkeypatch.setattr(action_executor_module.time, "sleep", lambda _: None)
    robot = _Robot()
    executor = PolicyActionExecutor(_Cfg(), robot)
    chunk = np.stack(
        [
            _action((100.0, 0.0, 100.0), 100.0),
            _action((110.0, 0.0, 100.0), 100.0),
        ],
        axis=0,
    )

    ok = executor.execute_chunk(chunk)

    assert ok is True
    assert len(robot.servo_calls) == 2
    assert robot.gripper_calls == []
    assert np.allclose(executor.last_exec_action10, chunk[-1])


def test_workspace_violation_aborts_before_robot_command(monkeypatch):
    _force_ros_running(monkeypatch)
    monkeypatch.setattr(action_executor_module.time, "sleep", lambda _: None)
    robot = _Robot()
    cfg = _Cfg(workspace_max_xyz=(120.0, 200.0, 400.0))
    executor = PolicyActionExecutor(cfg, robot)
    chunk = np.stack(
        [
            _action((100.0, 0.0, 100.0), 100.0),
            _action((200.0, 0.0, 100.0), 100.0),
        ],
        axis=0,
    )

    ok = executor.execute_chunk(chunk)

    assert ok is False
    assert robot.servo_calls == []
    assert robot.gripper_calls == []


def test_step_violation_aborts_before_robot_command(monkeypatch):
    _force_ros_running(monkeypatch)
    monkeypatch.setattr(action_executor_module.time, "sleep", lambda _: None)
    robot = _Robot()
    cfg = _Cfg(max_step_trans=5.0)
    executor = PolicyActionExecutor(cfg, robot)
    chunk = np.stack(
        [
            _action((100.0, 0.0, 100.0), 100.0),
            _action((110.0, 0.0, 100.0), 100.0),
        ],
        axis=0,
    )

    ok = executor.execute_chunk(chunk)

    assert ok is False
    assert robot.servo_calls == []
    assert robot.gripper_calls == []
