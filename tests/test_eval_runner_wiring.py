from __future__ import annotations

import types
import sys
from dataclasses import dataclass, field

import numpy as np


from xarm_quest_teleop.ros import compat as fake_rospy

fake_rospy._shutdown = False
fake_rospy.is_shutdown = lambda: bool(fake_rospy._shutdown)
fake_rospy.loginfo = lambda *args, **kwargs: None
fake_rospy.loginfo_throttle = lambda *args, **kwargs: None
fake_rospy.logwarn = lambda *args, **kwargs: None
fake_rospy.logwarn_throttle = lambda *args, **kwargs: None
fake_rospy.signal_shutdown = lambda *args, **kwargs: setattr(fake_rospy, "_shutdown", True)
fake_rospy.Time = type("Time", (), {"now": staticmethod(lambda: types.SimpleNamespace(to_sec=lambda: 0.0))})


class _Rate:
    def __init__(self, hz):
        self.hz = hz

    def sleep(self):
        fake_rospy._shutdown = True


fake_rospy.Rate = _Rate

sys.modules.setdefault("cv_bridge", types.SimpleNamespace(CvBridge=lambda: object()))
sensor_msgs = sys.modules.setdefault("sensor_msgs", types.SimpleNamespace())
sensor_msgs_msg = sys.modules.setdefault("sensor_msgs.msg", types.SimpleNamespace(Image=object))
sensor_msgs.msg = sensor_msgs_msg


class _Listener:
    def __init__(self, on_press=None):
        self.on_press = on_press

    def start(self):
        return None


sys.modules.setdefault(
    "pynput",
    types.SimpleNamespace(keyboard=types.SimpleNamespace(Listener=_Listener)),
)

from xarm_quest_teleop.eval import eval_runner as eval_runner_module
from xarm_quest_teleop.eval.eval_runner import EvalRunner, RunState

eval_runner_module.rospy = fake_rospy


@dataclass
class _RobotSync:
    robot_match_window_s: float = 0.05


@dataclass
class _Cfg:
    result_log_dir: str
    task_name: str = "cleanup_table_d2"
    model_name: str = "test_policy"
    eval_name: str = "test_eval"
    seed: int = 0
    record: bool = False
    debug_no_actuate: bool = False
    live_viz: bool = False
    rgb_cams: list = field(default_factory=lambda: ["d405", "d435i_front"])
    obs_horizon: int = 2
    pred_horizon: int = 16
    exec_horizon: int = 8
    control_hz: float = 10.0
    obs_stale_s: float = 0.20
    n_rollouts: int = 20
    horizon: int = 1000
    robot_sync: _RobotSync = field(default_factory=_RobotSync)


class _Net:
    def __init__(self):
        self.calls = []
        self.last_visual_focus_records = []

    def infer_action(self, temporal_obs):
        self.calls.append(temporal_obs)
        action = np.zeros((16, 10), dtype=np.float32)
        action[:, 3:9] = np.asarray([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        return action


class _Executor:
    def __init__(self):
        self.calls = []
        self.reset_calls = 0

    def compute_exec_slice(self):
        return 1, 9

    def execute_chunk(self, chunk):
        self.calls.append(np.asarray(chunk).copy())
        return True

    def reset(self):
        self.reset_calls += 1


def _runner(tmp_path, *, debug_no_actuate=False):
    fake_rospy._shutdown = False
    cfg = _Cfg(result_log_dir=str(tmp_path), debug_no_actuate=debug_no_actuate)
    net = _Net()
    runner = EvalRunner(cfg=cfg, robot=object(), net=net, sample_ring=object())
    runner.state = RunState(running=True, continue_requested=True)
    runner.action_executor = _Executor()
    runner._build_temporal_obs = lambda: (0.0, {"rgb": {}, "low_dim": {}})
    return runner, net, runner.action_executor


def test_control_loop_calls_policy_then_executor(tmp_path):
    runner, net, executor = _runner(tmp_path)

    runner._control_loop()

    assert len(net.calls) == 1
    assert len(executor.calls) == 1
    assert executor.calls[0].shape == (8, 10)


def test_debug_no_actuate_skips_executor(tmp_path):
    runner, net, executor = _runner(tmp_path, debug_no_actuate=True)

    runner._control_loop()

    assert len(net.calls) == 1
    assert executor.calls == []
