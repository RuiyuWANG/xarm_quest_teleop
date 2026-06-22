from __future__ import annotations

from xarm_quest_teleop.policy.registry import create_policy, get_policy_factory, register_policy


class _Policy:
    def __init__(self, value=1):
        self.value = value
        self.n_obs_steps = 1
        self.n_action_steps = 1
        self.task_name = "test"
        self.last_visual_focus_records = []

    def infer_action(self, obs_dict):
        return obs_dict


def test_register_policy_factory():
    register_policy("unit_test_policy", _Policy)

    factory = get_policy_factory("unit_test_policy")
    policy = create_policy("unit_test_policy", value=7)

    assert factory is _Policy
    assert policy.value == 7

