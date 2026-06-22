from __future__ import annotations

from typing import Any, Dict, Protocol

import numpy as np


class PolicyBase(Protocol):
    n_obs_steps: int
    n_action_steps: int
    task_name: str
    last_visual_focus_records: list

    def infer_action(self, obs_dict: Dict[str, Any]) -> np.ndarray:
        ...

