# src/eval/net_wrapper.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch


class NetBase:
    def __init__(self, ckpt_path: str, device: str = "cuda"):
        self.ckpt_path = ckpt_path
        self.device = device
        self.policy = self.load_model(ckpt_path)
        self.policy.eval().to(self.device)

    def load_model(self, ckpt_path: str) -> torch.nn.Module:
        raise NotImplementedError(
            "Implement NetWrapper.load_model() to construct your diffusion policy and load weights."
        )

    @torch.no_grad()
    def infer_action(self, obs_dict: Dict[str, Any]) -> np.ndarray:
        raise NotImplementedError(
            "Implement NetWrapper.infer_action() to return policy inference results."
        )