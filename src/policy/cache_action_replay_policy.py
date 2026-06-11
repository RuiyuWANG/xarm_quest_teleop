from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np


class CacheActionReplayPolicy:
    """Replay cached absolute actions through the normal EvalRunner path."""

    def __init__(
        self,
        cache_dir: str,
        *,
        episode: int = 0,
        start: int = 0,
        horizon: int = 16,
        advance: int = 8,
        obs_horizon: int = 2,
    ):
        self.cache_dir = Path(os.path.expanduser(cache_dir)).resolve()
        self.episode = int(episode)
        self.cursor = int(start)
        self.n_action_steps = int(horizon)
        self.pred_horizon = int(horizon)
        self.n_obs_steps = int(obs_horizon)
        self.advance = max(1, int(advance))
        self.last_visual_focus_records = []

        meta_path = self.cache_dir / "meta.json"
        arrays_path = self.cache_dir / "arrays.npz"
        if not meta_path.is_file():
            raise FileNotFoundError(f"Missing cache meta: {meta_path}")
        if not arrays_path.is_file():
            raise FileNotFoundError(f"Missing cache arrays: {arrays_path}")

        with meta_path.open("r", encoding="utf-8") as f:
            self.meta = json.load(f)
        episode_lengths = [int(x) for x in self.meta["episode_lengths"]]
        if self.episode < 0 or self.episode >= len(episode_lengths):
            raise IndexError(
                f"Replay episode {self.episode} out of range [0, {len(episode_lengths)})"
            )

        with np.load(arrays_path, allow_pickle=True) as archive:
            self.actions = np.asarray(archive["action/absolute_action"], dtype=np.float32)

        ep_start = int(np.sum(episode_lengths[: self.episode]))
        ep_len = int(episode_lengths[self.episode])
        self.episode_actions = self.actions[ep_start : ep_start + ep_len]
        if self.episode_actions.ndim != 2 or self.episode_actions.shape[1] < 10:
            raise ValueError(
                f"Expected cached absolute actions [T, >=10], got {self.episode_actions.shape}"
            )
        if self.cursor < 0 or self.cursor >= ep_len:
            raise IndexError(f"Replay start {self.cursor} out of episode length {ep_len}")

        print(
            "[CacheActionReplayPolicy] "
            f"cache={self.cache_dir} episode={self.episode} "
            f"episode_len={ep_len} start={self.cursor} "
            f"horizon={self.n_action_steps} advance={self.advance}"
        )
        print(
            "[CacheActionReplayPolicy] first replay action xyz="
            f"{self.episode_actions[self.cursor, :3].round(2)}"
        )

    def infer_action(self, obs_dict: Dict[str, Any]) -> np.ndarray:
        _ = obs_dict
        ep_len = int(self.episode_actions.shape[0])
        idx = np.minimum(
            self.cursor + np.arange(self.n_action_steps, dtype=np.int64),
            ep_len - 1,
        )
        out = self.episode_actions[idx, :10].astype(np.float32, copy=True)
        print(
            "[CacheActionReplayPolicy] replay "
            f"cursor={self.cursor} idx=[{int(idx[0])}:{int(idx[-1])}] "
            f"xyz0={out[0, :3].round(2)} xyz_last={out[-1, :3].round(2)}"
        )
        self.cursor = min(self.cursor + self.advance, ep_len - 1)
        return out[None, :, :]
