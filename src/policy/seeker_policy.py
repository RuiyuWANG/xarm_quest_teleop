# src/eval/net_wrapper.py
from __future__ import annotations

import os
import copy
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import random

import hydra
import dill
from omegaconf import OmegaConf

from seeker.workspace.base_workspace import BaseWorkspace
from seeker import REPO_ROOT
from seeker.dataset.cache import load_metadata
from seeker.util.formatting import pretty_print_nested
from seeker.util.task_meta import (
    setup_task_embedding_cache,
    instruction_to_task_embedding,
    instruction_to_task_language_tokens,
)

from src.policy.network_base import NetBase
from src.policy.seeker_preprocessing import (
    REAL_POLICY_IMAGE_SIZE,
    REAL_POLICY_ROT_DIM,
    preprocess_real_obs_for_policy,
    task_name_to_instruction,
)
from src.utils.tree_utils import dict_apply

default_shape_meta = {
    "obs": {
        "agentview_image": {
            "shape": [3, 240, 240],
            "type": "rgb",
        },
        "robot0_eye_in_hand_image": {
            "shape": [3, 240, 240],
            "type": "rgb",
        },
        "robot0_eef_pos": {
            "shape": [3],
        },
        "robot0_eef_rot": {
            "shape": [9],
        },
        "robot0_gripper_qpos": {
            "shape": [2],
        },
    },
    "action": {
        "shape": [10],
    },
}

eval_shape_meta = {
    "rgb": {
        "d405": {
            "shape": [3, 256, 256],
            "type": "rgb",
        },
        "d435i_front": {
            "shape": [3, 256, 256],
            "type": "rgb",
        },
    },
    "low_dim": {
        "ee_pose6": {
            "shape": [6],
        },
        "gripper_state": {
            "shape": [1],
        },
    }
}

def try_task_embedding(task_description: str) -> Optional[np.ndarray]:
    """
    Returns: [N, D] float32 if available, else None.
    """
    print("Loading task embeddings...")
    setup_task_embedding_cache()
    emb = instruction_to_task_embedding(task_description)
    emb = emb.cpu().numpy().astype(np.float32)
    return _normalize_task_embedding(emb)


def _normalize_task_embedding(emb: np.ndarray) -> np.ndarray:
    emb = np.asarray(emb, dtype=np.float32)
    emb = np.squeeze(emb)
    if emb.ndim == 1:
        return emb[None, :]
    if emb.ndim == 2:
        return emb
    raise ValueError(f"Expected task embedding [D] or [N,D], got {emb.shape}")


def try_task_language_tokens(task_description: str) -> Optional[np.ndarray]:
    print("Loading task language token embeddings...")
    setup_task_embedding_cache()
    tokens = instruction_to_task_language_tokens(task_description)
    return _normalize_task_language_tokens(tokens.cpu().numpy().astype(np.float32))


def _normalize_task_language_tokens(tokens: np.ndarray) -> np.ndarray:
    tokens = np.asarray(tokens, dtype=np.float32)
    tokens = np.squeeze(tokens)
    if tokens.ndim != 2:
        raise ValueError(f"Expected task language tokens [77,D], got {tokens.shape}")
    if tokens.shape[0] == 77:
        return tokens
    if tokens.shape[1] == 77:
        return tokens.T.copy()
    raise ValueError(f"Expected task language tokens with sequence length 77, got {tokens.shape}")


def _truthy(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _compact_steps(steps: int) -> str:
    steps = int(steps)
    if steps % 1000 == 0:
        return f"{steps // 1000}k"
    if steps > 1000:
        return f"{steps / 1000:g}k"
    return str(steps)


def _attn_prior_suffix(enabled, weight, sigma, decay_steps) -> str:
    if not _truthy(enabled):
        return ""
    decay = int(decay_steps)
    decay_part = f"_decay-{_compact_steps(decay)}" if decay > 0 else ""
    return f"_attnprior-w{float(weight):g}-sig{float(sigma):g}{decay_part}"


def _register_checkpoint_resolvers() -> None:
    resolvers = {
        "eval": eval,
        "add_int": lambda a, b: int(a) + int(b),
        "divide": lambda a, b: float(a) / float(b),
        "get_max_steps": lambda task_name: {
            "square_d0": 400,
            "stack_d1": 400,
            "stack_three_d1": 400,
            "square_d2": 400,
            "threading_d2": 400,
            "coffee_d2": 400,
            "three_piece_assembly_d2": 500,
            "hammer_cleanup_d1": 500,
            "mug_cleanup_d1": 500,
            "kitchen_d1": 800,
            "nut_assembly_d0": 500,
            "pick_place_d0": 1000,
            "coffee_preparation_d1": 800,
            "tool_hang": 700,
            "can": 400,
            "lift": 400,
            "square": 400,
        }.get(str(task_name), 800),
        "demo_label": lambda n_demo: "all" if n_demo is None else str(n_demo),
        "focus_refine_iters_suffix": lambda pooling, iters: (
            f"_iters-{int(iters)}" if str(pooling) == "focus_refine" else ""
        ),
        "attn_prior_suffix": _attn_prior_suffix,
        "overlay_suffix": lambda enabled: "_overlay" if _truthy(enabled) else "",
        "no_eih_suffix": lambda enabled: "" if _truthy(enabled) else "_no_eih",
        "no_eih_tag": lambda enabled: "" if _truthy(enabled) else "no_eih",
        "overlay_prob": lambda enabled, prob: (
            float(prob) if _truthy(enabled) else 0.0
        ),
    }
    for name, resolver in resolvers.items():
        OmegaConf.register_new_resolver(name, resolver, replace=True)


class SeekerPolicy(NetBase):
    def __init__(self, ckpt_path: str, seed: int, device: str = "cuda"):
        super().__init__(ckpt_path, device)
        self.n_obs_steps = int(getattr(self.policy, "n_obs_steps", 1))
        self.n_action_steps = int(getattr(self.policy, "n_action_steps", 1))
        self.pred_horizon = int(getattr(self.policy, "horizon", self.n_action_steps))
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        self.task_name = str(self._cfg_select("task_name", "unknown"))
        self.task_embedding, self.task_language_tokens = self._load_task_context()
        self.robot_id = 0
        self.last_visual_focus_records = []
        self._print_eval_preprocessing_summary()

    def load_model(self, ckpt_path):
        _register_checkpoint_resolvers()
        with open(ckpt_path, "rb") as f:
            payload = torch.load(
                f,
                pickle_module=dill,
                map_location=self.device,
            )
        if "cfg" not in payload:
            raise KeyError(f"Checkpoint is missing payload['cfg']: {ckpt_path}")

        cfg = payload["cfg"]
        if not OmegaConf.is_config(cfg):
            cfg = OmegaConf.create(cfg)
        try:
            OmegaConf.resolve(cfg)
        except Exception as exc:
            print(f"[SeekerPolicy] warning: could not fully resolve checkpoint cfg: {exc}")

        self._resolve_relative_weight_paths(cfg)
        cls = hydra.utils.get_class(cfg._target_)
        workspace = cls(cfg, output_dir=None)
        workspace: BaseWorkspace
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)

        policy = workspace.model
        if getattr(cfg.training, "use_ema", False):
            policy = workspace.ema_model

        self._print_loaded_runtime_config(policy, cfg, ckpt_path)
            
        self.cfg = cfg
        return policy

    def _cfg_select(self, key: str, default: Any = None) -> Any:
        try:
            value = OmegaConf.select(self.cfg, key, default=default)
        except Exception:
            value = default
        return default if value is None else value

    def _print_loaded_runtime_config(self, policy, cfg, ckpt_path: str) -> None:
        if hasattr(policy, "get_runtime_config"):
            runtime_cfg = copy.deepcopy(policy.get_runtime_config())
        else:
            runtime_cfg = {}
        runtime_cfg["Checkpoint"] = {
            "Path": os.path.abspath(os.path.expanduser(str(ckpt_path))),
            "Config Source": "checkpoint payload cfg",
            "Task": OmegaConf.select(cfg, "task_name", default="unknown"),
            "Dataset": OmegaConf.select(cfg, "dataset_path", default=None),
            "EMA": bool(OmegaConf.select(cfg, "training.use_ema", default=False)),
        }
        pretty_print_nested(
            runtime_cfg,
            title="Loaded Checkpoint Runtime Configuration",
            pad_before=True,
            pad_after=True,
        )

    def _resolve_relative_weight_paths(self, cfg) -> None:
        try:
            source = cfg.policy.obs_encoder.focus_view_transform.source
        except Exception:
            return
        for key in ("checkpoint", "weights"):
            path = source.get(key, None) if hasattr(source, "get") else None
            if not path:
                continue
            path = str(path)
            if os.path.isabs(path):
                continue
            repo_path = os.path.join(str(REPO_ROOT), path)
            if os.path.exists(repo_path):
                source[key] = repo_path

    def _resolve_checkpoint_path(self, path_value: Any) -> Optional[Path]:
        if path_value is None:
            return None
        path = Path(str(path_value)).expanduser()
        if not path.is_absolute():
            path = Path(REPO_ROOT) / path
        return path.resolve()

    def _checkpoint_cache_dir(self) -> Optional[Path]:
        cache_dir = self._resolve_checkpoint_path(self._cfg_select("cache_dir", None))
        if cache_dir is not None and (cache_dir / "meta.json").exists():
            return cache_dir

        dataset_path = self._resolve_checkpoint_path(self._cfg_select("dataset_path", None))
        if dataset_path is None:
            dataset_path = self._resolve_checkpoint_path(
                self._cfg_select("task.dataset.dataset_path", None)
            )
        if dataset_path is None:
            return None

        candidates = [
            dataset_path,
            dataset_path / "rgb_lmdb",
            dataset_path / f"{dataset_path.name}_lmdb",
        ]
        for candidate in candidates:
            if (candidate / "meta.json").exists() and (candidate / "arrays.npz").exists():
                return candidate
        return None

    def _load_task_context(self) -> tuple[np.ndarray, Optional[np.ndarray]]:
        cache_dir = self._checkpoint_cache_dir()
        if cache_dir is not None:
            try:
                arrays_path = cache_dir / "arrays.npz"
                with np.load(arrays_path, allow_pickle=True) as arrays:
                    task_embedding_arr = arrays["lowdim/task_embedding"][0]
                    task_language_tokens_arr = (
                        arrays["lowdim/task_language_tokens"][0]
                        if "lowdim/task_language_tokens" in arrays
                        else None
                    )
                task_embedding = np.asarray(
                    task_embedding_arr,
                    dtype=np.float32,
                )
                if task_embedding.ndim == 2 and task_embedding.shape[0] == 1:
                    task_embedding = task_embedding[0]
                task_embedding = _normalize_task_embedding(task_embedding)

                task_language_tokens = None
                if task_language_tokens_arr is not None:
                    task_language_tokens = _normalize_task_language_tokens(
                        np.asarray(task_language_tokens_arr, dtype=np.float32)
                    )

                print(f"[SeekerPolicy] task context loaded from checkpoint dataset cache: {cache_dir}")
                return task_embedding.astype(np.float32), task_language_tokens
            except Exception as exc:
                print(
                    "[SeekerPolicy] warning: failed to load task context from "
                    f"checkpoint dataset cache {cache_dir}: {exc}"
                )

        instruction = None
        if cache_dir is not None:
            try:
                meta, _ = load_metadata(str(cache_dir))
                instruction = meta.get("task_instruction", None)
            except Exception:
                instruction = None
        if instruction is None:
            instruction = task_name_to_instruction(self.task_name)
        print(
            "[SeekerPolicy] warning: recomputing task context from instruction "
            f"instead of checkpoint cache arrays: {instruction!r}"
        )
        return (
            try_task_embedding(instruction).astype(np.float32),
            try_task_language_tokens(instruction),
        )

    def _shape_meta_obs(self):
        shape_meta = getattr(self.cfg, "shape_meta", None)
        if shape_meta is None and hasattr(self.cfg, "task"):
            shape_meta = getattr(self.cfg.task, "shape_meta", None)
        if shape_meta is None:
            return {}
        return shape_meta.get("obs", {}) if hasattr(shape_meta, "get") else {}

    def _policy_image_size(self) -> int:
        obs_meta = self._shape_meta_obs()
        image_meta = obs_meta.get("agentview_image", {}) if hasattr(obs_meta, "get") else {}
        shape = image_meta.get("shape", None) if hasattr(image_meta, "get") else None
        if shape is not None and len(shape) >= 3:
            return int(shape[-1])
        return int(REAL_POLICY_IMAGE_SIZE)

    def _policy_rot_dim(self) -> int:
        obs_meta = self._shape_meta_obs()
        rot_meta = obs_meta.get("robot0_eef_rot", {}) if hasattr(obs_meta, "get") else {}
        shape = rot_meta.get("shape", None) if hasattr(rot_meta, "get") else None
        if shape is not None and len(shape) >= 1:
            return int(shape[-1])
        return int(REAL_POLICY_ROT_DIM)

    def _print_eval_preprocessing_summary(self) -> None:
        obs_encoder = getattr(self.policy, "obs_encoder", None)
        focus = getattr(obs_encoder, "focus_view_transform", None)
        focus_source = getattr(focus, "focus_source", "none")
        view_modes = getattr(focus, "view_modes", {})
        vit_in = getattr(focus, "vit_in", None)
        low_res = getattr(focus, "low_res", None)
        out_res = getattr(focus, "out_res", None)
        print(
            "[SeekerPolicy] eval preprocessing: "
            f"real_camera_crop=enabled "
            f"policy_image_size={self._policy_image_size()} "
            f"rot_dim={self._policy_rot_dim()} "
            f"focus_source={focus_source} "
            f"view_modes={dict(view_modes)} "
            f"vit_in={vit_in} low_res={low_res} out_res={out_res}"
        )

    def format_policy_input(self, obs_dict):
        return preprocess_real_obs_for_policy(
            obs_dict,
            target_size=self._policy_image_size(),
            rot_dim=self._policy_rot_dim(),
        )

    def _match_policy_obs_horizon(self, obs_dict):
        expected = int(self.n_obs_steps)
        if expected <= 0:
            return obs_dict

        first_value = next(iter(obs_dict.values()))
        obs_steps = int(first_value.shape[1])
        if obs_steps == expected:
            return obs_dict

        out = {}
        for key, value in obs_dict.items():
            if obs_steps < expected:
                pad = np.repeat(value[:, :1, ...], expected - obs_steps, axis=1)
                out[key] = np.concatenate([pad, value], axis=1)
            else:
                out[key] = value[:, -expected:, ...]
        return out
    
    @torch.no_grad()
    def infer_action(self, np_obs_dict):
        # Not using past action
        np_obs_dict = self.format_policy_input(np_obs_dict)
        np_obs_dict = self._match_policy_obs_horizon(np_obs_dict)
        first_value = next(iter(np_obs_dict.values()))
        batch_size, obs_steps = first_value.shape[:2]
        task_embedding = _normalize_task_embedding(self.task_embedding)
        # Match RealWorldDataset batches: [B, T, ...]. Seeker flattens internally.
        np_obs_dict["task_embedding"] = np.broadcast_to(
            task_embedding[:, None, :],
            (batch_size, obs_steps, task_embedding.shape[-1]),
        ).copy()
        np_obs_dict["robot_id"] = np.full(
            (batch_size, obs_steps, 1),
            int(self.robot_id),
            dtype=np.float32,
        )
        if self.task_language_tokens is not None:
            tokens = np.asarray(self.task_language_tokens, dtype=np.float32)
            if tokens.ndim == 2:
                tokens = tokens[None, :, :]
            np_obs_dict["task_language_tokens"] = np.broadcast_to(
                tokens[:, None, :, :],
                (batch_size, obs_steps, tokens.shape[-2], tokens.shape[-1]),
            ).copy()
        obs_dict = dict_apply(
            np_obs_dict, lambda x: torch.from_numpy(x).to(device=self.device)
        )
        
        with torch.no_grad():
            action_dict = self.policy.predict_action(obs_dict)
        obs_encoder = getattr(self.policy, "obs_encoder", None)
        self.last_visual_focus_records = list(
            getattr(obs_encoder, "last_visual_focus_records", [])
        )
        np_action_dict = dict_apply(
            action_dict, lambda x: x.detach().to("cpu").numpy()
        )
        
        return np_action_dict["action"]
    
    def reset(self):
        self.policy.reset()
    
def test():
    ckpt_path = "/root/catkin_ws/src/experiment/demos-50_agentview-pass_through_eye_in_hand-pass_through_seed-0/checkpoints/latest.ckpt"
    policy = SeekerPolicy(ckpt_path=ckpt_path, seed=0, device="cuda")

if __name__ == "__main__":
    test()
