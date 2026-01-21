# src/eval/net_wrapper.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, List

import numpy as np
import torch
import random
import cv2

import hydra
import dill

from attention_seeker.workspace.base_workspace import BaseWorkspace
from attention_seeker.util.task_embedding import (
    setup_task_embedding_cache,
    instruction_to_task_embedding,
)

from src.policy.network_base import NetBase
from src.utils.conversion_utils import dict_apply, pose6_to_xyz6, center_square_crop, action_abs_to_xyz6g

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

TASK_DICT = {
    "real_pick_place_veggis_d1": "Pick up he purple toy eggplant and place it in the brown box.",
}

def try_task_embedding(task_descriptions: List[str]) -> Optional[np.ndarray]:
    """
    Returns: [N, D] float32 if available, else None.
    """
    print("Loading task embeddings...")
    setup_task_embedding_cache()
    emb = instruction_to_task_embedding(task_descriptions)
    emb = emb.cpu().numpy().astype(np.float32)
    return emb


class SeekerPolicy(NetBase):
    def __init__(self, ckpt_path: str, seed: int, device: str = "cuda"):
        super().__init__(ckpt_path, device)
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        self.task_embedding = try_task_embedding(TASK_DICT[self.cfg.task_name]).astype(np.float32)
    
    def load_model(self, ckpt_path):
        payload = torch.load(open(ckpt_path, "rb"), pickle_module=dill)
        cfg = payload["cfg"]
        cls = hydra.utils.get_class(cfg._target_)
        workspace = cls(cfg, output_dir=None)
        workspace: BaseWorkspace
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)

        print(f"Action mode: {cfg.action_mode}")
        policy = workspace.model
        if getattr(cfg.training, "use_ema", False):
            policy = workspace.ema_model
            
        self.cfg = cfg
        return policy
    
    def format_policy_input(self, obs_dict):
        """
        Returns a dict matching the policy obs keys:
          - agentview_image:            (T, 3, H, W) float32 in [0,1]
          - robot0_eye_in_hand_image:   (T, 3, H, W) float32 in [0,1]
          - robot0_eef_pos:             (T, 3) float32   (mm)
          - robot0_eef_rot:             (T, 6) float32   (flattened R[:6])
          - robot0_gripper_qpos:        (T, 2) float32   ([g/2, -g/2])
        """
        
        if "rgb" not in obs_dict or "low_dim" not in obs_dict:
            raise ValueError(f"Unknown obs_dict format. Keys={list(obs_dict.keys())}")

        rgb = obs_dict["rgb"]
        low = obs_dict["low_dim"]

        # ---- camera mapping (adjust to your actual keys) ----
        cam_map = {
            "d435i_front": "agentview_image",
            "d405": "robot0_eye_in_hand_image",
        }

        # infer T
        some_cam = next(iter(rgb.keys()))
        T = len(rgb[some_cam])

        out = {}

        # ---- images: HWC RGB uint8 -> CHW float32, crop/resize ----
        for cam_key, seq in rgb.items():
            if cam_key not in cam_map:
                continue
            pol_key = cam_map[cam_key]

            imgs = []
            for t in range(T):
                im = seq[t]  # expected HWC RGB uint8
                im = np.asarray(im)

                if im.ndim != 3 or im.shape[-1] != 3:
                    raise ValueError(f"{cam_key}: expected HWC RGB, got {im.shape}")

                # HARDCODED
                im = center_square_crop(im)
                target = 240
                if (im.shape[0] != target) or (im.shape[1] != target):
                    im = cv2.resize(im, (target, target), interpolation=cv2.INTER_AREA)

                # to float [0,1] and CHW
                im = im.astype(np.float32) / 255.0
                im = np.moveaxis(im, -1, 0)  # CHW
                imgs.append(im)

            imgs_np = np.stack(imgs, axis=0).astype(np.float32)  # (T,3,H,W)
            out[pol_key] = imgs_np[None, ...] # (1,T,3,H,W)

        # ensure required images exist
        if "agentview_image" not in out or "robot0_eye_in_hand_image" not in out:
            raise KeyError(f"Missing required image keys after mapping. Got={list(out.keys())}")

        # ---- lowdim: ee_pose6 -> eef_pos (3) + eef_rot (6) ----
        ee_pose6 = np.asarray(low["ee_pose6"], dtype=np.float32)  # (T,6)
        if ee_pose6.ndim == 1:
            ee_pose6 = ee_pose6[None, :]

        eef_pos, eef_rot6 = pose6_to_xyz6(ee_pose6)  # (T,3), (T,6)

        out["robot0_eef_pos"] = eef_pos[None, ...].astype(np.float32) # (1,T,3)
        out["robot0_eef_rot"] = eef_rot6[None, ...].astype(np.float32) # (1,T,6)

        # ---- gripper: scalar -> (T,2) as in your conversion code ----
        g = np.asarray(low["gripper_state"], dtype=np.float32)
        g = g.reshape(-1, 1)  # (T,1)

        g = g / 2.0
        g_np = np.concatenate([g, -g], axis=-1).astype(np.float32)  # (T,2)
        out["robot0_gripper_qpos"] = g_np[None, ...] # (1,T,2)

        return out
    
    @torch.no_grad()
    def infer_action(self, np_obs_dict):
        # Not using past action
        np_obs_dict = self.format_policy_input(np_obs_dict)
        obs_dict = dict_apply(
            np_obs_dict, lambda x: torch.from_numpy(x).to(device=self.device)
        )
        task_embedding = torch.from_numpy(self.task_embedding.copy()).to(device=self.device)
        if task_embedding.ndim == 1:
            task_embedding = task_embedding[None, ...]
        
        with torch.no_grad():
            action_dict = self.policy.predict_action(
                obs_dict, task_embedding, viz_dir=None
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