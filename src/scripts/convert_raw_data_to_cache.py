from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.policy.seeker_preprocessing import (
    REAL_POLICY_IMAGE_SIZE,
    REAL_RGB_TO_POLICY_KEY,
    pose6_to_xyz_rot,
    preprocess_real_rgb_image,
    task_name_to_instruction,
)
from src.utils.conversion_utils import (
    REAL_FRONT_CROP_RATIO,
    action_abs_to_xyz6g,
    adjust_intrinsics_for_image_geometry,
)


ARRAYS_ARCHIVE_NAME = "arrays.npz"
BUILD_DONE_NAME = "build_done.flag"
DEFAULT_CAMS = ["d435i_front", "d405"]
# Delta action cache export is intentionally disabled; rollout consumes absolute actions.
# DEFAULT_DELTA_HORIZONS = [16]
DEFAULT_IMAGE_SIZE = REAL_POLICY_IMAGE_SIZE
DEFAULT_JPEG_QUALITY = 95
DEFAULT_LMDB_MAP_SIZE_GB = 16
DEFAULT_COMMIT_EVERY = 5000
DEFAULT_TRANS_THRESH_MM = 3.0
DEFAULT_ROT_THRESH_RAD = 0.02
DEFAULT_GRIP_THRESH = 2.0
DEFAULT_IMAGE_WORKERS = max(1, min(8, (os.cpu_count() or 4)))
DEFAULT_POSITION_SCALE = 0.001
DEFAULT_FRONT_CROP_RATIO = REAL_FRONT_CROP_RATIO


def archive_key(rel_path: str | os.PathLike) -> str:
    key = str(rel_path).replace(os.sep, "/")
    if key.endswith(".npy"):
        key = key[: -len(".npy")]
    return key


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


# Delta action cache export is intentionally disabled. If it is re-enabled, build
# rot6d with src.utils.conversion_utils.R_to_rot6d instead of row-major slicing.
# def _delta_action_chunks(...):
#     ...


def still_frame_keep_indices(
    ee_pose6: np.ndarray,
    gripper: np.ndarray,
    trans_thresh_mm: float,
    rot_thresh_rad: float,
    grip_thresh: float,
) -> np.ndarray:
    ee_pose6 = np.asarray(ee_pose6, dtype=np.float32)
    gripper = np.asarray(gripper, dtype=np.float32).reshape(-1)
    n_frames = int(ee_pose6.shape[0])
    if n_frames == 0:
        return np.zeros((0,), dtype=np.int64)

    keep = [0]
    last_keep = 0
    for t in range(1, n_frames):
        trans = float(np.linalg.norm(ee_pose6[t, :3] - ee_pose6[last_keep, :3]))
        r_last = R.from_euler("xyz", ee_pose6[last_keep, 3:6], degrees=False)
        r_now = R.from_euler("xyz", ee_pose6[t, 3:6], degrees=False)
        rot = float((r_last.inv() * r_now).magnitude())
        grip = float(abs(gripper[t] - gripper[last_keep]))

        if trans > trans_thresh_mm or rot > rot_thresh_rad or grip > grip_thresh:
            keep.append(t)
            last_keep = t

    if keep[-1] != n_frames - 1:
        keep.append(n_frames - 1)

    return np.asarray(keep, dtype=np.int64)


def encode_frame_images(
    ep_path: str,
    raw_t: int,
    cams: Tuple[str, ...],
    image_size: int,
    jpeg_quality: int,
) -> Tuple[int, Dict[str, bytes]]:
    ep = Path(ep_path)
    encoded: Dict[str, bytes] = {}
    for cam in cams:
        img_path = ep / cam / "rgb" / f"{int(raw_t):06d}.png"
        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f"Failed to read image: {img_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb = preprocess_real_rgb_image(
            img_rgb,
            camera_name=cam,
            target_size=image_size,
        )
        ok, jpg = cv2.imencode(
            ".jpg",
            img_rgb,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
        )
        if not ok:
            raise RuntimeError("cv2.imencode(.jpg) failed")
        encoded[cam] = jpg.tobytes()
    return int(raw_t), encoded


def build_camera_matrix_arrays(
    *,
    task_meta: dict,
    cams: List[str],
    image_size: int,
    eef_pos: np.ndarray,
    eef_rot9: np.ndarray,
    position_scale: float,
    front_crop_ratio: float,
) -> Dict[str, np.ndarray]:
    calibration = task_meta.get("config", {}).get("calibration", {})
    if not calibration:
        return {}

    n_samples = int(np.asarray(eef_pos).shape[0])
    out: Dict[str, np.ndarray] = {}
    for camera_name in cams:
        camera_cfg = calibration.get(camera_name)
        if camera_cfg is None or "rgb" not in camera_cfg:
            print(f"[oracle] skip {camera_name}: missing calibration", flush=True)
            continue

        rgb_cfg = camera_cfg["rgb"]
        intr = rgb_cfg.get("intrinsics", {})
        K = adjust_intrinsics_for_image_geometry(
            np.asarray(intr["K"], dtype=np.float32),
            camera_name=camera_name,
            src_width=int(intr["width"]),
            src_height=int(intr["height"]),
            image_size=int(image_size),
            square_crop=True,
            front_crop_ratio=float(front_crop_ratio),
        )

        setup = str(camera_cfg.get("setup", "")).strip().lower()
        K4 = np.eye(4, dtype=np.float32)
        K4[:3, :3] = K
        if setup == "eye_to_hand":
            X_W_C = np.asarray(rgb_cfg["extrinsics"]["X_C"], dtype=np.float32)
            matrix = (K4 @ np.linalg.inv(X_W_C)).astype(np.float32)
            matrix[:, :3] *= float(position_scale)
            matrices = np.repeat(matrix[None, :, :], n_samples, axis=0)
        elif setup == "eye_in_hand":
            X_E_C = np.asarray(camera_cfg["extrinsics"]["X_C"], dtype=np.float32)
            ee_pos = np.asarray(eef_pos, dtype=np.float32) * float(position_scale)
            ee_rot = np.asarray(eef_rot9, dtype=np.float32).reshape(-1, 3, 3)
            X_W_E = np.repeat(np.eye(4, dtype=np.float32)[None, :, :], n_samples, axis=0)
            X_W_E[:, :3, :3] = ee_rot
            X_W_E[:, :3, 3] = ee_pos
            X_W_C = X_W_E @ X_E_C[None, :, :]
            matrices = np.stack(
                [
                    (K4 @ np.linalg.inv(pose)).astype(np.float32)
                    for pose in X_W_C
                ],
                axis=0,
            )
            matrices[:, :, :3] *= float(position_scale)
        else:
            raise ValueError(f"Unsupported camera setup for {camera_name}: {setup!r}")

        if matrices.shape != (n_samples, 4, 4):
            raise ValueError(
                f"camera_matrix_{camera_name} shape mismatch: {matrices.shape}"
            )
        policy_key = REAL_RGB_TO_POLICY_KEY[camera_name]
        oracle_name = policy_key[: -len("_image")] if policy_key.endswith("_image") else policy_key
        out[f"camera_matrix_{oracle_name}"] = matrices.astype(
            np.float32,
            copy=False,
        )
    return out


def derive_joint_vel(joint_states: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    joints = np.asarray(joint_states, dtype=np.float32)
    times = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    vel = np.zeros_like(joints, dtype=np.float32)
    if joints.shape[0] <= 1:
        return vel
    dt = np.diff(times)
    finite_dt = dt[np.isfinite(dt) & (dt > 1e-6)]
    fallback_dt = float(np.median(finite_dt)) if finite_dt.size else 1.0
    dt = np.where(np.isfinite(dt) & (dt > 1e-6), dt, fallback_dt).astype(np.float32)
    vel[1:] = (joints[1:] - joints[:-1]) / dt[:, None]
    return vel


def task_embedding(task_instruction: str) -> Tuple[np.ndarray, Optional[np.ndarray], str]:
    try:
        from seeker.util.task_meta import (
            setup_task_embedding_cache,
            instruction_to_task_embedding,
            instruction_to_task_language_tokens,
        )
    except Exception:
        try:
            from attention_seeker.util.task_embedding import setup_task_embedding_cache, instruction_to_task_embedding
        except Exception:
            return np.zeros((1, 0), dtype=np.float32), None, "missing_task_embedding_dependency"

        setup_task_embedding_cache()
        emb = instruction_to_task_embedding([task_instruction]).cpu().numpy().astype(np.float32)
        return emb, None, "attention_seeker"

    setup_task_embedding_cache()
    emb = instruction_to_task_embedding(task_instruction).cpu().numpy().astype(np.float32)
    if emb.ndim == 1:
        emb = emb[None, :]
    tokens = instruction_to_task_language_tokens(task_instruction).cpu().numpy().astype(np.float32)
    if tokens.ndim == 2:
        if tokens.shape[0] != 77 and tokens.shape[1] == 77:
            tokens = tokens.T.copy()
        tokens = tokens[None, :, :]
    return emb, tokens, "seeker.util.task_meta"


def open_lmdb(path: Path, map_size_gb: int):
    try:
        import lmdb
    except ImportError as exc:
        raise ImportError("lmdb is required. Install python-lmdb in the conversion environment.") from exc
    return lmdb.open(
        str(path),
        map_size=int(map_size_gb * (1024 ** 3)),
        subdir=False,
        readonly=False,
        meminit=False,
        map_async=True,
        max_dbs=1,
    )


def convert_task(
    *,
    task_dir: Path,
    out_dir: Path,
    stream: str,
    cams: List[str],
    image_size: int,
    jpeg_quality: int,
    lmdb_map_size_gb: int,
    commit_every: int,
    filter_still_frames: bool,
    trans_thresh_mm: float,
    rot_thresh_rad: float,
    grip_thresh: float,
    image_workers: int,
    n_demo: Optional[int],
    start_index: int,
    overwrite: bool,
) -> None:
    task_dir = task_dir.expanduser().resolve()
    out_dir = out_dir.expanduser().resolve()
    stream_dir = task_dir / stream

    task_meta_path = task_dir / "task_meta.json"
    if task_meta_path.is_file():
        with task_meta_path.open("r", encoding="utf-8") as f:
            task_meta = json.load(f)
    else:
        task_meta = {}
    task_cfg = task_meta.get("config", {}).get("dataset", {})
    task_name = str(task_cfg.get("task", task_dir.name))
    task_instruction = task_name_to_instruction(task_name, default=task_cfg.get("description", task_name))
    task_emb, task_tokens, task_emb_source = task_embedding(task_instruction)

    if not stream_dir.is_dir():
        raise FileNotFoundError(f"Stream dir not found: {stream_dir}")
    demo_prefix = str(task_cfg.get("demo_prefix", "episode_"))
    episodes = [p for p in stream_dir.iterdir() if p.is_dir() and p.name.startswith(demo_prefix)]
    episodes.sort(key=lambda p: int(p.name.split(demo_prefix, 1)[1]))
    if not episodes:
        raise ValueError(f"No episode folders found under {stream_dir}")
    episodes = episodes[int(start_index):]

    required_lowdim_keys = [
        "timestamp",
        "joint_states",
        "ee_pose6",
        "gripper_state",
        "pose_targets6",
        "pose_targets_gripper",
        "deadman_released",
    ]
    valid_episodes: List[Tuple[Path, int, int]] = []
    skipped = 0
    for ep_dir in episodes:
        ok = True
        reason = "ok"
        n_frames = 0
        lowdim_path = ep_dir / "lowdim.npz"
        if not lowdim_path.is_file():
            ok = False
            reason = f"missing lowdim file: {lowdim_path}"
        else:
            try:
                with np.load(lowdim_path, allow_pickle=True) as low:
                    missing = [key for key in required_lowdim_keys if key not in low]
                    if missing:
                        ok = False
                        reason = f"lowdim missing keys: {missing}"
                    else:
                        n_frames = int(np.asarray(low["ee_pose6"]).shape[0])
                        if n_frames <= 0:
                            ok = False
                            reason = "lowdim has zero frames"
                        else:
                            for key in required_lowdim_keys:
                                key_len = int(np.asarray(low[key]).shape[0])
                                if key_len != n_frames:
                                    ok = False
                                    reason = f"lowdim key {key} length {key_len} != {n_frames}"
                                    break

                if ok:
                    for cam in cams:
                        rgb_dir = ep_dir / cam / "rgb"
                        if not rgb_dir.is_dir():
                            ok = False
                            reason = f"missing rgb dir for camera {cam}: {rgb_dir}"
                            break
                        n_png = len([p for p in rgb_dir.iterdir() if p.suffix == ".png"])
                        if n_png != n_frames:
                            ok = False
                            reason = f"camera {cam} has {n_png} png frames != lowdim {n_frames}"
                            break
            except Exception as exc:
                ok = False
                reason = f"validation error: {exc}"

        ep_idx = int(ep_dir.name.split("_")[-1])
        if ok:
            print(f"[scan] valid {ep_dir.name}: frames={n_frames}", flush=True)
            valid_episodes.append((ep_dir, ep_idx, n_frames))
        else:
            print(f"[scan] skip {ep_dir.name}: {reason}", flush=True)
            skipped += 1

    print(f"[scan] valid demos={len(valid_episodes)} skipped={skipped} total={len(episodes)}", flush=True)
    if n_demo is not None:
        valid_episodes = valid_episodes[: int(n_demo)]
    if not valid_episodes:
        raise ValueError("No complete episodes selected for conversion")

    if out_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output dir exists: {out_dir}. Pass --overwrite to replace it.")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = open_lmdb(out_dir / "images.lmdb", lmdb_map_size_gb)
    txn = env.begin(write=True)
    put_count = 0
    global_step = 0

    episode_lengths: List[int] = []
    source_demo_indices: List[int] = []
    eef_pos_all: List[np.ndarray] = []
    eef_rot_all: List[np.ndarray] = []
    gripper_qpos_all: List[np.ndarray] = []
    joint_states_all: List[np.ndarray] = []
    joint_vel_all: List[np.ndarray] = []
    timestamp_all: List[np.ndarray] = []
    deadman_all: List[np.ndarray] = []
    abs_action_all: List[np.ndarray] = []
    # Delta action export is disabled; rollout consumes action/absolute_action.
    # delta_action_all: Dict[int, List[np.ndarray]] = {}

    pbar = (
        tqdm(valid_episodes, desc="Convert realworld cache", unit="demo")
        if tqdm is not None
        else valid_episodes
    )
    total_raw_frames = 0
    total_kept_frames = 0

    for ep_path, ep_index, _ in pbar:
        with np.load(ep_path / "lowdim.npz", allow_pickle=True) as low_npz:
            low = {key: np.asarray(low_npz[key]) for key in low_npz.files}
        raw_ee_pose6 = np.asarray(low["ee_pose6"], dtype=np.float32)
        raw_gripper = np.asarray(low["gripper_state"], dtype=np.float32).reshape(-1)
        keep_idx = (
            still_frame_keep_indices(
                ee_pose6=raw_ee_pose6,
                gripper=raw_gripper,
                trans_thresh_mm=float(trans_thresh_mm),
                rot_thresh_rad=float(rot_thresh_rad),
                grip_thresh=float(grip_thresh),
            )
            if filter_still_frames
            else np.arange(raw_ee_pose6.shape[0], dtype=np.int64)
        )
        total_raw_frames += int(raw_ee_pose6.shape[0])
        total_kept_frames += int(keep_idx.shape[0])
        filter_msg = f"{ep_path.name}: raw={raw_ee_pose6.shape[0]} kept={keep_idx.shape[0]}"
        if tqdm is not None:
            pbar.set_postfix_str(
                f"kept={total_kept_frames}/{total_raw_frames} samples={global_step}"
            )
            tqdm.write(f"[filter] {filter_msg}")
        else:
            print(f"[filter] {filter_msg}", flush=True)

        low = dict(low)
        for key in [
            "timestamp",
            "joint_states",
            "ee_pose6",
            "gripper_state",
            "pose_targets6",
            "pose_targets_gripper",
            "deadman_released",
        ]:
            low[key] = np.asarray(low[key])[keep_idx]

        eef_pos, eef_rot9 = pose6_to_xyz_rot(np.asarray(low["ee_pose6"], dtype=np.float32), rot_dim=9)
        grip = np.asarray(low["gripper_state"], dtype=np.float32).reshape(-1, 1)
        gripper_qpos = np.concatenate([grip / 2.0, -grip / 2.0], axis=-1).astype(np.float32)
        abs_action = action_abs_to_xyz6g(
            np.asarray(low["pose_targets6"], dtype=np.float32),
            np.asarray(low["pose_targets_gripper"], dtype=np.float32),
        ).astype(np.float32)
        # Delta action cache export is intentionally disabled.
        # abs_action_posmat = _absolute_action_posmat(low)

        n_frames = int(eef_pos.shape[0])
        episode_lengths.append(n_frames)
        source_demo_indices.append(ep_index)
        eef_pos_all.append(eef_pos)
        eef_rot_all.append(eef_rot9)
        gripper_qpos_all.append(gripper_qpos)
        joint_states_all.append(np.asarray(low["joint_states"], dtype=np.float32))
        joint_vel_all.append(derive_joint_vel(low["joint_states"], low["timestamp"]))
        timestamp_all.append(np.asarray(low["timestamp"], dtype=np.float64).reshape(-1, 1))
        deadman_all.append(np.asarray(low["deadman_released"], dtype=np.float32).reshape(-1, 1))
        abs_action_all.append(abs_action)
        # for horizon in delta_action_all:
        #     delta_action_all[horizon].append(_delta_action_chunks(eef_pos, eef_rot9, abs_action_posmat, horizon))

        frame_args = [
            (str(ep_path), int(raw_t), tuple(cams), int(image_size), int(jpeg_quality))
            for raw_t in keep_idx.tolist()
        ]
        if int(image_workers) > 1 and len(frame_args) > 1:
            with ThreadPoolExecutor(max_workers=int(image_workers)) as pool:
                encoded_frames = pool.map(lambda args: encode_frame_images(*args), frame_args)
                for _, encoded_by_cam in encoded_frames:
                    for cam in cams:
                        key = f"{REAL_RGB_TO_POLICY_KEY[cam]}/{global_step:08d}".encode("ascii")
                        txn.put(key, encoded_by_cam[cam])
                        put_count += 1
                        if put_count % int(commit_every) == 0:
                            txn.commit()
                            txn = env.begin(write=True)
                    global_step += 1
        else:
            encoded_frames = (encode_frame_images(*args) for args in frame_args)
            for _, encoded_by_cam in encoded_frames:
                for cam in cams:
                    key = f"{REAL_RGB_TO_POLICY_KEY[cam]}/{global_step:08d}".encode("ascii")
                    txn.put(key, encoded_by_cam[cam])
                    put_count += 1
                    if put_count % int(commit_every) == 0:
                        txn.commit()
                        txn = env.begin(write=True)
                global_step += 1
        if tqdm is not None:
            pbar.set_postfix_str(
                f"kept={total_kept_frames}/{total_raw_frames} samples={global_step}"
            )

    txn.put(b"__len__", str(global_step).encode("ascii"))
    txn.commit()
    env.sync()
    env.close()

    eef_pos_cat = np.concatenate(eef_pos_all, axis=0).astype(np.float32)
    eef_rot_cat = np.concatenate(eef_rot_all, axis=0).astype(np.float32)
    camera_arrays = build_camera_matrix_arrays(
        task_meta=task_meta,
        cams=cams,
        image_size=int(image_size),
        eef_pos=eef_pos_cat,
        eef_rot9=eef_rot_cat,
        position_scale=DEFAULT_POSITION_SCALE,
        front_crop_ratio=DEFAULT_FRONT_CROP_RATIO,
    )

    arrays: Dict[str, np.ndarray] = {
        archive_key("lowdim/robot0_eef_pos.npy"): eef_pos_cat,
        archive_key("lowdim/robot0_eef_rot.npy"): eef_rot_cat,
        archive_key("lowdim/robot0_gripper_qpos.npy"): np.concatenate(gripper_qpos_all, axis=0).astype(np.float32),
        archive_key("lowdim/joint_states.npy"): np.concatenate(joint_states_all, axis=0).astype(np.float32),
        archive_key("lowdim/robot0_joint_vel.npy"): np.concatenate(joint_vel_all, axis=0).astype(np.float32),
        archive_key("lowdim/timestamp.npy"): np.concatenate(timestamp_all, axis=0).astype(np.float64),
        archive_key("lowdim/deadman_released.npy"): np.concatenate(deadman_all, axis=0).astype(np.float32),
        archive_key("action/absolute_action.npy"): np.concatenate(abs_action_all, axis=0).astype(np.float32),
        archive_key("lowdim/task_embedding.npy"): np.repeat(task_emb, len(episode_lengths), axis=0).astype(np.float32),
        archive_key("lowdim/robot_id.npy"): np.zeros((len(episode_lengths),), dtype=np.int64),
        archive_key("lowdim/task_id.npy"): np.zeros((len(episode_lengths),), dtype=np.int64),
        archive_key("task_instructions.npy"): np.asarray([task_instruction] * len(episode_lengths), dtype=object),
    }
    if task_tokens is not None:
        arrays[archive_key("lowdim/task_language_tokens.npy")] = np.repeat(
            task_tokens,
            len(episode_lengths),
            axis=0,
        ).astype(np.float32)
    # Delta action arrays are intentionally not written.
    # for horizon, chunks in delta_action_all.items():
    #     arrays[archive_key(f"action/delta_action_h{horizon}.npy")] = np.concatenate(chunks, axis=0).astype(np.float32)
    for key, arr in camera_arrays.items():
        arrays[archive_key(f"oracle/{key}.npy")] = arr
    np.savez(out_dir / ARRAYS_ARCHIVE_NAME, **arrays)

    meta = {
        "cache_format": "realworld_lmdb_npz_v1",
        "realworld_preconverted_actions": True,
        "source_type": "real_collector",
        "source_root": str(task_dir),
        "stream_name": stream,
        "task_name": task_name,
        "task_instruction": task_instruction,
        "task_embedding_source": task_emb_source,
        "dataset": json_safe(task_cfg),
        "rgb_keys": [REAL_RGB_TO_POLICY_KEY[cam] for cam in cams],
        "camera_names": list(cams),
        "lowdim_keys": [
            "robot0_eef_pos",
            "robot0_eef_rot",
            "robot0_gripper_qpos",
        ],
        "extra_lowdim_keys": [
            "joint_states",
            "robot0_joint_vel",
            "timestamp",
            "deadman_released",
        ],
        "action_keys": ["absolute_action"],
        "action_convention": {
            "absolute_action": "[x_mm,y_mm,z_mm,rot6d_first_two_columns,gripper]",
            "rotation_6d": "first_two_columns",
            "delta_action": "disabled",
        },
        "oracle_keys": sorted(camera_arrays.keys()),
        "episode_lengths": [int(x) for x in episode_lengths],
        "n_demo": int(len(episode_lengths)),
        "n_samples": int(sum(episode_lengths)),
        "source_demo_indices": [int(x) for x in source_demo_indices],
        "image_size": int(image_size),
        "square_crop": True,
        "front_crop_ratio": float(DEFAULT_FRONT_CROP_RATIO),
        "filter_still_frames": bool(filter_still_frames),
        "filter": {
            "trans_thresh_mm": float(trans_thresh_mm),
            "rot_thresh_rad": float(rot_thresh_rad),
            "grip_thresh": float(grip_thresh),
            "rule": "keep first/final frames; keep t when consecutive-frame translation, rotation, or gripper changes exceed thresholds",
        },
        "delta_action_horizons": [],
    }
    with (out_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")
    (out_dir / BUILD_DONE_NAME).write_text("build completed\n", encoding="utf-8")
    print(
        f"[built] out_dir={out_dir} valid_converted_demos={meta['n_demo']} samples={meta['n_samples']}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert real-world collector episodes to a cache directly compatible with seeker.dataset.realworld_dataset.RealWorldDataset."
    )
    parser.add_argument("--task-dir", type=str, required=True, help="Task directory containing task_meta.json and stream folders.")
    parser.add_argument("--out-dir", type=str, default=None, help="Output cache dir. Defaults to <task-dir>/rgb_lmdb.")
    parser.add_argument("--stream", type=str, default="rgb", choices=["rgb", "all_sensors"])
    parser.add_argument("--cams", nargs="+", default=DEFAULT_CAMS, choices=list(REAL_RGB_TO_POLICY_KEY.keys()))
    parser.add_argument("--n-demo", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--jpeg-quality", type=int, default=DEFAULT_JPEG_QUALITY)
    parser.add_argument("--lmdb-map-size-gb", type=int, default=DEFAULT_LMDB_MAP_SIZE_GB)
    parser.add_argument("--commit-every", type=int, default=DEFAULT_COMMIT_EVERY)
    # Delta action export is disabled; current rollout consumes absolute actions.
    # parser.add_argument("--delta-horizons", type=int, nargs="+", default=DEFAULT_DELTA_HORIZONS)
    parser.add_argument("--no-filter-still-frames", action="store_true", help="Disable zero/still-frame filtering.")
    parser.add_argument("--trans-thresh-mm", type=float, default=DEFAULT_TRANS_THRESH_MM)
    parser.add_argument("--rot-thresh-rad", type=float, default=DEFAULT_ROT_THRESH_RAD)
    parser.add_argument("--grip-thresh", type=float, default=DEFAULT_GRIP_THRESH)
    parser.add_argument("--image-workers", type=int, default=DEFAULT_IMAGE_WORKERS, help="Parallel workers for PNG decode, preprocessing, and JPEG encode.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    task_dir = Path(args.task_dir)
    out_dir = Path(args.out_dir) if args.out_dir else task_dir / "rgb_lmdb"
    convert_task(
        task_dir=task_dir,
        out_dir=out_dir,
        stream=str(args.stream),
        cams=list(args.cams),
        image_size=int(args.image_size),
        jpeg_quality=int(args.jpeg_quality),
        lmdb_map_size_gb=int(args.lmdb_map_size_gb),
        commit_every=int(args.commit_every),
        filter_still_frames=not bool(args.no_filter_still_frames),
        trans_thresh_mm=float(args.trans_thresh_mm),
        rot_thresh_rad=float(args.rot_thresh_rad),
        grip_thresh=float(args.grip_thresh),
        image_workers=max(1, int(args.image_workers)),
        n_demo=args.n_demo,
        start_index=int(args.start_index),
        overwrite=bool(args.overwrite),
    )


if __name__ == "__main__":
    main()
