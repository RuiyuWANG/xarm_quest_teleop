#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xarm_quest_teleop.policy.seeker_preprocessing import preprocess_real_rgb_image


TASKS = ("cleanup_table_d2", "coffee_transport_d1", "lego_build_d2")
IMAGE_KEY_TO_CAMERA = {
    "agentview_image": "d435i_front",
    "robot0_eye_in_hand_image": "d405",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export center-cropped real-task replay GIFs and init-distribution GIFs "
            "from converted real caches."
        )
    )
    parser.add_argument(
        "--data-root",
        default="data/lmdbs",
        help="Root containing <task>/meta.json and images.lmdb caches.",
    )
    parser.add_argument(
        "--raw-data-root",
        default="data",
        help="Fallback root containing <task>/rgb/episode_xxxx decoded frames.",
    )
    parser.add_argument(
        "--output-root",
        default="real_task_gifs",
        help="Output directory.",
    )
    parser.add_argument(
        "--task",
        action="append",
        choices=TASKS,
        help="Task to export. Repeatable. Defaults to all real tasks.",
    )
    parser.add_argument(
        "--image-key",
        default="agentview_image",
        help="Policy-facing RGB key to render.",
    )
    parser.add_argument(
        "--trajs-per-task",
        type=int,
        default=3,
        help="Number of replay GIFs to export per task.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for trajectory selection.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=256,
        help="Output square GIF size.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=4,
        help="Replay GIF frame stride. Use 1 for every frame.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="Replay GIF playback FPS.",
    )
    parser.add_argument(
        "--init-fps",
        type=float,
        default=4.0,
        help="Init-distribution GIF playback FPS.",
    )
    parser.add_argument(
        "--max-init-frames",
        type=int,
        default=0,
        help="Limit init-distribution frames per task. 0 means all episodes.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rewrite existing GIFs.",
    )
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return (WORKSPACE_ROOT / path).resolve()


def resolve_cache_dir(data_root: Path, task: str) -> Path:
    task_dir = data_root / task
    candidates = [
        task_dir,
        task_dir / "rgb_lmdb",
        task_dir / f"{task}_lmdb",
    ]
    for candidate in candidates:
        if (candidate / "meta.json").is_file() and (candidate / "images.lmdb").exists():
            return candidate
    raise FileNotFoundError(
        f"No real cache found for task={task!r} under {task_dir}. "
        "Expected meta.json and images.lmdb."
    )


def load_meta(cache_dir: Path) -> dict:
    with (cache_dir / "meta.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def episode_starts(lengths: list[int]) -> list[int]:
    return np.cumsum([0, *lengths[:-1]]).astype(int).tolist()


def choose_episodes(n_demo: int, count: int, seed: int) -> list[int]:
    count = min(max(int(count), 0), int(n_demo))
    rng = random.Random(int(seed))
    return sorted(rng.sample(range(int(n_demo)), count))


def decode_jpeg_rgb(buf) -> np.ndarray:
    raw = np.frombuffer(bytes(buf), dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("Failed to decode JPEG bytes from LMDB")
    # Real caches are written by passing already-RGB arrays to cv2.imencode.
    # Decoding with OpenCV returns the same channel order expected by
    # RealWorldDataset, so an extra BGR->RGB conversion would swap red/blue.
    return image


def center_preprocess(image_rgb: np.ndarray, camera_name: str, size: int) -> np.ndarray:
    return preprocess_real_rgb_image(
        image_rgb,
        camera_name=camera_name,
        target_size=int(size),
    )


def read_lmdb_episode_frames(
    *,
    txn,
    image_key: str,
    start: int,
    length: int,
    stride: int,
    size: int,
) -> list[np.ndarray]:
    frames = []
    stride = max(1, int(stride))
    for global_idx in range(int(start), int(start) + int(length), stride):
        key = f"{image_key}/{global_idx:08d}".encode("ascii")
        buf = txn.get(key)
        if buf is None:
            raise KeyError(f"Missing LMDB image key: {key!r}")
        frame = decode_jpeg_rgb(buf)
        if frame.shape[0] != int(size) or frame.shape[1] != int(size):
            frame = cv2.resize(frame, (int(size), int(size)), interpolation=cv2.INTER_AREA)
        frames.append(frame.astype(np.uint8, copy=False))
    return frames


def local_rgb_dir(raw_data_root: Path, task: str, source_episode: int, camera_name: str) -> Path:
    return raw_data_root / task / "rgb" / f"episode_{int(source_episode):04d}" / camera_name / "rgb"


def sorted_image_files(path: Path) -> list[Path]:
    files = []
    for pattern in ("*.png", "*.jpg", "*.jpeg"):
        files.extend(path.glob(pattern))
    return sorted(files)


def read_local_episode_frames(
    *,
    raw_data_root: Path,
    task: str,
    source_episode: int,
    camera_name: str,
    stride: int,
    size: int,
) -> list[np.ndarray]:
    rgb_dir = local_rgb_dir(raw_data_root, task, source_episode, camera_name)
    files = sorted_image_files(rgb_dir)
    if not files:
        raise FileNotFoundError(f"No RGB frames found in {rgb_dir}")
    stride = max(1, int(stride))
    frames = []
    for path in files[::stride]:
        image = np.asarray(Image.open(path).convert("RGB"))
        frames.append(center_preprocess(image, camera_name, size))
    return frames


def read_local_first_frame(
    *,
    raw_data_root: Path,
    task: str,
    source_episode: int,
    camera_name: str,
    size: int,
) -> np.ndarray:
    rgb_dir = local_rgb_dir(raw_data_root, task, source_episode, camera_name)
    files = sorted_image_files(rgb_dir)
    if not files:
        raise FileNotFoundError(f"No RGB frames found in {rgb_dir}")
    image = np.asarray(Image.open(files[0]).convert("RGB"))
    return center_preprocess(image, camera_name, size)


def write_gif(path: Path, frames: list[np.ndarray], fps: float, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        return
    if not frames:
        raise ValueError(f"No frames to write for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = int(round(1000.0 / max(1.0, float(fps))))
    pil_frames = [Image.fromarray(frame.astype(np.uint8, copy=False)) for frame in frames]
    pil_frames[0].save(
        path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
    )


def lmdb_context(cache_dir: Path):
    try:
        import lmdb
    except ModuleNotFoundError:
        return None, None
    lmdb_path = cache_dir / "images.lmdb"
    env = lmdb.open(
        str(lmdb_path),
        subdir=lmdb_path.is_dir(),
        readonly=True,
        lock=False,
        readahead=False,
        max_readers=64,
    )
    txn = env.begin(write=False, buffers=True)
    return env, txn


def export_task(
    *,
    task: str,
    data_root: Path,
    raw_data_root: Path,
    output_root: Path,
    image_key: str,
    trajs_per_task: int,
    seed: int,
    size: int,
    stride: int,
    fps: float,
    init_fps: float,
    max_init_frames: int,
    overwrite: bool,
) -> dict:
    cache_dir = resolve_cache_dir(data_root, task)
    meta = load_meta(cache_dir)
    rgb_keys = set(map(str, meta.get("rgb_keys", [])))
    if image_key not in rgb_keys:
        raise KeyError(f"{task}: image_key={image_key!r} not in rgb_keys={sorted(rgb_keys)}")

    camera_name = IMAGE_KEY_TO_CAMERA.get(str(image_key), str(image_key))
    lengths = [int(x) for x in meta["episode_lengths"]]
    starts = episode_starts(lengths)
    source_episode_ids = [int(x) for x in meta.get("source_demo_indices", range(len(lengths)))]
    selected = choose_episodes(len(lengths), trajs_per_task, seed)
    task_out = output_root / task
    replay_out = task_out / "replay"
    init_out = task_out / "init_distribution.gif"

    env = txn = None
    source = "lmdb"
    try:
        env, txn = lmdb_context(cache_dir)
        if txn is None:
            source = "raw_rgb"

        manifest = {
            "task": task,
            "cache_dir": str(cache_dir),
            "source": source,
            "image_key": image_key,
            "camera_name": camera_name,
            "size": int(size),
            "stride": int(stride),
            "selected_episodes": [],
            "init_distribution_gif": str(init_out),
        }

        for episode_id in selected:
            source_episode = int(source_episode_ids[episode_id])
            if txn is not None:
                frames = read_lmdb_episode_frames(
                    txn=txn,
                    image_key=image_key,
                    start=starts[episode_id],
                    length=lengths[episode_id],
                    stride=stride,
                    size=size,
                )
            else:
                frames = read_local_episode_frames(
                    raw_data_root=raw_data_root,
                    task=task,
                    source_episode=source_episode,
                    camera_name=camera_name,
                    stride=stride,
                    size=size,
                )
            out_path = replay_out / f"episode_{episode_id:04d}_source_{source_episode:04d}.gif"
            write_gif(out_path, frames, fps, overwrite)
            manifest["selected_episodes"].append(
                {
                    "episode_id": int(episode_id),
                    "source_episode": int(source_episode),
                    "length": int(lengths[episode_id]),
                    "gif": str(out_path),
                }
            )

        init_episode_ids = list(range(len(lengths)))
        if max_init_frames > 0 and len(init_episode_ids) > int(max_init_frames):
            init_episode_ids = choose_episodes(
                len(lengths),
                int(max_init_frames),
                seed + 10007,
            )
        init_frames = []
        for episode_id in init_episode_ids:
            source_episode = int(source_episode_ids[episode_id])
            if txn is not None:
                init_frames.extend(
                    read_lmdb_episode_frames(
                        txn=txn,
                        image_key=image_key,
                        start=starts[episode_id],
                        length=1,
                        stride=1,
                        size=size,
                    )
                )
            else:
                init_frames.append(
                    read_local_first_frame(
                        raw_data_root=raw_data_root,
                        task=task,
                        source_episode=source_episode,
                        camera_name=camera_name,
                        size=size,
                    )
                )
        write_gif(init_out, init_frames, init_fps, overwrite)
        manifest["init_distribution_count"] = len(init_frames)

        task_out.mkdir(parents=True, exist_ok=True)
        with (task_out / "manifest.json").open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
            f.write("\n")
        print(f"[ok] {task}: replay={len(selected)} init={len(init_frames)} -> {task_out}")
        return manifest
    finally:
        if txn is not None:
            txn.abort()
        if env is not None:
            env.close()


def main() -> int:
    args = parse_args()
    tasks = tuple(args.task or TASKS)
    data_root = resolve_path(args.data_root)
    raw_data_root = resolve_path(args.raw_data_root)
    output_root = resolve_path(args.output_root)

    manifests = []
    for offset, task in enumerate(tasks):
        manifests.append(
            export_task(
                task=task,
                data_root=data_root,
                raw_data_root=raw_data_root,
                output_root=output_root,
                image_key=str(args.image_key),
                trajs_per_task=int(args.trajs_per_task),
                seed=int(args.seed) + offset,
                size=int(args.size),
                stride=int(args.stride),
                fps=float(args.fps),
                init_fps=float(args.init_fps),
                max_init_frames=int(args.max_init_frames),
                overwrite=bool(args.overwrite),
            )
        )

    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump({"tasks": manifests}, f, indent=2)
        f.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
