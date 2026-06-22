from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from xarm_quest_teleop.utils.conversion_utils import rot6d_to_R


DEFAULT_CACHE_DIR = "data/cleanup_table_d2/rgb_lmdb"
DEFAULT_IMAGE_KEY = "agentview_image"
DEFAULT_CAMERA_MATRIX_KEY = "oracle/camera_matrix_agentview"
DEFAULT_ACTION_KEY = "action/absolute_action"


def _load_meta(cache_dir: Path) -> dict:
    with (cache_dir / "meta.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def _episode_start(meta: dict, episode: int) -> int:
    lengths = list(map(int, meta["episode_lengths"]))
    if int(episode) < 0 or int(episode) >= len(lengths):
        raise ValueError(f"episode {episode} out of range [0, {len(lengths) - 1}]")
    return int(sum(lengths[: int(episode)]))


def _load_array(cache_dir: Path, key: str) -> np.ndarray:
    arrays_path = cache_dir / "arrays.npz"
    if not arrays_path.exists():
        raise FileNotFoundError(f"Missing arrays archive: {arrays_path}")
    with np.load(arrays_path, allow_pickle=True) as archive:
        if key not in archive:
            raise KeyError(f"Missing key {key!r} in {arrays_path}")
        return np.asarray(archive[key])


def _decode_lmdb_image(cache_dir: Path, image_key: str, global_idx: int) -> np.ndarray:
    try:
        import lmdb
    except ImportError as exc:
        raise ImportError("lmdb is required to read images.lmdb") from exc

    env = lmdb.open(
        str(cache_dir / "images.lmdb"),
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        subdir=False,
    )
    try:
        with env.begin(write=False, buffers=True) as txn:
            key = f"{image_key}/{int(global_idx):08d}".encode("ascii")
            buf = txn.get(key)
            if buf is None:
                raise KeyError(f"Missing LMDB image key: {key!r}")
            arr = np.frombuffer(bytes(buf), dtype=np.uint8)
            image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"cv2.imdecode failed for {key!r}")
            return image
    finally:
        env.close()


def project_points(camera_matrix: np.ndarray, points_xyz_mm: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Project robot-base XYZ points in millimeters with a cache camera matrix."""
    points = np.asarray(points_xyz_mm, dtype=np.float32).reshape(-1, 3)
    matrix = np.asarray(camera_matrix, dtype=np.float32).reshape(4, 4)
    points_h = np.concatenate(
        [points, np.ones((points.shape[0], 1), dtype=np.float32)],
        axis=1,
    )
    proj = (matrix @ points_h.T).T
    z = proj[:, 2]
    valid = np.isfinite(z) & (np.abs(z) > 1e-6)
    uv = np.full((points.shape[0], 2), np.nan, dtype=np.float32)
    uv[valid] = proj[valid, :2] / z[valid, None]
    return uv, valid


def _draw_polyline(image: np.ndarray, uv: np.ndarray, valid: np.ndarray) -> None:
    for i in range(len(uv)):
        if not valid[i]:
            continue
        p = tuple(np.round(uv[i]).astype(int))
        cv2.circle(image, p, 4, (0, 255, 255), -1)
        cv2.putText(
            image,
            str(i),
            (p[0] + 5, p[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 255),
            1,
        )
        if i > 0 and valid[i - 1]:
            q = tuple(np.round(uv[i - 1]).astype(int))
            cv2.line(image, q, p, (0, 255, 255), 2)


def _draw_action_frame(
    image: np.ndarray,
    camera_matrix: np.ndarray,
    action: np.ndarray,
    axis_len_mm: float,
) -> None:
    xyz = np.asarray(action[:3], dtype=np.float32)
    rot = rot6d_to_R(np.asarray(action[3:9], dtype=np.float32)).reshape(3, 3)
    points = np.stack(
        [
            xyz,
            xyz + rot[:, 0] * float(axis_len_mm),
            xyz + rot[:, 1] * float(axis_len_mm),
            xyz + rot[:, 2] * float(axis_len_mm),
        ],
        axis=0,
    )
    uv, valid = project_points(camera_matrix, points)
    if not np.all(valid):
        return
    uv = np.round(uv).astype(int)
    origin = tuple(uv[0])
    for end, color in [
        (uv[1], (0, 0, 255)),
        (uv[2], (0, 255, 0)),
        (uv[3], (255, 0, 0)),
    ]:
        cv2.line(image, origin, tuple(end), color, 2)
    cv2.circle(image, origin, 5, (255, 255, 255), -1)


def overlay_action_chunk(
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    episode: int = 0,
    frame: int = 0,
    horizon: int = 16,
    output_path: str | Path | None = None,
    image_key: str = DEFAULT_IMAGE_KEY,
    camera_matrix_key: str = DEFAULT_CAMERA_MATRIX_KEY,
    action_key: str = DEFAULT_ACTION_KEY,
    axis_len_mm: float = 40.0,
) -> Path:
    """Overlay an absolute action chunk from a realworld LMDB cache onto agentview."""
    cache_dir = Path(cache_dir).expanduser().resolve()
    meta = _load_meta(cache_dir)
    episode_lengths = list(map(int, meta["episode_lengths"]))
    ep_start = _episode_start(meta, int(episode))
    ep_len = int(episode_lengths[int(episode)])
    if int(frame) < 0 or int(frame) >= ep_len:
        raise ValueError(f"frame {frame} out of range [0, {ep_len - 1}]")

    global_idx = ep_start + int(frame)
    end_idx = min(ep_start + ep_len, global_idx + int(horizon))

    actions = _load_array(cache_dir, action_key)
    camera_matrices = _load_array(cache_dir, camera_matrix_key)
    image = _decode_lmdb_image(cache_dir, image_key, global_idx)

    chunk = np.asarray(actions[global_idx:end_idx], dtype=np.float32)
    camera_matrix = np.asarray(camera_matrices[global_idx], dtype=np.float32)
    uv, valid = project_points(camera_matrix, chunk[:, :3])
    _draw_polyline(image, uv, valid)
    if chunk.shape[0] > 0:
        _draw_action_frame(image, camera_matrix, chunk[0], axis_len_mm=float(axis_len_mm))

    cv2.putText(
        image,
        f"episode={episode} frame={frame} global={global_idx} horizon={chunk.shape[0]}",
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )

    if output_path is None:
        output_path = cache_dir / "debug_action_projection" / (
            f"episode_{int(episode):04d}_frame_{int(frame):06d}_h{int(horizon)}.png"
        )
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Project an action chunk from a realworld LMDB cache onto agentview."
    )
    parser.add_argument("--cache-dir", type=str, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--axis-len-mm", type=float, default=40.0)
    args = parser.parse_args()

    out = overlay_action_chunk(
        cache_dir=args.cache_dir,
        episode=int(args.episode),
        frame=int(args.frame),
        horizon=int(args.horizon),
        output_path=args.out,
        axis_len_mm=float(args.axis_len_mm),
    )
    print(f"[wrote] {out}")


if __name__ == "__main__":
    main()
