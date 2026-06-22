#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
from PIL import Image


if "rospy" not in sys.modules:
    sys.modules["rospy"] = types.SimpleNamespace(
        loginfo=lambda *args, **kwargs: None,
        logwarn=lambda *args, **kwargs: None,
    )

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "seeker-dev")
    ),
)

from xarm_quest_teleop.eval.live_viz import LiveVizRenderer
from xarm_quest_teleop.policy.seeker_preprocessing import preprocess_real_rgb_image


METHOD_ALIASES = {
    "real_rvt2_policy": "rvt2",
    "rtv2": "rvt2",
    "focus_pool_l1": "focuspool_l1",
    "focus_pool_l2": "focuspool_l2",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def workspace_root() -> Path:
    return repo_root().parent


def default_eval_root() -> Path:
    return repo_root() / "evaluation"


def normalize_method(method: str) -> str:
    method = str(method).strip()
    return METHOD_ALIASES.get(method, method)


def iter_raw_eval_mp4s(path: Path):
    if path.is_file():
        if path.suffix.lower() == ".mp4" and is_raw_eval_video(path):
            yield path
            return
        sibling = raw_sibling_for_overlay_video(path)
        if sibling is not None:
            yield sibling
        return
    for src in sorted(path.rglob("*.mp4")):
        if is_raw_eval_video(src):
            yield src


def is_raw_eval_video(path: Path) -> bool:
    parts = set(path.parts)
    return "videos" in parts and "attention_videos" not in parts and "visualization_videos" not in parts


def raw_sibling_for_overlay_video(path: Path) -> Path | None:
    parts = list(path.parts)
    for overlay_dir in ("attention_videos", "visualization_videos"):
        if overlay_dir not in parts:
            continue
        i = parts.index(overlay_dir)
        candidate = Path(*parts[:i], "videos", *parts[i + 1 :])
        if candidate.exists() and candidate.suffix.lower() == ".mp4":
            return candidate
    return None


def parse_eval_video_path(path: Path) -> tuple[str, str, str]:
    parts = list(path.resolve().parts)
    try:
        i = parts.index("evaluation")
    except ValueError as exc:
        raise ValueError(f"expected path under an evaluation/ tree: {path}") from exc
    if len(parts) <= i + 5:
        raise ValueError(f"expected evaluation/<task>/<method>/<eval>/videos/*.mp4: {path}")
    task = parts[i + 1]
    method = normalize_method(parts[i + 2])
    eval_name = parts[i + 3]
    if parts[i + 4] != "videos":
        raise ValueError(f"expected raw video under videos/: {path}")
    return task, method, eval_name


def load_eval_config(task: str, method: str) -> dict:
    cfg_path = repo_root() / "config" / "eval" / f"{task}_{method}.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"eval config not found for {task}/{method}: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    payload["_path"] = str(cfg_path)
    return payload


def resolve_workspace_path(value: str) -> str:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return str(path)
    return str((workspace_root() / path).resolve())


def make_renderer_cfg(payload: dict, *, task: str, method: str) -> SimpleNamespace:
    return SimpleNamespace(
        task_name=str(payload.get("task", task)),
        model_name=str(payload.get("name", method)),
        model_ckpt_path=resolve_workspace_path(str(payload["ckpt"])),
        calibration_path=str(payload.get("calib", "all_cams_calib.json")),
        rgb_cams=["d435i_front"],
    )


def open_video(path: Path):
    reader = cv2.VideoCapture(str(path))
    if not reader.isOpened():
        raise RuntimeError(f"failed to open video: {path}")
    fps = float(reader.get(cv2.CAP_PROP_FPS) or 5.0)
    return reader, fps


def read_frames(path: Path) -> tuple[list[np.ndarray], float]:
    reader, fps = open_video(path)
    frames = []
    while True:
        ok, frame_bgr = reader.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    reader.release()
    if not frames:
        raise RuntimeError(f"video has no readable frames: {path}")
    return frames, fps


def neutral_lowdim() -> dict[str, np.ndarray]:
    return {
        "ee_pose6": np.asarray([300.0, 0.0, 180.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "gripper_state": np.asarray([850.0], dtype=np.float32),
    }


def temporal_obs_from_front_history(history: list[np.ndarray], obs_horizon: int) -> dict:
    if not history:
        raise ValueError("history must not be empty")
    while len(history) < int(obs_horizon):
        history.insert(0, history[0])
    seq = list(history[-int(obs_horizon) :])
    low = neutral_lowdim()
    return {
        "rgb": {
            "d435i_front": seq,
            "d405": seq,
        },
        "low_dim": {
            "ee_pose6": [low["ee_pose6"].copy() for _ in seq],
            "gripper_state": [low["gripper_state"].copy() for _ in seq],
        },
    }


def fallback_front_frame(frame_rgb: np.ndarray, output_size: int) -> np.ndarray:
    return preprocess_real_rgb_image(
        frame_rgb,
        camera_name="d435i_front",
        target_size=int(output_size),
    )


def redraw_frames(
    src: Path,
    *,
    net: Any,
    renderer: LiveVizRenderer,
    output_size: int,
) -> tuple[list[Image.Image], float]:
    frames, fps = read_frames(src)
    obs_horizon = max(1, int(getattr(net, "n_obs_steps", 2)))
    history: list[np.ndarray] = []
    out: list[Image.Image] = []

    for frame in frames:
        history.append(frame)
        temporal_obs = temporal_obs_from_front_history(history, obs_horizon)
        try:
            net.infer_action(temporal_obs)
            rendered = renderer.make_method_viz_frame(
                temporal_obs=temporal_obs,
                focus_records=list(getattr(net, "last_visual_focus_records", [])),
            )
        except Exception:
            rendered = None
        if rendered is None:
            rendered = fallback_front_frame(frame, output_size)
        if rendered.shape[0] != int(output_size) or rendered.shape[1] != int(output_size):
            rendered = cv2.resize(
                rendered,
                (int(output_size), int(output_size)),
                interpolation=cv2.INTER_AREA,
            )
        out.append(Image.fromarray(rendered.astype(np.uint8, copy=False)))

    return out, fps


def gif_destination(src: Path, scan_root: Path, output_root: Path | None) -> Path:
    if output_root is None:
        output_root = repo_root() / "visualization_gifs"
    try:
        rel = src.relative_to(scan_root if scan_root.is_dir() else scan_root.parent)
    except ValueError:
        rel = Path(src.name)
    rel_parts = list(rel.parts)
    rel_parts = ["visualization_gifs" if part == "videos" else part for part in rel_parts]
    return (output_root / Path(*rel_parts)).with_suffix(".gif")


def save_gif(path: Path, frames: list[Image.Image], fps: float, overwrite: bool) -> bool:
    if path.exists() and not overwrite:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = int(round(1000.0 / max(1.0, float(fps or 5.0))))
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Redraw method-specific eval GIFs from raw rollout MP4s by loading "
            "the corresponding policy checkpoint."
        )
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=str(default_eval_root()),
        help="Raw eval MP4 or directory to scan. Only videos/*.mp4 are processed.",
    )
    parser.add_argument("--device", default="cuda", help="Device for policy inference.")
    parser.add_argument("--output-size", type=int, default=256, help="Output GIF side length.")
    parser.add_argument(
        "--output-root",
        default=None,
        help="Output root. Defaults to xarm_quest_teleop/visualization_gifs.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Rewrite GIFs that already exist.")
    args = parser.parse_args(argv)

    root = Path(args.path).expanduser()
    if not root.is_absolute():
        root = (Path.cwd() / root).resolve()
    if not root.exists():
        raise FileNotFoundError(root)

    output_root = None
    if args.output_root is not None:
        output_root = Path(args.output_root).expanduser()
        if not output_root.is_absolute():
            output_root = (Path.cwd() / output_root).resolve()

    try:
        from xarm_quest_teleop.policy.seeker_policy import SeekerPolicy
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Model-redraw conversion requires the Seeker policy environment. "
            "Activate the env with hydra/torch/seeker installed before running "
            "this script."
        ) from exc

    policy_cache: dict[tuple[str, str], tuple[Any, LiveVizRenderer]] = {}
    converted = 0
    skipped = 0
    failed = 0

    print(
        "[convert_eval_videos_to_gifs] reading raw videos only; old "
        "attention_videos/ and visualization_videos/ MP4s are skipped."
    )
    print(
        "[convert_eval_videos_to_gifs] warning: old MP4s do not contain "
        "synchronized lowdim/eih observations, so conversion replays the model "
        "with neutral lowdim and duplicated front image for eih."
    )

    for src in iter_raw_eval_mp4s(root):
        try:
            task, method, _ = parse_eval_video_path(src)
            key = (task, method)
            if key not in policy_cache:
                payload = load_eval_config(task, method)
                cfg = make_renderer_cfg(payload, task=task, method=method)
                net = SeekerPolicy(
                    ckpt_path=cfg.model_ckpt_path,
                    seed=int(payload.get("seed", 0)),
                    device=str(args.device),
                )
                renderer = LiveVizRenderer(cfg=cfg)
                policy_cache[key] = (net, renderer)
            net, renderer = policy_cache[key]
            frames, fps = redraw_frames(
                src,
                net=net,
                renderer=renderer,
                output_size=int(args.output_size),
            )
            dst = gif_destination(src, root, output_root)
            if save_gif(dst, frames, fps, overwrite=bool(args.overwrite)):
                converted += 1
                print(f"{src} -> {dst}")
            else:
                skipped += 1
        except Exception as exc:
            failed += 1
            print(f"FAILED {src}: {exc}", file=sys.stderr)

    print(f"converted={converted} skipped_existing={skipped} failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
