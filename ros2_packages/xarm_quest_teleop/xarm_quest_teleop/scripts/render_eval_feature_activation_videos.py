#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import types
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from PIL import Image, ImageSequence


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

from xarm_quest_teleop.policy.seeker_preprocessing import convert_real_low_dim, preprocess_real_rgb_image
from xarm_quest_teleop.utils.tree_utils import dict_apply


METHOD_ALIASES = {
    "real_rvt2_policy": "rvt2",
    "rtv2": "rvt2",
    "focus_pool_l1": "focuspool_l1",
    "focus_pool_l2": "focuspool_l2",
}
DEFAULT_FRAME_STRIDE = 3
DEFAULT_OUTPUT_FPS = 2.0


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def workspace_root() -> Path:
    return repo_root().parent


def default_eval_root() -> Path:
    return repo_root() / "evaluation"


def normalize_method(method: str) -> str:
    method = str(method).strip()
    return METHOD_ALIASES.get(method, method)


def is_raw_eval_video(path: Path) -> bool:
    parts = set(path.parts)
    return (
        path.suffix.lower() in {".mp4", ".gif"}
        and "videos" in parts
        and "attention_videos" not in parts
        and "visualization_videos" not in parts
        and "feature_activation_videos" not in parts
    )


def iter_raw_eval_videos(path: Path):
    if path.is_file():
        if is_raw_eval_video(path):
            yield path
        return
    for src in sorted(path.rglob("*")):
        if is_raw_eval_video(src):
            yield src


def parse_eval_video_path(path: Path) -> tuple[str, str, str]:
    parts = list(path.resolve().parts)
    try:
        i = parts.index("evaluation")
    except ValueError as exc:
        raise ValueError(f"expected path under an evaluation/ tree: {path}") from exc
    if len(parts) <= i + 5:
        raise ValueError(f"expected evaluation/<task>/<method>/<eval>/videos/*: {path}")
    task = parts[i + 1]
    method = normalize_method(parts[i + 2])
    eval_name = parts[i + 3]
    if parts[i + 4] != "videos":
        raise ValueError(f"expected raw video under videos/: {path}")
    return task, method, eval_name


def matching_state_trace(video_path: Path) -> Optional[Path]:
    trace = video_path.parent.parent / "state_traces" / f"{video_path.stem}.jsonl"
    return trace if trace.exists() else None


def load_state_trace(path: Path) -> tuple[dict, list[dict]]:
    header = {}
    events = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if item.get("event") == "start":
                header = item
            elif item.get("event") == "inference_lowdim":
                events.append(item)
    if not events:
        raise ValueError(f"no inference_lowdim events in {path}")
    return header, events


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


def read_video(path: Path) -> tuple[list[np.ndarray], float]:
    if path.suffix.lower() == ".gif":
        image = Image.open(path)
        frames = [np.asarray(frame.convert("RGB")) for frame in ImageSequence.Iterator(image)]
        duration_ms = float(image.info.get("duration", 200) or 200)
        fps = 1000.0 / max(1.0, duration_ms)
        if not frames:
            raise RuntimeError(f"GIF has no readable frames: {path}")
        return frames, fps

    reader = cv2.VideoCapture(str(path))
    if not reader.isOpened():
        raise RuntimeError(f"failed to open video: {path}")
    fps = float(reader.get(cv2.CAP_PROP_FPS) or 5.0)
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


def write_gif(path: Path, frames: list[np.ndarray], fps: float) -> None:
    if not frames:
        raise ValueError("frames must not be empty")
    path.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = int(round(1000.0 / max(1.0, float(fps or 5.0))))
    pil_frames = [
        Image.fromarray(frame.astype(np.uint8, copy=False))
        for frame in frames
    ]
    pil_frames[0].save(
        path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
    )


def event_for_frame(header: dict, events: list[dict], frame_idx: int, fps: float) -> dict:
    if len(events) == 1:
        return events[0]
    start = float(header.get("created_wall_time", events[0].get("wall_time", 0.0)))
    t = start + float(frame_idx) / max(1.0, float(fps or 5.0))
    return min(events, key=lambda item: abs(float(item.get("wall_time", t)) - t))


def ensure_policy_image(frame_rgb: np.ndarray, policy_size: int) -> np.ndarray:
    frame = np.asarray(frame_rgb)
    if frame.shape[0] == int(policy_size) and frame.shape[1] == int(policy_size):
        return frame.astype(np.uint8, copy=False)
    return preprocess_real_rgb_image(
        frame,
        camera_name="d435i_front",
        target_size=int(policy_size),
    )


def build_policy_obs(net: Any, frame_history: list[np.ndarray], event: dict) -> dict:
    import torch

    policy_size = int(net._policy_image_size())
    rot_dim = int(net._policy_rot_dim())
    obs_horizon = max(1, int(getattr(net, "n_obs_steps", 2)))
    while len(frame_history) < obs_horizon:
        frame_history.insert(0, frame_history[0])
    frames = list(frame_history[-obs_horizon:])
    images = np.stack([ensure_policy_image(frame, policy_size) for frame in frames], axis=0)

    low_dim = event["low_dim"]
    ee = np.asarray(low_dim["ee_pose6"], dtype=np.float32)
    grip = np.asarray(low_dim["gripper_state"], dtype=np.float32)
    if ee.ndim == 1:
        ee = ee[None, :]
    if grip.ndim == 1:
        grip = grip[:, None] if grip.shape[0] == ee.shape[0] else grip[None, :]
    if ee.shape[0] < obs_horizon:
        pad = np.repeat(ee[:1], obs_horizon - ee.shape[0], axis=0)
        ee = np.concatenate([pad, ee], axis=0)
    if grip.shape[0] < obs_horizon:
        pad = np.repeat(grip[:1], obs_horizon - grip.shape[0], axis=0)
        grip = np.concatenate([pad, grip], axis=0)

    low_proc = convert_real_low_dim(
        {
            "ee_pose6": ee[-obs_horizon:],
            "gripper_state": grip[-obs_horizon:],
        },
        rot_dim=rot_dim,
    )

    obs = {
        "agentview_image": np.moveaxis(images, -1, 1)[None].astype(np.uint8),
        "robot0_eye_in_hand_image": np.moveaxis(images, -1, 1)[None].astype(np.uint8),
        "robot0_eef_pos": low_proc["robot0_eef_pos"][None].astype(np.float32),
        "robot0_eef_rot": low_proc["robot0_eef_rot"][None].astype(np.float32),
        "robot0_gripper_qpos": low_proc["robot0_gripper_qpos"][None].astype(np.float32),
    }
    task_embedding = np.asarray(net.task_embedding, dtype=np.float32)
    obs["task_embedding"] = np.broadcast_to(
        task_embedding[:, None, :],
        (1, obs_horizon, task_embedding.shape[-1]),
    ).copy()
    obs["robot_id"] = np.full((1, obs_horizon, 1), int(net.robot_id), dtype=np.float32)
    if net.task_language_tokens is not None:
        tokens = np.asarray(net.task_language_tokens, dtype=np.float32)
        if tokens.ndim == 2:
            tokens = tokens[None, :, :]
        obs["task_language_tokens"] = np.broadcast_to(
            tokens[:, None, :, :],
            (1, obs_horizon, tokens.shape[-2], tokens.shape[-1]),
        ).copy()

    obs_t = dict_apply(obs, lambda x: torch.from_numpy(x).to(device=net.device))
    return obs_t


def front_record(records: list[Any]):
    for record in records:
        if str(getattr(record, "view", "")) == "agentview":
            return record
    return None


def to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().to("cpu").numpy()
    return np.asarray(value)


def normalized_map(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    value = np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
    value = np.maximum(value, 0.0)
    lo = float(value.min()) if value.size else 0.0
    hi = float(value.max()) if value.size else 0.0
    if hi - lo <= 1e-8:
        return np.zeros_like(value, dtype=np.float32)
    return ((value - lo) / (hi - lo)).astype(np.float32)


def tensor_image_to_rgb(image, obs_horizon: int) -> np.ndarray:
    import torch

    image = image.detach()
    if image.shape[0] >= obs_horizon and image.shape[0] % obs_horizon == 0:
        image = image.reshape(image.shape[0] // obs_horizon, obs_horizon, *image.shape[1:])[:, -1]
    else:
        image = image[-1:]
    image = image[0].float()
    mean = torch.tensor([0.485, 0.456, 0.406], device=image.device, dtype=image.dtype).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=image.device, dtype=image.dtype).view(3, 1, 1)
    image = (image * std + mean).clamp(0.0, 1.0)
    image = image.detach().to("cpu").numpy()
    image = np.moveaxis(image, 0, -1)
    return np.clip(image * 255.0, 0, 255).astype(np.uint8)


def activation_map_for_frame(
    net: Any,
    obs_t: dict,
    method: str,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], list[Any]]:
    import torch

    obs_encoder = getattr(net.policy, "obs_encoder", None)
    view_encoder = getattr(obs_encoder, "agentview_encoder", None)
    captured = {}
    restore = None

    if view_encoder is not None and hasattr(view_encoder, "_forward_backbone"):
        original_forward_backbone = view_encoder._forward_backbone

        def wrapped_forward_backbone(image):
            feat = original_forward_backbone(image)
            captured["feat"] = feat.detach()
            captured["image"] = image.detach()
            return feat

        view_encoder._forward_backbone = wrapped_forward_backbone
        restore = lambda: setattr(view_encoder, "_forward_backbone", original_forward_backbone)
    else:
        agentview_cnn = getattr(obs_encoder, "agentview_cnn", None)
        trunk = getattr(agentview_cnn, "trunk", None)
        if trunk is not None and hasattr(trunk, "forward"):
            original_forward = trunk.forward

            def wrapped_trunk(image):
                feat = original_forward(image)
                captured["feat"] = feat.detach()
                captured["image"] = image.detach()
                return feat

            trunk.forward = wrapped_trunk
            restore = lambda: setattr(trunk, "forward", original_forward)

    if restore is None:
        with torch.no_grad():
            net.policy.obs_encoder(obs_t)
        records = list(getattr(obs_encoder, "last_visual_focus_records", []))
        net.last_visual_focus_records = records
        return None, None, records

    try:
        with torch.no_grad():
            net.policy.obs_encoder(obs_t)
        records = list(getattr(obs_encoder, "last_visual_focus_records", []))
        net.last_visual_focus_records = records
    finally:
        restore()

    feat = captured.get("feat")
    if feat is None:
        return None, None, records
    feat = feat.detach()
    obs_horizon = max(1, int(getattr(net, "n_obs_steps", 2)))
    if feat.shape[0] >= obs_horizon and feat.shape[0] % obs_horizon == 0:
        feat = feat.reshape(feat.shape[0] // obs_horizon, obs_horizon, *feat.shape[1:])[:, -1]
    else:
        feat = feat[-1:]
    activation = torch.linalg.vector_norm(feat[0], dim=0).detach().to("cpu").numpy()
    activation = normalized_map(activation)

    if method.startswith("focuspool"):
        record = front_record(list(getattr(net, "last_visual_focus_records", [])))
        mask = None
        if record is not None:
            metadata = getattr(record, "metadata", {}) or {}
            mask = metadata.get("grid_heatmap", None)
            if mask is None:
                pred = getattr(record, "prediction", None)
                mask = None if pred is None else getattr(pred, "heatmap", None)
        if mask is not None:
            mask = to_numpy(mask).reshape(-1, *to_numpy(mask).shape[-2:])[-1]
            if mask.shape != activation.shape:
                mask = cv2.resize(
                    normalized_map(mask),
                    (activation.shape[1], activation.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
            activation = normalized_map(activation * normalized_map(mask))
    base_frame = None
    if method.startswith("rvt2") and captured.get("image") is not None:
        base_frame = tensor_image_to_rgb(captured["image"], obs_horizon)
    return activation, base_frame, records


def draw_grid_attention_mask(image: np.ndarray, record: Any) -> bool:
    metadata = getattr(record, "metadata", {}) or {}
    pred = getattr(record, "prediction", None)
    heatmap = metadata.get("grid_heatmap")
    if heatmap is None and pred is not None:
        heatmap = getattr(pred, "heatmap", None)
    if heatmap is None and pred is not None:
        heatmap = getattr(pred, "mask_grid", None)
    if heatmap is None:
        return False

    grid = to_numpy(heatmap)
    if grid.size == 0:
        return False
    grid = grid.reshape(-1, *grid.shape[-2:])[-1].astype(np.float32, copy=False)
    grid = normalized_map(grid)
    if float(grid.max()) <= 1e-8:
        return False

    h, w = image.shape[:2]
    mask = cv2.resize(grid, (w, h), interpolation=cv2.INTER_NEAREST)
    color = np.asarray([20.0, 120.0, 255.0], dtype=np.float32)
    alpha = (0.12 + 0.58 * mask[..., None]).astype(np.float32)
    image[:] = np.clip(
        image.astype(np.float32) * (1.0 - alpha) + color[None, None, :] * alpha,
        0,
        255,
    ).astype(np.uint8)

    gh, gw = grid.shape[-2:]
    grid_overlay = image.copy()
    for i in range(1, gh):
        y = int(round(h * float(i) / float(max(1, gh))))
        cv2.line(grid_overlay, (0, y), (w, y), (255, 255, 255), 1)
    for j in range(1, gw):
        x = int(round(w * float(j) / float(max(1, gw))))
        cv2.line(grid_overlay, (x, 0), (x, h), (255, 255, 255), 1)
    image[:] = cv2.addWeighted(grid_overlay, 0.25, image, 0.75, 0.0)
    return True


def draw_spatial_keypoints(image: np.ndarray, record: Any) -> bool:
    metadata = getattr(record, "metadata", {}) or {}
    points = metadata.get("points_px")
    if points is None:
        return False
    points = to_numpy(points).reshape(-1, 2).astype(np.float32, copy=False)
    if points.size == 0:
        return False

    mean_idx = metadata.get("mean_point_index")
    if mean_idx is not None:
        mean_idx = int(mean_idx)
        keep = np.ones((points.shape[0],), dtype=bool)
        if 0 <= mean_idx < points.shape[0]:
            keep[mean_idx] = False
        points = points[keep]

    src_h, src_w = getattr(record, "image_size", (image.shape[0], image.shape[1]))
    sx = float(image.shape[1]) / float(src_w)
    sy = float(image.shape[0]) / float(src_h)
    for p in points:
        x = int(round(float(p[0]) * sx))
        y = int(round(float(p[1]) * sy))
        cv2.circle(image, (x, y), 5, (0, 0, 0), -1)
        cv2.circle(image, (x, y), 3, (0, 255, 255), -1)
    return True


def draw_record_box(image: np.ndarray, record: Any) -> bool:
    pred = getattr(record, "prediction", None)
    box = None if pred is None else getattr(pred, "box_px", None)
    if box is None:
        return False
    box_np = to_numpy(box).reshape(-1, 4)[-1].astype(np.float32, copy=False)
    src_h, src_w = getattr(record, "image_size", (image.shape[0], image.shape[1]))
    sx = float(image.shape[1]) / float(src_w)
    sy = float(image.shape[0]) / float(src_h)
    x1, y1, x2, y2 = box_np
    p1 = (int(round(x1 * sx)), int(round(y1 * sy)))
    p2 = (int(round(x2 * sx)), int(round(y2 * sy)))
    cv2.rectangle(image, p1, p2, (255, 220, 0), 3)
    return True


def method_viz_frame(frame_rgb: np.ndarray, records: list[Any], method: str) -> np.ndarray:
    image = preprocess_real_rgb_image(
        np.asarray(frame_rgb, dtype=np.uint8),
        camera_name="d435i_front",
    ).copy()
    record = front_record(records)
    if record is None:
        return image

    method = str(method).lower()
    drew = False
    if method.startswith("focuspool"):
        drew = draw_grid_attention_mask(image, record)
    elif method.startswith("spatial_softmax"):
        drew = draw_spatial_keypoints(image, record)
    elif method.startswith("rvt2"):
        drew = draw_record_box(image, record)
    else:
        drew = draw_grid_attention_mask(image, record)
        if not drew:
            drew = draw_spatial_keypoints(image, record)
        if not drew:
            drew = draw_record_box(image, record)
    _ = drew
    return image


def overlay_activation(
    frame_rgb: np.ndarray,
    heatmap: Optional[np.ndarray],
    *,
    preprocess_frame: bool = True,
) -> np.ndarray:
    if preprocess_frame:
        frame = preprocess_real_rgb_image(
            np.asarray(frame_rgb, dtype=np.uint8),
            camera_name="d435i_front",
        ).copy()
    else:
        frame = np.asarray(frame_rgb, dtype=np.uint8).copy()
    if heatmap is None:
        return frame
    heatmap = normalized_map(heatmap)

    h, w = frame.shape[:2]
    mask = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
    color = cv2.applyColorMap(
        np.clip(mask * 255.0, 0, 255).astype(np.uint8),
        cv2.COLORMAP_TURBO,
    )
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB).astype(np.float32)
    alpha = (0.08 + 0.58 * mask[..., None]).astype(np.float32)
    return np.clip(
        frame.astype(np.float32) * (1.0 - alpha) + color * alpha,
        0,
        255,
    ).astype(np.uint8)


def output_path_for(video_path: Path, output_root: Optional[Path]) -> Path:
    if output_root is None:
        return video_path.parent.parent / "feature_activation_videos" / f"{video_path.stem}.gif"
    try:
        rel = video_path.relative_to(default_eval_root())
    except ValueError:
        rel = Path(video_path.name)
    rel_parts = ["feature_activation_videos" if p == "videos" else p for p in rel.parts]
    return (output_root / Path(*rel_parts)).with_suffix(".gif")


def visualization_output_path_for(video_path: Path, output_root: Optional[Path]) -> Path:
    if output_root is None:
        return video_path.parent.parent / "visualization_videos" / f"{video_path.stem}.gif"
    try:
        rel = video_path.relative_to(default_eval_root())
    except ValueError:
        rel = Path(video_path.name)
    rel_parts = ["visualization_videos" if p == "videos" else p for p in rel.parts]
    return (output_root / Path(*rel_parts)).with_suffix(".gif")


def render_one(
    video_path: Path,
    *,
    trace_path: Path,
    net: Any,
    method: str,
    output_path: Path,
    visualization_path: Path,
    overwrite: bool,
    frame_stride: int,
    output_fps: float,
) -> dict[str, bool]:
    need_feature = overwrite or not output_path.exists()
    need_visualization = overwrite or not visualization_path.exists()
    if not need_feature and not need_visualization:
        return {"feature": False, "visualization": False}

    header, events = load_state_trace(trace_path)
    frames, fps = read_video(video_path)
    obs_horizon = max(1, int(getattr(net, "n_obs_steps", 2)))
    history: list[np.ndarray] = []
    feature_frames = []
    visualization_frames = []
    frame_stride = max(1, int(frame_stride))

    for idx, frame in enumerate(frames):
        history.append(frame)
        if len(history) > obs_horizon:
            history = history[-obs_horizon:]
        if idx % frame_stride != 0:
            continue
        event = event_for_frame(header, events, idx, fps)
        obs_t = build_policy_obs(net, history, event)
        heatmap, base_frame, records = activation_map_for_frame(net, obs_t, method)
        if need_feature:
            if base_frame is None:
                feature_frames.append(overlay_activation(frame, heatmap))
            else:
                feature_frames.append(
                    overlay_activation(base_frame, heatmap, preprocess_frame=False)
                )
        if need_visualization:
            visualization_frames.append(method_viz_frame(frame, records, method))

    if need_feature:
        write_gif(output_path, feature_frames, output_fps)
    if need_visualization:
        write_gif(visualization_path, visualization_frames, output_fps)
    return {"feature": need_feature, "visualization": need_visualization}


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Render cropped feature-activation GIF overlays from saved eval videos and "
            "matching lowdim state_traces/*.jsonl."
        )
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=str(default_eval_root()),
        help="Eval video file or directory to scan.",
    )
    parser.add_argument("--device", default="cuda", help="Device for policy inference.")
    parser.add_argument("--output-root", default=None, help="Optional output root.")
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=DEFAULT_FRAME_STRIDE,
        help="Save one rendered frame every N source frames.",
    )
    parser.add_argument(
        "--output-fps",
        type=float,
        default=DEFAULT_OUTPUT_FPS,
        help="Playback FPS for saved GIFs.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Rewrite existing outputs.")
    args = parser.parse_args(argv)

    try:
        from xarm_quest_teleop.policy.seeker_policy import SeekerPolicy
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Feature activation rendering requires the Seeker policy environment "
            "with hydra/torch/seeker installed."
        ) from exc

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

    policy_cache: dict[tuple[str, str], Any] = {}
    feature_converted = 0
    visualization_converted = 0
    skipped = 0
    missing_trace = 0
    failed = 0

    for video_path in iter_raw_eval_videos(root):
        trace_path = matching_state_trace(video_path)
        if trace_path is None:
            missing_trace += 1
            continue

        try:
            task, method, _ = parse_eval_video_path(video_path)
            key = (task, method)
            if key not in policy_cache:
                cfg = load_eval_config(task, method)
                net = SeekerPolicy(
                    ckpt_path=resolve_workspace_path(str(cfg["ckpt"])),
                    seed=int(cfg.get("seed", 0)),
                    device=str(args.device),
                )
                policy_cache[key] = net
            out_path = output_path_for(video_path, output_root)
            viz_path = visualization_output_path_for(video_path, output_root)
            did_write = render_one(
                video_path,
                trace_path=trace_path,
                net=policy_cache[key],
                method=method,
                output_path=out_path,
                visualization_path=viz_path,
                overwrite=bool(args.overwrite),
                frame_stride=int(args.frame_stride),
                output_fps=float(args.output_fps),
            )
            if did_write["feature"]:
                feature_converted += 1
                print(f"{video_path} -> {out_path}")
            if did_write["visualization"]:
                visualization_converted += 1
                print(f"{video_path} -> {viz_path}")
            if not did_write["feature"] and not did_write["visualization"]:
                skipped += 1
        except Exception as exc:
            failed += 1
            print(f"FAILED {video_path}: {exc}", file=sys.stderr)

    print(
        f"feature_converted={feature_converted} "
        f"visualization_converted={visualization_converted} "
        f"skipped_existing={skipped} "
        f"missing_trace={missing_trace} failed={failed}"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
