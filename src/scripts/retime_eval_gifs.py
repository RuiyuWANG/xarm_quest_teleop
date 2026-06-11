#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

from PIL import Image, ImageSequence


DEFAULT_STRIDE = 3
DEFAULT_FPS = 20.0


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rewrite existing evaluation GIFs with a fixed frame stride and FPS."
    )
    parser.add_argument(
        "eval_root",
        nargs="?",
        default=str(repo_root() / "evaluation"),
        help="Evaluation root or a single GIF. Defaults to CloudGripper_Manipulation/evaluation.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=DEFAULT_STRIDE,
        help=f"Keep every Nth frame. Default: {DEFAULT_STRIDE}.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=DEFAULT_FPS,
        help=f"Output GIF playback FPS. Default: {DEFAULT_FPS:g}.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be rewritten without changing files.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help=(
            "Write retimed copies under this directory instead of rewriting in place. "
            "The eval-root-relative path is preserved."
        ),
    )
    return parser.parse_args()


def iter_gifs(path: Path) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix.lower() == ".gif" else []
    return sorted(p for p in path.rglob("*.gif") if p.is_file())


def load_strided_frames(path: Path, stride: int) -> list[Image.Image]:
    frames: list[Image.Image] = []
    with Image.open(path) as img:
        for i, frame in enumerate(ImageSequence.Iterator(img)):
            if i % stride == 0:
                frames.append(frame.convert("RGBA"))
    if not frames:
        raise RuntimeError(f"no readable frames in {path}")
    return frames


def output_path_for(src: Path, root: Path, output_root: Path | None) -> Path:
    if output_root is None:
        return src
    base = root if root.is_dir() else root.parent
    try:
        rel = src.relative_to(base)
    except ValueError:
        rel = Path(src.name)
    return output_root / rel


def rewrite_gif(
    src: Path,
    dst: Path,
    *,
    stride: int,
    fps: float,
    dry_run: bool,
) -> tuple[int, int]:
    stride = max(1, int(stride))
    duration_ms = max(1, int(round(1000.0 / max(float(fps), 1e-6))))
    frames = load_strided_frames(src, stride)

    if dry_run:
        return len(frames), duration_ms

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dst.with_name(f".{dst.stem}.retime_tmp.gif")
    try:
        frames[0].save(
            tmp_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=0,
            optimize=False,
            disposal=2,
        )
        os.replace(tmp_path, dst)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return len(frames), duration_ms


def main() -> int:
    args = parse_args()
    root = Path(args.eval_root).expanduser().resolve()
    output_root = None
    if args.output_root is not None:
        output_root = Path(args.output_root).expanduser().resolve()
    gifs = iter_gifs(root)
    if not gifs:
        print(f"No GIFs found under {root}")
        return 1

    converted = 0
    failed = 0
    for path in gifs:
        dst = output_path_for(path, root, output_root)
        try:
            n_frames, duration_ms = rewrite_gif(
                path,
                dst,
                stride=args.stride,
                fps=args.fps,
                dry_run=args.dry_run,
            )
        except Exception as exc:
            failed += 1
            print(f"FAILED {path}: {exc}")
            continue
        converted += 1
        action = "would rewrite" if args.dry_run else "rewrote"
        if dst == path:
            print(f"{action} {path} ({n_frames} frames, {duration_ms} ms/frame)")
        else:
            print(
                f"{action} {path} -> {dst} "
                f"({n_frames} frames, {duration_ms} ms/frame)"
            )

    print(
        f"Done: converted={converted} failed={failed} "
        f"stride={max(1, int(args.stride))} fps={float(args.fps):g}"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
