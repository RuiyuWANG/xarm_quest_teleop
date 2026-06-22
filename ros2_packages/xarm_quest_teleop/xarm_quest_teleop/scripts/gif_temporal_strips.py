#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageSequence


DEFAULT_NUM_FRAMES = 8
DEFAULT_GAP = 16
DEFAULT_BG = (255, 255, 255)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert GIFs into temporal strip PNGs by sampling evenly spaced "
            "frames and concatenating them in one row."
        )
    )
    parser.add_argument(
        "input",
        help="Input GIF or folder containing GIFs.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help=(
            "Output directory. For folders, preserves relative paths. "
            "Default: <input>_temporal_strips for folders, input parent for one GIF."
        ),
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=DEFAULT_NUM_FRAMES,
        help=f"Number of evenly sampled frames per strip. Default: {DEFAULT_NUM_FRAMES}.",
    )
    parser.add_argument(
        "--gap",
        type=int,
        default=DEFAULT_GAP,
        help=f"White pixels between consecutive frames. Default: {DEFAULT_GAP}.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing PNG strips.",
    )
    return parser.parse_args()


def iter_gifs(path: Path) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix.lower() == ".gif" else []
    return sorted(p for p in path.rglob("*.gif") if p.is_file())


def evenly_spaced_indices(n_total: int, n_sample: int) -> list[int]:
    n_sample = max(1, int(n_sample))
    if n_total <= 1:
        return [0] * n_sample
    if n_sample == 1:
        return [0]
    return [
        int(round(i * (n_total - 1) / float(n_sample - 1)))
        for i in range(n_sample)
    ]


def load_sampled_frames(path: Path, n_frames: int) -> list[Image.Image]:
    with Image.open(path) as gif:
        frames = [frame.convert("RGBA") for frame in ImageSequence.Iterator(gif)]
    if not frames:
        raise RuntimeError(f"no readable frames in {path}")
    return [frames[i] for i in evenly_spaced_indices(len(frames), n_frames)]


def composite_on_white(frame: Image.Image) -> Image.Image:
    out = Image.new("RGB", frame.size, DEFAULT_BG)
    if frame.mode == "RGBA":
        out.paste(frame, mask=frame.getchannel("A"))
    else:
        out.paste(frame.convert("RGB"))
    return out


def make_strip(frames: list[Image.Image], gap: int) -> Image.Image:
    rgb_frames = [composite_on_white(frame) for frame in frames]
    widths = [frame.width for frame in rgb_frames]
    heights = [frame.height for frame in rgb_frames]
    gap = max(0, int(gap))
    out_w = sum(widths) + gap * max(0, len(rgb_frames) - 1)
    out_h = max(heights)
    strip = Image.new("RGB", (out_w, out_h), DEFAULT_BG)

    x = 0
    for frame in rgb_frames:
        y = (out_h - frame.height) // 2
        strip.paste(frame, (x, y))
        x += frame.width + gap
    return strip


def output_path_for(src: Path, root: Path, output_root: Path | None) -> Path:
    if output_root is None:
        if root.is_file():
            return src.with_name(f"{src.stem}_strip.png")
        output_root = root.with_name(f"{root.name}_temporal_strips")
    base = root if root.is_dir() else root.parent
    try:
        rel = src.relative_to(base)
    except ValueError:
        rel = Path(src.name)
    return (output_root / rel).with_suffix(".png")


def convert_one(
    src: Path,
    dst: Path,
    *,
    n_frames: int,
    gap: int,
    overwrite: bool,
) -> bool:
    if dst.exists() and not overwrite:
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    frames = load_sampled_frames(src, n_frames)
    strip = make_strip(frames, gap)
    strip.save(dst)
    return True


def main() -> int:
    args = parse_args()
    root = Path(args.input).expanduser().resolve()
    output_root = (
        Path(args.output_root).expanduser().resolve()
        if args.output_root is not None
        else None
    )
    gifs = iter_gifs(root)
    if not gifs:
        print(f"No GIFs found: {root}")
        return 1

    converted = 0
    skipped = 0
    failed = 0
    for src in gifs:
        dst = output_path_for(src, root, output_root)
        try:
            did_convert = convert_one(
                src,
                dst,
                n_frames=args.frames,
                gap=args.gap,
                overwrite=args.overwrite,
            )
        except Exception as exc:
            failed += 1
            print(f"FAILED {src}: {exc}")
            continue
        if did_convert:
            converted += 1
            print(f"wrote {dst}")
        else:
            skipped += 1
            print(f"skipped existing {dst}")

    print(
        f"Done: converted={converted} skipped={skipped} failed={failed} "
        f"frames={max(1, int(args.frames))} gap={max(0, int(args.gap))}"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
