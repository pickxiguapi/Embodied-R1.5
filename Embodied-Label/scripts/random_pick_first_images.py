#!/usr/bin/env python3
"""Sample media from random leaf folders into examples/<name>_<timestamp>."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from datetime import datetime
from pathlib import Path

import cv2

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def list_images_direct(folder: Path) -> list[Path]:
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


def list_videos_direct(folder: Path) -> list[Path]:
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS])


def find_leaf_folders_with_media(source_root: Path, source_type: str) -> list[tuple[Path, list[Path]]]:
    leaves: list[tuple[Path, list[Path]]] = []
    for p in source_root.rglob("*"):
        if not p.is_dir():
            continue
        try:
            children = list(p.iterdir())
        except Exception:
            continue
        has_child_dir = any(c.is_dir() for c in children)
        if has_child_dir:
            continue
        if source_type == "image":
            media = sorted([c for c in children if c.is_file() and c.suffix.lower() in IMAGE_EXTS])
        else:
            media = sorted([c for c in children if c.is_file() and c.suffix.lower() in VIDEO_EXTS])
        if media:
            leaves.append((p, media))
    leaves.sort(key=lambda x: str(x[0]))
    return leaves


def pick_by_1based_rank(items: list[Path], rank_1based: int) -> tuple[Path, int]:
    if not items:
        raise RuntimeError("Cannot pick from empty list")
    rank_1based = max(1, int(rank_1based))
    idx = min(rank_1based - 1, len(items) - 1)
    return items[idx], idx + 1


def extract_video_frame_1based(video_path: Path, out_path: Path, frame_1based: int = 1) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    frame_1based = max(1, int(frame_1based))

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total > 0:
        frame_1based = min(frame_1based, total)

    # try random access first
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_1based - 1)
    ok, frame = cap.read()
    actual_frame = frame_1based

    if not ok or frame is None:
        # fallback to sequential read and keep last available
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        last = None
        idx = 0
        while True:
            ok2, fr = cap.read()
            if not ok2 or fr is None:
                break
            idx += 1
            last = fr
            if idx >= frame_1based:
                break
        if last is None:
            cap.release()
            raise RuntimeError(f"Failed to read frames from video: {video_path}")
        frame = last
        actual_frame = idx

    cap.release()
    if not cv2.imwrite(str(out_path), frame):
        raise RuntimeError(f"Failed to save frame to: {out_path}")
    return actual_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Randomly sample leaf folders and export selected image/video frame"
    )
    parser.add_argument("--source-root", required=True, help="Source root directory")
    parser.add_argument(
        "--target-path",
        required=True,
        help="User target path. Only the last path component will be used for output folder naming.",
    )
    parser.add_argument("--num-samples", type=int, required=True, help="How many samples to keep")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--source-type", choices=["image", "video"], default="image", help="Media source type")
    parser.add_argument(
        "--kth-video",
        type=int,
        default=1,
        help="When source-type=video, pick kth video in each leaf folder (1-based, clamp to last)",
    )
    parser.add_argument(
        "--f-index",
        type=int,
        default=1,
        help=(
            "For source-type=video: pick f-th frame (1-based, clamp to last). "
            "For source-type=image: pick f-th image (1-based, clamp to last)."
        ),
    )
    parser.add_argument("--examples-root", default="./examples", help="Examples root directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_root).resolve()
    if not source_root.exists():
        raise RuntimeError(f"source-root not found: {source_root}")
    if args.num_samples <= 0:
        raise RuntimeError("--num-samples must be > 0")

    examples_root = Path(args.examples_root).resolve()
    examples_root.mkdir(parents=True, exist_ok=True)

    last_layer = Path(args.target_path).name.strip()
    if not last_layer:
        raise RuntimeError("Failed to parse last layer from --target-path")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = examples_root / f"{last_layer}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    leaf_with_media = find_leaf_folders_with_media(source_root, source_type=args.source_type)
    if not leaf_with_media:
        raise RuntimeError(f"No leaf folders with {args.source_type} files found under: {source_root}")

    rng = random.Random(args.seed)
    if args.num_samples >= len(leaf_with_media):
        chosen_leaf = leaf_with_media
    else:
        chosen_leaf = rng.sample(leaf_with_media, args.num_samples)
    chosen_leaf = sorted(chosen_leaf, key=lambda x: str(x[0]))

    mapping = []
    for idx, (leaf_dir, media_files) in enumerate(chosen_leaf, start=1):
        if args.source_type == "video":
            src, picked_rank = pick_by_1based_rank(media_files, args.kth_video)
            dst_suffix = ".jpg"
        else:
            src, picked_rank = pick_by_1based_rank(media_files, args.f_index)
            dst_suffix = src.suffix.lower()

        new_name = f"{idx:04d}{dst_suffix}"
        dst = out_dir / new_name
        if args.source_type == "video":
            actual_frame = extract_video_frame_1based(src, dst, frame_1based=args.f_index)
            mapping.append(
                {
                    "index": idx,
                    "leaf_dir": str(leaf_dir),
                    "src_type": "video",
                    "src_video": str(src),
                    "num_videos_in_leaf": len(media_files),
                    "picked_video_rank_1based": int(picked_rank),
                    "requested_frame_1based": int(max(1, args.f_index)),
                    "actual_frame_1based": int(actual_frame),
                    "new_name": new_name,
                    "new_path": str(dst),
                }
            )
            print(f"[sample] video rank={picked_rank}, frame={actual_frame}: {src} -> {dst}")
        else:
            shutil.copy2(src, dst)
            mapping.append(
                {
                    "index": idx,
                    "leaf_dir": str(leaf_dir),
                    "src_type": "image",
                    "src_image": str(src),
                    "num_images_in_leaf": len(media_files),
                    "picked_image_rank_1based": int(picked_rank),
                    "new_name": new_name,
                    "new_path": str(dst),
                }
            )
            print(f"[sample] {src} -> {dst}")

    payload = {
        "source_root": str(source_root),
        "source_type": args.source_type,
        "seed": args.seed,
        "kth_video": int(max(1, args.kth_video)),
        "f_index": int(max(1, args.f_index)),
        "num_candidate_leaf_folders": len(leaf_with_media),
        "num_selected_leaf_folders": len(chosen_leaf),
        "output_dir": str(out_dir),
        "mapping": mapping,
    }
    with open(out_dir / "mapping.json", "w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)
    print(f"[sample] mapping: {out_dir / 'mapping.json'}")


if __name__ == "__main__":
    main()
