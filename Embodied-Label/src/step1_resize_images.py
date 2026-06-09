"""Resize RGB images so width/height are multiples of 14 with minimal ratio drift."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def compute_target_size(width: int, height: int, multiple: int = 14) -> tuple[int, int, float]:
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size: {width}x{height}")

    scale_w = round(width / multiple) * multiple / width
    scale_h = round(height / multiple) * multiple / height
    scale = (scale_w + scale_h) / 2.0

    new_w = max(multiple, int(round(width * scale / multiple) * multiple))
    new_h = max(multiple, int(round(height * scale / multiple) * multiple))
    return new_w, new_h, scale


def collect_images(image_dir: str) -> list[Path]:
    root = Path(image_dir)
    paths = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS]
    paths.sort()
    return paths


def resize_folder(input_dir: str, output_dir: str, multiple: int = 14) -> dict[str, dict[str, float]]:
    in_paths = collect_images(input_dir)
    if not in_paths:
        raise RuntimeError(f"No RGB image found in: {input_dir}")

    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    records: dict[str, dict[str, float]] = {}
    for in_path in in_paths:
        image = cv2.imread(str(in_path))
        if image is None:
            continue
        h, w = image.shape[:2]
        new_w, new_h, scale = compute_target_size(w, h, multiple=multiple)
        interpolation = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
        resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

        out_path = out_root / in_path.name
        cv2.imwrite(str(out_path), resized)
        records[in_path.name] = {
            "old_w": float(w),
            "old_h": float(h),
            "new_w": float(new_w),
            "new_h": float(new_h),
            "scale": float(scale),
        }

    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Resize images to multiples of 14")
    parser.add_argument("--input-dir", required=True, help="Input image directory")
    parser.add_argument("--output-dir", required=True, help="Output image directory")
    parser.add_argument("--multiple", type=int, default=14, help="Target multiple for width/height")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    records = resize_folder(args.input_dir, args.output_dir, multiple=args.multiple)
    summary_path = Path(args.output_dir) / "resize_summary.json"
    with open(summary_path, "w", encoding="utf-8") as fp:
        json.dump(records, fp, ensure_ascii=False, indent=2)
    print(f"Resized {len(records)} images -> {args.output_dir}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()

