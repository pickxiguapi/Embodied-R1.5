"""Step 3: estimate depth, normal, and intrinsics from RGB images with MoGE2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch


VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def collect_images(image_dir: str) -> list[Path]:
    root = Path(image_dir)
    paths = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS]
    paths.sort()
    return paths


def to_tensor(image_rgb: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(image_rgb).to(device=device, dtype=torch.float32).permute(2, 0, 1) / 255.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Step3: geometry estimation with MoGE2")
    parser.add_argument("--image-dir", required=True, help="Input RGB image directory")
    parser.add_argument("--output-dir", required=True, help="Output directory for MoGE2 predictions")
    parser.add_argument("--moge-model-path", required=True, help="MoGE2 checkpoint path")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    return parser


def normal_to_vis(normal: np.ndarray) -> np.ndarray:
    return np.clip((normal + 1.0) * 127.5, 0, 255).astype(np.uint8)


def depth_to_vis(depth: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Create a human-friendly depth visualization (does not affect raw depth outputs)."""
    valid = np.isfinite(depth) & (depth > 0)
    if not np.any(valid):
        return np.zeros((*depth.shape, 3), dtype=np.uint8), 0.0, 0.0

    valid_depth = depth[valid].astype(np.float32)
    lo = float(np.percentile(valid_depth, 2.0))
    hi = float(np.percentile(valid_depth, 98.0))
    if hi - lo < 1e-8:
        lo = float(np.min(valid_depth))
        hi = float(np.max(valid_depth) + 1e-8)

    norm = np.clip((depth - lo) / (hi - lo + 1e-8), 0.0, 1.0)
    # Invert so near objects are visually brighter.
    inv = 1.0 - norm
    vis_u8 = np.clip(inv * 255.0, 0, 255).astype(np.uint8)
    vis_bgr = cv2.applyColorMap(vis_u8, cv2.COLORMAP_TURBO)
    vis_bgr[~valid] = 0
    return vis_bgr, lo, hi


def main() -> None:
    args = build_parser().parse_args()
    images = collect_images(args.image_dir)
    if not images:
        raise RuntimeError(f"No RGB image found in: {args.image_dir}")

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    from moge.model.v2 import MoGeModel

    model = MoGeModel.from_pretrained(args.moge_model_path).to(device)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    index: dict[str, dict[str, str]] = {}
    with torch.no_grad():
        for image_path in images:
            image_id = image_path.stem.lower()
            image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image_bgr is None:
                print(f"[step3] skip unreadable image: {image_path.name}")
                continue
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            image_tensor = to_tensor(image_rgb, device)

            output = model.infer(image_tensor)
            depth = output["depth"].detach().cpu().numpy().astype(np.float32)
            normal = output["normal"].detach().cpu().numpy().astype(np.float32)
            intrinsics = output["intrinsics"].detach().cpu().numpy().astype(np.float32)

            image_out = out_root / image_id
            image_out.mkdir(parents=True, exist_ok=True)

            np.save(image_out / "depth.npy", depth)
            np.save(image_out / "normal.npy", normal)
            np.save(image_out / "intrinsics.npy", intrinsics)

            depth_u16 = np.clip(depth * 20000.0, 0, 65535).astype(np.uint16)
            depth_vis, depth_vis_lo, depth_vis_hi = depth_to_vis(depth)
            cv2.imwrite(str(image_out / "depth.png"), depth_u16)
            cv2.imwrite(str(image_out / "depth_vis.png"), depth_vis)
            cv2.imwrite(str(image_out / "normal.png"), normal_to_vis(normal))

            meta = {
                "image_path": str(image_path),
                "depth_shape": list(depth.shape),
                "normal_shape": list(normal.shape),
                "intrinsics_shape": list(intrinsics.shape),
                "depth_vis_percentile_range": {
                    "p2": depth_vis_lo,
                    "p98": depth_vis_hi,
                },
            }
            with open(image_out / "moge_meta.json", "w", encoding="utf-8") as fp:
                json.dump(meta, fp, ensure_ascii=False, indent=2)

            index[image_id] = {
                "image_path": str(image_path),
                "depth_npy": str(image_out / "depth.npy"),
                "normal_npy": str(image_out / "normal.npy"),
                "intrinsics_npy": str(image_out / "intrinsics.npy"),
                "depth_png": str(image_out / "depth.png"),
                "depth_vis_png": str(image_out / "depth_vis.png"),
                "normal_png": str(image_out / "normal.png"),
                "meta_json": str(image_out / "moge_meta.json"),
            }
            print(f"[step3] {image_path.name}")

    with open(out_root / "step3_index.json", "w", encoding="utf-8") as fp:
        json.dump(index, fp, ensure_ascii=False, indent=2)
    print(f"Step3 output: {out_root}")


if __name__ == "__main__":
    main()
