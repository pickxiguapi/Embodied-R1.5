"""Step 5: back-project depth/normal/mask to camera-space labeled 3D points."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def collect_images(image_dir: str) -> list[Path]:
    root = Path(image_dir)
    paths = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS]
    paths.sort()
    return paths


def decode_mask(encoded_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    class_ids = (encoded_mask >> 16).astype(np.int32)
    instance_ids = (encoded_mask & 0xFFFF).astype(np.int32)
    return class_ids, instance_ids


def backproject(depth: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    h, w = depth.shape
    fx = float(intrinsics[0, 0]) * w
    fy = float(intrinsics[1, 1]) * h
    cx = float(intrinsics[0, 2]) * w
    cy = float(intrinsics[1, 2]) * h

    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    z = depth
    return np.stack([x, y, z], axis=-1).astype(np.float32)


def write_labeled_ply(
    out_path: Path,
    points: np.ndarray,
    colors_rgb: np.ndarray,
    normals: np.ndarray,
    class_ids: np.ndarray,
    instance_ids: np.ndarray,
) -> None:
    n = points.shape[0]
    with open(out_path, "w", encoding="utf-8") as fp:
        fp.write("ply\n")
        fp.write("format ascii 1.0\n")
        fp.write(f"element vertex {n}\n")
        fp.write("property float x\n")
        fp.write("property float y\n")
        fp.write("property float z\n")
        fp.write("property uchar red\n")
        fp.write("property uchar green\n")
        fp.write("property uchar blue\n")
        fp.write("property float nx\n")
        fp.write("property float ny\n")
        fp.write("property float nz\n")
        fp.write("property int class_id\n")
        fp.write("property int instance_id\n")
        fp.write("end_header\n")
        for i in range(n):
            x, y, z = points[i]
            r, g, b = colors_rgb[i]
            nx, ny, nz = normals[i]
            cls = int(class_ids[i])
            ins = int(instance_ids[i])
            fp.write(f"{x} {y} {z} {int(r)} {int(g)} {int(b)} {nx} {ny} {nz} {cls} {ins}\n")


def normalize_rows(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=1, keepdims=True) + 1e-8
    return v / n


def _sliding_window_2d(array: np.ndarray, window_h: int, window_w: int) -> np.ndarray:
    from numpy.lib.stride_tricks import as_strided

    h, w = array.shape
    out_h = h - window_h + 1
    out_w = w - window_w + 1
    s0, s1 = array.strides
    return as_strided(
        array,
        shape=(out_h, out_w, window_h, window_w),
        strides=(s0, s1, s0, s1),
    )


def compute_edge_mask(
    depth: np.ndarray,
    valid_mask: np.ndarray,
    thickness: int,
    tol: float,
) -> np.ndarray:
    """Depth-discontinuity filtering used in original PlaneAligner."""
    if thickness <= 0:
        return np.zeros_like(valid_mask, dtype=bool)

    disp = np.where(valid_mask, 1.0 / (depth + 1e-8), 0.0).astype(np.float32)
    disp_pad = np.pad(disp, thickness, constant_values=0.0)
    mask_pad = np.pad(valid_mask, thickness, constant_values=False)

    kernel_size = 2 * thickness + 1
    disp_win = _sliding_window_2d(disp_pad, kernel_size, kernel_size)
    mask_win = _sliding_window_2d(mask_pad, kernel_size, kernel_size)

    weights = mask_win.astype(np.float32)
    sum_w = np.sum(weights, axis=(-2, -1))
    sum_disp = np.sum(disp_win * weights, axis=(-2, -1))
    disp_mean = np.where(sum_w > 0, sum_disp / np.maximum(sum_w, 1e-8), 0.0)

    fg_edge = valid_mask & (disp > (1.0 + tol) * disp_mean)
    bg_edge = valid_mask & (disp_mean > (1.0 + tol) * disp)

    kernel = np.ones((3, 3), dtype=np.uint8)
    fg_d = cv2.dilate(fg_edge.astype(np.uint8), kernel, iterations=thickness) > 0
    bg_d = cv2.dilate(bg_edge.astype(np.uint8), kernel, iterations=thickness) > 0
    return fg_d & bg_d


def voxel_downsample_to_target(
    points: np.ndarray,
    colors_rgb: np.ndarray,
    normals: np.ndarray,
    class_ids: np.ndarray,
    instance_ids: np.ndarray,
    target_points: int,
    max_iter: int = 12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    n_points = int(points.shape[0])
    if target_points <= 0 or n_points <= target_points:
        return points, colors_rgb, normals, class_ids, instance_ids, {
            "enabled": False,
            "target_points": target_points,
            "before_points": n_points,
            "after_points": n_points,
            "voxel_size": 0.0,
        }

    extent = np.ptp(points, axis=0)
    scene_size = float(max(np.max(extent), 1e-6))
    voxel_size = scene_size / float(target_points ** (1.0 / 3.0))
    voxel_size = max(voxel_size, 1e-6)

    best = None
    for _ in range(max_iter):
        grid = np.floor(points / voxel_size).astype(np.int64)
        uniq, inverse, counts = np.unique(grid, axis=0, return_inverse=True, return_counts=True)
        n_vox = int(uniq.shape[0])

        points_sum = np.zeros((n_vox, 3), dtype=np.float64)
        normals_sum = np.zeros((n_vox, 3), dtype=np.float64)
        colors_sum = np.zeros((n_vox, 3), dtype=np.float64)
        np.add.at(points_sum, inverse, points)
        np.add.at(normals_sum, inverse, normals)
        np.add.at(colors_sum, inverse, colors_rgb.astype(np.float64))

        points_ds = (points_sum / counts[:, None]).astype(np.float32)
        normals_ds = normalize_rows((normals_sum / counts[:, None]).astype(np.float32))
        colors_ds = np.clip(colors_sum / counts[:, None], 0, 255).astype(np.uint8)

        order = np.argsort(inverse, kind="stable")
        inv_sorted = inverse[order]
        first_pos = np.r_[0, np.flatnonzero(np.diff(inv_sorted)) + 1]
        first_indices = order[first_pos]
        class_ds = class_ids[first_indices]
        inst_ds = instance_ids[first_indices]

        best = (points_ds, colors_ds, normals_ds, class_ds, inst_ds, voxel_size, n_vox)

        if abs(n_vox - target_points) <= max(1, int(target_points * 0.1)):
            break
        if n_vox > target_points:
            voxel_size *= 1.2
        else:
            voxel_size /= 1.2

    assert best is not None
    points_ds, colors_ds, normals_ds, class_ds, inst_ds, voxel_size, n_vox = best
    return points_ds, colors_ds, normals_ds, class_ds, inst_ds, {
        "enabled": True,
        "target_points": int(target_points),
        "before_points": n_points,
        "after_points": int(n_vox),
        "voxel_size": float(voxel_size),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Step5: back-project to labeled 3D points")
    parser.add_argument("--image-dir", required=True, help="RGB image directory")
    parser.add_argument("--moge-dir", required=True, help="Step3 output directory")
    parser.add_argument("--seg-dir", required=True, help="Step4 segmentation directory")
    parser.add_argument("--output-dir", required=True, help="Step5 output directory")
    parser.add_argument(
        "--downsample-points",
        type=int,
        default=0,
        help="Target number of points after voxel downsampling; 0 disables downsampling",
    )
    parser.add_argument(
        "--edge-filter-thickness",
        type=int,
        default=1,
        help="Occlusion edge filter thickness; 0 disables edge filtering",
    )
    parser.add_argument(
        "--edge-filter-tol",
        type=float,
        default=0.04,
        help="Occlusion edge filter tolerance",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    images = collect_images(args.image_dir)
    if not images:
        raise RuntimeError(f"No RGB image found in: {args.image_dir}")

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    index: dict[str, dict[str, str]] = {}

    for image_path in images:
        image_id = image_path.stem.lower()
        step3_dir = Path(args.moge_dir) / image_id
        seg_dir = Path(args.seg_dir) / image_id
        depth_path = step3_dir / "depth.npy"
        normal_path = step3_dir / "normal.npy"
        intrinsics_path = step3_dir / "intrinsics.npy"
        mask_path = seg_dir / "instance_mask.npy"

        required = [depth_path, normal_path, intrinsics_path, mask_path]
        if not all(p.exists() for p in required):
            print(f"[step5] skip {image_path.name}: missing step3 or step4 files")
            continue

        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            print(f"[step5] skip unreadable image: {image_path.name}")
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        depth = np.load(depth_path).astype(np.float32)
        normal = np.load(normal_path).astype(np.float32)
        intrinsics = np.load(intrinsics_path).astype(np.float32)
        encoded_mask = np.load(mask_path).astype(np.int32)

        h, w = depth.shape
        if image_rgb.shape[:2] != (h, w):
            image_rgb = cv2.resize(image_rgb, (w, h), interpolation=cv2.INTER_LINEAR)
        if normal.shape[:2] != (h, w):
            normal = cv2.resize(normal, (w, h), interpolation=cv2.INTER_LINEAR)
        if encoded_mask.shape != (h, w):
            encoded_mask = cv2.resize(encoded_mask, (w, h), interpolation=cv2.INTER_NEAREST).astype(np.int32)

        points_map = backproject(depth, intrinsics)
        valid_depth = np.isfinite(depth) & (depth > 0)
        if args.edge_filter_thickness > 0:
            edge_mask = compute_edge_mask(
                depth=depth,
                valid_mask=valid_depth,
                thickness=args.edge_filter_thickness,
                tol=args.edge_filter_tol,
            )
            valid = valid_depth & (~edge_mask)
        else:
            edge_mask = np.zeros_like(valid_depth, dtype=bool)
            valid = valid_depth

        n_before = int(np.sum(valid_depth))
        n_removed = int(np.sum(valid_depth & edge_mask))
        n_after = int(np.sum(valid))

        points = points_map[valid]
        normals = normal[valid]
        colors_rgb = image_rgb[valid].astype(np.uint8)
        encoded_flat = encoded_mask[valid]
        class_ids, instance_ids = decode_mask(encoded_flat)

        (
            points_out,
            colors_out,
            normals_out,
            class_out,
            instance_out,
            downsample_meta,
        ) = voxel_downsample_to_target(
            points=points,
            colors_rgb=colors_rgb,
            normals=normals,
            class_ids=class_ids,
            instance_ids=instance_ids,
            target_points=args.downsample_points,
        )

        image_out = out_root / image_id
        image_out.mkdir(parents=True, exist_ok=True)

        write_labeled_ply(
            out_path=image_out / "pointcloud_cam.ply",
            points=points_out,
            colors_rgb=colors_out,
            normals=normals_out,
            class_ids=class_out,
            instance_ids=instance_out,
        )

        meta = {
            "image_path": str(image_path),
            "num_points": int(points_out.shape[0]),
            "num_classes": int(np.unique(class_out).shape[0]),
            "num_instances": int(np.unique(instance_out).shape[0]),
            "edge_filter": {
                "enabled": bool(args.edge_filter_thickness > 0),
                "thickness": int(args.edge_filter_thickness),
                "tol": float(args.edge_filter_tol),
                "before_points": n_before,
                "removed_points": n_removed,
                "after_points": n_after,
            },
            "downsample": downsample_meta,
        }
        with open(image_out / "backproject_meta.json", "w", encoding="utf-8") as fp:
            json.dump(meta, fp, ensure_ascii=False, indent=2)

        index[image_id] = {
            "image_path": str(image_path),
            "pointcloud_cam": str(image_out / "pointcloud_cam.ply"),
            "meta_json": str(image_out / "backproject_meta.json"),
        }
        print(f"[step5] {image_path.name}")

    with open(out_root / "step5_index.json", "w", encoding="utf-8") as fp:
        json.dump(index, fp, ensure_ascii=False, indent=2)
    print(f"Step5 output: {out_root}")


if __name__ == "__main__":
    main()
