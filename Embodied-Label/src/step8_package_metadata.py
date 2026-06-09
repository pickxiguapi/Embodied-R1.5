"""Step 8: package single-image scene outputs into a unified JSON."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np


VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def collect_images(image_dir: str) -> list[Path]:
    root = Path(image_dir)
    paths = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS]
    paths.sort()
    return paths


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def read_labeled_ply(ply_path: Path) -> dict[str, np.ndarray]:
    prop_names: list[str] = []
    n_vertices = None
    header_lines = 0
    with open(ply_path, "r", encoding="utf-8") as fp:
        while True:
            line = fp.readline()
            if line == "":
                raise RuntimeError(f"Invalid PLY (unexpected EOF in header): {ply_path}")
            header_lines += 1
            s = line.strip()
            if s.startswith("element vertex"):
                n_vertices = int(s.split()[-1])
            elif s.startswith("property"):
                prop_names.append(s.split()[-1])
            elif s == "end_header":
                break

    if n_vertices is None:
        raise RuntimeError(f"Invalid PLY (no vertex count): {ply_path}")
    if n_vertices == 0:
        return {name: np.zeros((0,), dtype=np.float32) for name in prop_names}

    data = np.loadtxt(ply_path, skiprows=header_lines, max_rows=n_vertices)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] != len(prop_names):
        raise RuntimeError(f"Invalid PLY column count in {ply_path}: {data.shape[1]} vs {len(prop_names)}")

    out: dict[str, np.ndarray] = {}
    for i, name in enumerate(prop_names):
        out[name] = data[:, i]
    return out


def pca_obb(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centroid = points.mean(axis=0)
    centered = points - centroid
    if points.shape[0] < 3:
        axes = np.eye(3, dtype=np.float64)
        pmin = points.min(axis=0)
        pmax = points.max(axis=0)
        lengths = pmax - pmin
        return centroid, axes, lengths

    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]

    proj = centered @ eigvecs
    min_proj = proj.min(axis=0)
    max_proj = proj.max(axis=0)
    lengths = max_proj - min_proj
    return centroid, eigvecs, lengths


def load_class_id_map(tags_clean_dir: Path, tags_dir: Path) -> tuple[dict[int, str], list[dict[str, Any]]]:
    table_path = tags_clean_dir / "semantic_label_table.json"
    if not table_path.exists():
        table_path = tags_dir / "semantic_label_table.json"
    rows = read_json(table_path, [])

    class_id_to_name: dict[int, str] = {}
    for row in rows:
        try:
            class_id_to_name[int(row["class_id"])] = str(row["label"])
        except Exception:
            continue
    return class_id_to_name, rows


def as_list(x: np.ndarray, ndigits: int = 6) -> list[float]:
    return [round(float(v), ndigits) for v in x.tolist()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Step8: package outputs into a single JSON")
    parser.add_argument("--image-dir", required=True, help="Input image directory (same as step1 output)")
    parser.add_argument("--tags-dir", required=True, help="Step2 output directory")
    parser.add_argument(
        "--tags-clean-dir",
        required=True,
        help="Step2 cleaned-tags directory (usually same as --tags-dir)",
    )
    parser.add_argument("--seg-dir", required=True, help="Step4 output directory")
    parser.add_argument("--moge-dir", required=True, help="Step3 output directory")
    parser.add_argument("--backproject-dir", required=True, help="Step5 output directory")
    parser.add_argument("--normal-dir", required=True, help="Step6 output directory")
    parser.add_argument("--world-dir", required=True, help="Step7 output directory")
    parser.add_argument("--output-dir", required=True, help="Step8 output directory")
    parser.add_argument("--dataset", default="embodied-label", help="Dataset identifier")
    parser.add_argument(
        "--bbox-mode",
        choices=["aabb", "obb"],
        default="aabb",
        help="3D bbox mode: aabb (fast) or obb (PCA-based, slower)",
    )
    parser.add_argument(
        "--split-per-scene",
        action="store_true",
        help="Also dump one JSON per scene under <output-dir>/scenes/",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    image_paths = collect_images(args.image_dir)
    if not image_paths:
        raise RuntimeError(f"No RGB image found in: {args.image_dir}")

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    tags_per_image_raw = read_json(Path(args.tags_dir) / "tags_per_image.json", {})
    tags_per_image_clean = read_json(Path(args.tags_clean_dir) / "tags_per_image.json", {})
    class_id_to_name, semantic_rows = load_class_id_map(Path(args.tags_clean_dir), Path(args.tags_dir))

    scenes: dict[str, Any] = {}
    for image_path in image_paths:
        image_id = image_path.stem.lower()

        seg_dir = Path(args.seg_dir) / image_id
        moge_dir = Path(args.moge_dir) / image_id
        back_dir = Path(args.backproject_dir) / image_id
        normal_dir = Path(args.normal_dir) / image_id
        world_dir = Path(args.world_dir) / image_id

        required = [
            seg_dir / "instances.json",
            seg_dir / "instance_mask.npy",
            moge_dir / "depth.npy",
            moge_dir / "intrinsics.npy",
            back_dir / "pointcloud_cam.ply",
            normal_dir / "dominant_normal.json",
            world_dir / "pointcloud_world.ply",
            world_dir / "transform_matrix.txt",
        ]
        if not all(p.exists() for p in required):
            print(f"[step8] skip {image_id}: missing files from previous steps")
            continue

        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            print(f"[step8] skip {image_id}: unreadable image")
            continue
        img_h, img_w = image_bgr.shape[:2]

        intrinsics = np.load(moge_dir / "intrinsics.npy").astype(np.float32)
        fx = float(intrinsics[0, 0]) * img_w
        fy = float(intrinsics[1, 1]) * img_h
        cx = float(intrinsics[0, 2]) * img_w
        cy = float(intrinsics[1, 2]) * img_h

        instances_2d = read_json(seg_dir / "instances.json", [])
        dominant_payload = read_json(normal_dir / "dominant_normal.json", {})
        cam_to_world = np.loadtxt(world_dir / "transform_matrix.txt", dtype=np.float64)

        ply_world = read_labeled_ply(world_dir / "pointcloud_world.ply")
        world_points = np.stack(
            [ply_world["x"].astype(np.float64), ply_world["y"].astype(np.float64), ply_world["z"].astype(np.float64)],
            axis=1,
        )
        world_class_ids = ply_world["class_id"].astype(np.int32)
        world_instance_ids = ply_world["instance_id"].astype(np.int32)

        room_center = world_points.mean(axis=0) if world_points.size > 0 else np.zeros((3,), dtype=np.float64)
        room_min = world_points.min(axis=0) if world_points.size > 0 else np.zeros((3,), dtype=np.float64)
        room_max = world_points.max(axis=0) if world_points.size > 0 else np.zeros((3,), dtype=np.float64)
        room_extent = room_max - room_min
        room_size = float(room_extent[0] * room_extent[1])  # XY area in world frame

        # Build quick mapping from instance_id to 2D bbox row
        inst2d_map: dict[int, dict[str, Any]] = {}
        for row in instances_2d:
            try:
                inst2d_map[int(row["instance_id"])] = row
            except Exception:
                continue

        object_counts: dict[str, int] = {}
        object_bboxes: dict[str, list[dict[str, Any]]] = {}

        unique_pairs = np.unique(np.stack([world_class_ids, world_instance_ids], axis=1), axis=0)
        for class_id, instance_id in unique_pairs:
            class_id_i = int(class_id)
            instance_id_i = int(instance_id)
            if instance_id_i <= 0:
                continue

            mask = (world_class_ids == class_id_i) & (world_instance_ids == instance_id_i)
            pts = world_points[mask]
            if pts.shape[0] == 0:
                continue

            class_name = class_id_to_name.get(class_id_i, f"class_{class_id_i}")
            object_counts[class_name] = object_counts.get(class_name, 0) + 1

            pmin = pts.min(axis=0)
            pmax = pts.max(axis=0)
            if args.bbox_mode == "obb":
                centroid, axes, lengths = pca_obb(pts)
                axes_row_major = axes.T
            else:
                centroid = pts.mean(axis=0)
                lengths = pmax - pmin
                axes_row_major = np.eye(3, dtype=np.float64)

            bbox_entry = {
                "instance_id": instance_id_i,
                "class_id": class_id_i,
                "bbox_mode": args.bbox_mode,
                "point_count": int(pts.shape[0]),
                "centroid": as_list(centroid),
                "axesLengths": as_list(lengths),
                "normalizedAxes": [as_list(row) for row in axes_row_major],
                "min": as_list(pmin),
                "max": as_list(pmax),
            }
            row2d = inst2d_map.get(instance_id_i)
            if row2d:
                bbox_entry["bbox_2d"] = [float(x) for x in row2d.get("bbox_xyxy", [0, 0, 0, 0])]
                bbox_entry["seg_score"] = float(row2d.get("score", 0.0))
                bbox_entry["mask_area_2d"] = int(row2d.get("mask_area", 0))

            object_bboxes.setdefault(class_name, []).append(bbox_entry)

        frame_bboxes = []
        for row in instances_2d:
            frame_bboxes.append(
                {
                    "instance_id": int(row.get("instance_id", 0)),
                    "class_id": int(row.get("class_id", 0)),
                    "class_name": str(row.get("class_name", "")),
                    "score": float(row.get("score", 0.0)),
                    "bbox_2d": [float(x) for x in row.get("bbox_xyxy", [0, 0, 0, 0])],
                    "mask_area": int(row.get("mask_area", 0)),
                }
            )

        scene_payload = {
            "scene_id": image_id,
            "dataset": args.dataset,
            "camera_intrinsics": {
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy,
            },
            "img_width": int(img_w),
            "img_height": int(img_h),
            "camera_pose_camera_to_world": cam_to_world.tolist(),
            "dominant_normal_cam": dominant_payload.get("dominant_normal_cam", []),
            "room_size": room_size,
            "room_center": as_list(room_center),
            "room_extent_xyz": as_list(room_extent),
            "object_counts": object_counts,
            "object_bboxes": object_bboxes,
            "frames": [
                {
                    "frame_id": 0,
                    "file_path_color": str(image_path),
                    "file_path_depth": str(moge_dir / "depth.png"),
                    "file_path_normal_vis": str(moge_dir / "normal.png"),
                    "bboxes_2d": frame_bboxes,
                }
            ],
            "tags": {
                "raw_tags_step2": tags_per_image_raw.get(image_id, []),
                "clean_tags_step2": tags_per_image_clean.get(image_id, []),
            },
            "artifacts": {
                "seg_instances_json": str(seg_dir / "instances.json"),
                "seg_mask_npy": str(seg_dir / "instance_mask.npy"),
                "moge_dir": str(moge_dir),
                "pointcloud_cam_ply": str(back_dir / "pointcloud_cam.ply"),
                "dominant_normal_json": str(normal_dir / "dominant_normal.json"),
                "pointcloud_world_ply": str(world_dir / "pointcloud_world.ply"),
                "transform_matrix_txt": str(world_dir / "transform_matrix.txt"),
            },
        }
        scenes[image_id] = scene_payload
        print(f"[step8] {image_id}")

    output_payload = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "dataset": args.dataset,
        "bbox_mode": args.bbox_mode,
        "semantic_label_table": semantic_rows,
        "scenes": scenes,
    }

    with open(out_root / "scene_package.json", "w", encoding="utf-8") as fp:
        json.dump(output_payload, fp, ensure_ascii=False, indent=2)
    per_scene_jsons: dict[str, str] = {}
    if args.split_per_scene:
        per_scene_dir = out_root / "scenes"
        per_scene_dir.mkdir(parents=True, exist_ok=True)
        for scene_id, scene_payload in scenes.items():
            single_scene_payload = {
                "schema_version": output_payload["schema_version"],
                "generated_at": output_payload["generated_at"],
                "dataset": output_payload["dataset"],
                "bbox_mode": output_payload["bbox_mode"],
                "semantic_label_table": output_payload["semantic_label_table"],
                "scene": scene_payload,
            }
            out_path = per_scene_dir / f"{scene_id}.json"
            with open(out_path, "w", encoding="utf-8") as sfp:
                json.dump(single_scene_payload, sfp, ensure_ascii=False, indent=2)
            per_scene_jsons[scene_id] = str(out_path)

    with open(out_root / "step8_index.json", "w", encoding="utf-8") as fp:
        json.dump(
            {
                "scene_package_json": str(out_root / "scene_package.json"),
                "num_scenes": len(scenes),
                "split_per_scene": bool(args.split_per_scene),
                "per_scene_jsons": per_scene_jsons,
            },
            fp,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Step8 output: {out_root}")


if __name__ == "__main__":
    main()
