"""Step 7: build world frame and transform labeled point cloud to world coordinates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-8:
        return v
    return v / n


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
        raise RuntimeError(
            f"Invalid PLY column count in {ply_path}: {data.shape[1]} vs header {len(prop_names)}"
        )

    out: dict[str, np.ndarray] = {}
    for i, name in enumerate(prop_names):
        out[name] = data[:, i]
    return out


def build_camera_to_world(dominant_normal: np.ndarray, world_origin_cam: np.ndarray) -> np.ndarray:
    z_axis = unit(dominant_normal.astype(np.float64))
    x_ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    x_axis = x_ref - np.dot(x_ref, z_axis) * z_axis
    if np.linalg.norm(x_axis) < 1e-8:
        y_ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        x_axis = y_ref - np.dot(y_ref, z_axis) * z_axis
    x_axis = unit(x_axis)
    y_axis = unit(np.cross(z_axis, x_axis))

    rot = np.vstack([x_axis, y_axis, z_axis])  # camera -> world
    trans = -rot @ world_origin_cam.astype(np.float64)

    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = rot
    mat[:3, 3] = trans
    return mat


def transform_points(points: np.ndarray, mat: np.ndarray) -> np.ndarray:
    n = points.shape[0]
    homo = np.hstack([points.astype(np.float64), np.ones((n, 1), dtype=np.float64)])
    out = (mat @ homo.T).T
    return out[:, :3].astype(np.float32)


def transform_normals(normals: np.ndarray, mat: np.ndarray) -> np.ndarray:
    rot = mat[:3, :3]
    out = (rot @ normals.astype(np.float64).T).T
    return out.astype(np.float32)


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


def write_matrix_txt(out_path: Path, mat: np.ndarray) -> None:
    np.savetxt(out_path, mat, fmt="%.8f")


def estimate_table_origin(
    points_cam: np.ndarray,
    dominant_normal_cam: np.ndarray,
    n_bins: int,
    table_threshold: float,
) -> tuple[np.ndarray, float, int]:
    proj = points_cam @ dominant_normal_cam
    counts, edges = np.histogram(proj, bins=n_bins)
    peak_idx = int(np.argmax(counts))
    table_height = float((edges[peak_idx] + edges[peak_idx + 1]) * 0.5)
    table_mask = np.abs(proj - table_height) < table_threshold
    if not np.any(table_mask):
        raise RuntimeError("No table inliers found; try a larger --table-threshold")
    origin = np.mean(points_cam[table_mask], axis=0).astype(np.float32)
    return origin, table_height, int(np.sum(table_mask))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Step7: world frame estimation and transform")
    parser.add_argument("--backproject-dir", required=True, help="Step5 output directory")
    parser.add_argument("--normal-dir", required=True, help="Step6 output directory")
    parser.add_argument("--output-dir", required=True, help="Step7 output directory")
    parser.add_argument("--n-bins", type=int, default=500, help="Histogram bins for plane peak")
    parser.add_argument("--table-threshold", type=float, default=0.03, help="Table inlier threshold in meters")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    back_root = Path(args.backproject_dir)
    normal_root = Path(args.normal_dir)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    if not back_root.exists():
        raise RuntimeError(f"Backproject dir not found: {back_root}")
    if not normal_root.exists():
        raise RuntimeError(f"Normal dir not found: {normal_root}")

    index: dict[str, dict[str, str]] = {}

    for image_dir in sorted(p for p in back_root.iterdir() if p.is_dir()):
        image_id = image_dir.name
        normal_json = normal_root / image_id / "dominant_normal.json"
        if not normal_json.exists():
            print(f"[step7] skip {image_id}: dominant normal not found")
            continue

        ply_path = image_dir / "pointcloud_cam.ply"
        if not ply_path.exists():
            print(f"[step7] skip {image_id}: missing step5 ply")
            continue

        try:
            ply = read_labeled_ply(ply_path)
            points_cam = np.stack(
                [
                    ply["x"].astype(np.float32),
                    ply["y"].astype(np.float32),
                    ply["z"].astype(np.float32),
                ],
                axis=1,
            )
            normals_cam = np.stack(
                [
                    ply["nx"].astype(np.float32),
                    ply["ny"].astype(np.float32),
                    ply["nz"].astype(np.float32),
                ],
                axis=1,
            )
            colors_rgb = np.stack(
                [
                    ply["red"].astype(np.uint8),
                    ply["green"].astype(np.uint8),
                    ply["blue"].astype(np.uint8),
                ],
                axis=1,
            )
            class_ids = ply["class_id"].astype(np.int32)
            instance_ids = ply["instance_id"].astype(np.int32)
        except Exception as exc:
            print(f"[step7] failed to read ply for {image_id}: {exc}")
            continue

        with open(normal_json, "r", encoding="utf-8") as fp:
            payload = json.load(fp)
        dominant = np.array(payload["dominant_normal_cam"], dtype=np.float32)
        dominant = unit(dominant)

        try:
            world_origin_cam, table_height, table_inliers = estimate_table_origin(
                points_cam=points_cam,
                dominant_normal_cam=dominant,
                n_bins=args.n_bins,
                table_threshold=args.table_threshold,
            )
        except Exception as exc:
            print(f"[step7] failed {image_id}: {exc}")
            continue

        t_cam_to_world = build_camera_to_world(dominant, world_origin_cam)
        points_world = transform_points(points_cam, t_cam_to_world)
        normals_world = transform_normals(normals_cam, t_cam_to_world)

        image_out = out_root / image_id
        image_out.mkdir(parents=True, exist_ok=True)

        write_labeled_ply(
            out_path=image_out / "pointcloud_world.ply",
            points=points_world,
            colors_rgb=colors_rgb,
            normals=normals_world,
            class_ids=class_ids,
            instance_ids=instance_ids,
        )
        write_matrix_txt(image_out / "transform_matrix.txt", t_cam_to_world)

        index[image_id] = {
            "pointcloud_world": str(image_out / "pointcloud_world.ply"),
            "transform_matrix_txt": str(image_out / "transform_matrix.txt"),
        }
        print(f"[step7] {image_id}")

    with open(out_root / "step7_index.json", "w", encoding="utf-8") as fp:
        json.dump(index, fp, ensure_ascii=False, indent=2)
    print(f"Step7 output: {out_root}")


if __name__ == "__main__":
    main()
