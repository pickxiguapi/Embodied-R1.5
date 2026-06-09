"""Step 6: estimate dominant plane normal from 3D point normals."""

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


def estimate_dominant_normal(
    normals: np.ndarray,
    iterations: int = 256,
    angle_threshold_deg: float = 5.0,
) -> tuple[np.ndarray, int, float]:
    valid = np.all(np.isfinite(normals), axis=1)
    nrm = normals[valid]
    if nrm.shape[0] == 0:
        raise RuntimeError("No valid normals for dominant-normal estimation")

    nrm = nrm / (np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-8)
    upward = nrm[:, 1] < 0  # OpenCV camera coordinates: y-down, so up is negative y
    candidates = np.where(upward)[0]
    if candidates.size == 0:
        candidates = np.arange(nrm.shape[0])

    cos_th = float(np.cos(np.deg2rad(angle_threshold_deg)))
    best_count = -1
    best_seed = None

    for _ in range(iterations):
        seed_idx = int(np.random.choice(candidates))
        seed = nrm[seed_idx]
        dots = nrm @ seed
        inliers = dots > cos_th
        count = int(np.sum(inliers))
        if count > best_count:
            best_count = count
            best_seed = seed

    if best_seed is None:
        raise RuntimeError("Failed to estimate dominant normal")

    dots = nrm @ best_seed
    inliers = dots > cos_th
    cluster = nrm[inliers]
    dominant = unit(np.mean(cluster, axis=0))
    if dominant[1] > 0:
        dominant = -dominant

    ratio = float(best_count) / float(nrm.shape[0])
    return dominant.astype(np.float32), best_count, ratio


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Step6: dominant normal estimation")
    parser.add_argument("--backproject-dir", required=True, help="Step5 output directory")
    parser.add_argument("--output-dir", required=True, help="Step6 output directory")
    parser.add_argument("--iterations", type=int, default=256, help="RANSAC iterations")
    parser.add_argument("--angle-threshold-deg", type=float, default=5.0, help="Inlier angle threshold")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    back_root = Path(args.backproject_dir)
    if not back_root.exists():
        raise RuntimeError(f"Backproject dir not found: {back_root}")

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    index: dict[str, dict[str, object]] = {}

    for image_dir in sorted(p for p in back_root.iterdir() if p.is_dir()):
        image_id = image_dir.name
        ply_path = image_dir / "pointcloud_cam.ply"
        if not ply_path.exists():
            continue

        try:
            ply = read_labeled_ply(ply_path)
            normals = np.stack(
                [
                    ply["nx"].astype(np.float32),
                    ply["ny"].astype(np.float32),
                    ply["nz"].astype(np.float32),
                ],
                axis=1,
            )
        except Exception as exc:
            print(f"[step6] failed to read ply for {image_id}: {exc}")
            continue

        try:
            dominant, inlier_count, inlier_ratio = estimate_dominant_normal(
                normals=normals,
                iterations=args.iterations,
                angle_threshold_deg=args.angle_threshold_deg,
            )
        except Exception as exc:
            print(f"[step6] failed {image_id}: {exc}")
            continue

        image_out = out_root / image_id
        image_out.mkdir(parents=True, exist_ok=True)

        payload = {
            "dominant_normal_cam": [float(x) for x in dominant.tolist()],
            "inlier_count": int(inlier_count),
            "inlier_ratio": float(inlier_ratio),
            "iterations": int(args.iterations),
            "angle_threshold_deg": float(args.angle_threshold_deg),
        }
        with open(image_out / "dominant_normal.json", "w", encoding="utf-8") as fp:
            json.dump(payload, fp, ensure_ascii=False, indent=2)

        index[image_id] = {
            "dominant_normal_json": str(image_out / "dominant_normal.json"),
            "inlier_count": inlier_count,
            "inlier_ratio": inlier_ratio,
        }
        print(f"[step6] {image_id}")

    with open(out_root / "step6_index.json", "w", encoding="utf-8") as fp:
        json.dump(index, fp, ensure_ascii=False, indent=2)
    print(f"Step6 output: {out_root}")


if __name__ == "__main__":
    main()
