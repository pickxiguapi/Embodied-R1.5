"""Run Grounded-SAM2 instance segmentation from a semantic label table."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pycocotools.mask as mask_util
import torch


VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def collect_images(image_dir: str) -> list[Path]:
    root = Path(image_dir)
    paths = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS]
    paths.sort()
    return paths


def load_class_id_map(semantic_table_path: str) -> dict[str, int]:
    with open(semantic_table_path, "r", encoding="utf-8") as fp:
        rows = json.load(fp)
    class_id_map: dict[str, int] = {}
    for row in rows:
        class_id_map[str(row["label"]).strip().lower()] = int(row["class_id"])
    return class_id_map


def load_tags_per_image(tags_path: str) -> dict[str, list[str]]:
    with open(tags_path, "r", encoding="utf-8") as fp:
        payload = json.load(fp)
    out: dict[str, list[str]] = {}
    for image_id, labels in payload.items():
        cleaned: list[str] = []
        for label in labels:
            token = str(label).strip().lower()
            if token:
                cleaned.append(token)
        out[str(image_id).strip().lower()] = sorted(set(cleaned))
    return out


def _single_mask_to_rle(mask: np.ndarray) -> dict[str, Any]:
    rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def _decode_rle(rle: dict[str, Any]) -> np.ndarray:
    mask = mask_util.decode(rle)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return mask.astype(bool)


def encode_instance_mask(
    results: dict[str, Any],
    class_id_map: dict[str, int],
    out_mask_path: Path,
    out_instances_path: Path,
    out_viz_path: Path,
    annotations_key: str = "annotations",
) -> None:
    h = int(results["img_height"])
    w = int(results["img_width"])
    encoded_mask = np.zeros((h, w), dtype=np.int32)

    rows: list[dict[str, Any]] = []
    instance_id = 1
    for ann in results.get(annotations_key, []):
        class_name = str(ann.get("class_name", "")).strip().lower()
        class_id = class_id_map.get(class_name, 0)
        if class_id <= 0:
            continue

        mask = _decode_rle(ann["segmentation"])
        if mask.shape != encoded_mask.shape:
            mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)

        packed = (class_id << 16) | instance_id
        encoded_mask[mask] = packed
        rows.append(
            {
                "instance_id": instance_id,
                "class_id": class_id,
                "class_name": class_name,
                "score": float(ann.get("score", 0.0)),
                "bbox_xyxy": [float(x) for x in ann.get("bbox", [0, 0, 0, 0])],
                "mask_area": int(np.sum(mask)),
            }
        )
        instance_id += 1

    out_mask_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_mask_path, encoded_mask)
    with open(out_instances_path, "w", encoding="utf-8") as fp:
        json.dump(rows, fp, ensure_ascii=False, indent=2)

    viz = np.zeros((h, w, 3), dtype=np.uint8)
    rng = np.random.default_rng(seed=42)
    for value in np.unique(encoded_mask):
        if value == 0:
            continue
        color = rng.integers(64, 255, size=(3,), dtype=np.uint8)
        viz[encoded_mask == value] = color
    cv2.imwrite(str(out_viz_path), cv2.cvtColor(viz, cv2.COLOR_RGB2BGR))


@dataclass
class SegmentConfig:
    gsam2_root: str
    sam_config_dir: str
    sam_config_name: str
    sam_checkpoint: str
    gdino_config: str
    gdino_checkpoint: str
    box_threshold: float = 0.35
    text_threshold: float = 0.25
    multimask_output: bool = False
    device: str = "auto"


class GroundedSam2Runner:
    def __init__(self, cfg: SegmentConfig):
        self.cfg = cfg
        self._active_device = None
        self._ready = False

    def _ensure_imports(self) -> None:
        if hasattr(self, "_build_sam2"):
            return
        gsam2_root = str(Path(self.cfg.gsam2_root).resolve())
        if gsam2_root not in sys.path:
            sys.path.append(gsam2_root)
        os.environ["PYTHONPATH"] = gsam2_root + ":" + os.environ.get("PYTHONPATH", "")

        from hydra import initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        from torchvision.ops import box_convert

        from grounding_dino.groundingdino.util.inference import load_image, load_model, predict
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        self._initialize_config_dir = initialize_config_dir
        self._GlobalHydra = GlobalHydra
        self._box_convert = box_convert
        self._load_image = load_image
        self._load_model = load_model
        self._predict = predict
        self._build_sam2 = build_sam2
        self._sam2_predictor_cls = SAM2ImagePredictor

    def _resolve_preferred_device(self) -> str:
        if self.cfg.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.cfg.device

    def _init_models(self, device: str) -> None:
        self._ensure_imports()
        self._GlobalHydra.instance().clear()
        with self._initialize_config_dir(
            config_dir=str(Path(self.cfg.sam_config_dir).resolve()),
            version_base="1.2",
        ):
            sam_model = self._build_sam2(
                config_file=self.cfg.sam_config_name,
                ckpt_path=self.cfg.sam_checkpoint,
                device=device,
            )
        self._sam_predictor = self._sam2_predictor_cls(sam_model)
        self._grounding_model = self._load_model(
            model_config_path=self.cfg.gdino_config,
            model_checkpoint_path=self.cfg.gdino_checkpoint,
            device=device,
        )
        self._active_device = device
        self._ready = True

    def _lazy_init(self) -> None:
        if self._ready:
            return
        preferred = self._resolve_preferred_device()
        if preferred == "cuda":
            try:
                print("[segment] trying cuda...")
                self._init_models("cuda")
            except Exception as exc:
                print(f"[segment] cuda init failed: {exc}; fallback to cpu")
                self._init_models("cpu")
        else:
            self._init_models(preferred)

    def _predict_masks_for_boxes(self, image: np.ndarray, input_boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self._sam_predictor.set_image(image)
        masks, scores, _ = self._sam_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=self.cfg.multimask_output,
        )
        if self.cfg.multimask_output:
            best = np.argmax(scores, axis=1)
            masks = masks[np.arange(masks.shape[0]), best]
            scores = scores[np.arange(scores.shape[0]), best]
        if masks.ndim == 4:
            masks = masks.squeeze(1)
        if masks.ndim == 2:
            masks = masks[None, ...]
        return masks.astype(bool), np.asarray(scores, dtype=np.float32).reshape(-1)

    def _predict_masks_on_crops(self, image_source: np.ndarray, input_boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h, w = image_source.shape[:2]
        masks_crop: list[np.ndarray] = []
        scores_crop: list[float] = []

        for box in input_boxes:
            x1 = int(np.floor(float(box[0])))
            y1 = int(np.floor(float(box[1])))
            x2 = int(np.ceil(float(box[2])))
            y2 = int(np.ceil(float(box[3])))

            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(1, min(x2, w))
            y2 = max(1, min(y2, h))

            if x2 <= x1 or y2 <= y1:
                masks_crop.append(np.zeros((h, w), dtype=bool))
                scores_crop.append(0.0)
                continue

            crop = image_source[y1:y2, x1:x2]
            if crop.size == 0:
                masks_crop.append(np.zeros((h, w), dtype=bool))
                scores_crop.append(0.0)
                continue

            crop_box = np.array([[0.0, 0.0, float(crop.shape[1] - 1), float(crop.shape[0] - 1)]], dtype=np.float32)
            crop_masks, crop_scores = self._predict_masks_for_boxes(crop, crop_box)

            full_mask = np.zeros((h, w), dtype=bool)
            full_mask[y1:y2, x1:x2] = crop_masks[0]
            masks_crop.append(full_mask)
            scores_crop.append(float(crop_scores[0]) if crop_scores.size > 0 else 0.0)

        return np.stack(masks_crop, axis=0), np.asarray(scores_crop, dtype=np.float32)

    @staticmethod
    def _build_annotations(
        class_names: list[str],
        input_boxes: np.ndarray,
        masks: np.ndarray,
        scores: np.ndarray,
    ) -> list[dict[str, Any]]:
        annotations: list[dict[str, Any]] = []
        for class_name, box, mask, score in zip(class_names, input_boxes, masks, scores):
            annotations.append(
                {
                    "class_name": class_name,
                    "bbox": box.tolist(),
                    "segmentation": _single_mask_to_rle(mask),
                    "score": float(score),
                }
            )
        return annotations

    def run(
        self,
        image_path: str,
        text_prompt: str,
        output_dir: Path,
        normal_image_path: str | None = None,
        mask_source: str = "rgb",
    ) -> dict[str, Any]:
        self._lazy_init()
        output_dir.mkdir(parents=True, exist_ok=True)

        image_source, image_gdino = self._load_image(image_path)
        try:
            boxes, confidences, labels = self._predict(
                model=self._grounding_model,
                image=image_gdino,
                caption=text_prompt,
                box_threshold=self.cfg.box_threshold,
                text_threshold=self.cfg.text_threshold,
                device=self._active_device,
            )
        except Exception as exc:
            if self._active_device == "cuda":
                print(f"[segment] cuda predict failed: {exc}; fallback to cpu")
                self._init_models("cpu")
                image_source, image_gdino = self._load_image(image_path)
                self._sam_predictor.set_image(image_source)
                boxes, confidences, labels = self._predict(
                    model=self._grounding_model,
                    image=image_gdino,
                    caption=text_prompt,
                    box_threshold=self.cfg.box_threshold,
                    text_threshold=self.cfg.text_threshold,
                    device=self._active_device,
                )
            else:
                raise

        h, w, _ = image_source.shape
        if boxes.numel() == 0:
            results = {
                "image_path": image_path,
                "text_prompt": text_prompt,
                "mask_source": mask_source,
                "normal_image_path": normal_image_path,
                "annotations": [],
                "img_width": w,
                "img_height": h,
            }
            with open(output_dir / "grounded_sam2_results.json", "w", encoding="utf-8") as fp:
                json.dump(results, fp, ensure_ascii=False, indent=2)
            return results

        boxes = boxes * torch.tensor([w, h, w, h], device=boxes.device)
        input_boxes = self._box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").cpu().numpy()
        class_names = [str(x).strip().lower() for x in labels]

        if mask_source == "rgb":
            masks, scores = self._predict_masks_for_boxes(image_source, input_boxes)
        elif mask_source == "normal":
            if not normal_image_path:
                raise RuntimeError("mask-source=normal requires normal_image_path")
            normal_bgr = cv2.imread(normal_image_path, cv2.IMREAD_COLOR)
            if normal_bgr is None:
                raise RuntimeError(f"failed to read normal image: {normal_image_path}")
            normal_rgb = cv2.cvtColor(normal_bgr, cv2.COLOR_BGR2RGB)
            if normal_rgb.shape[:2] != (h, w):
                normal_rgb = cv2.resize(normal_rgb, (w, h), interpolation=cv2.INTER_LINEAR)
            masks, scores = self._predict_masks_for_boxes(normal_rgb, input_boxes)
        elif mask_source == "crop":
            masks, scores = self._predict_masks_on_crops(image_source, input_boxes)
        else:
            raise ValueError(f"unknown mask_source: {mask_source}")

        annotations = self._build_annotations(class_names, input_boxes, masks, scores)

        results = {
            "image_path": image_path,
            "text_prompt": text_prompt,
            "mask_source": mask_source,
            "normal_image_path": normal_image_path,
            "annotations": annotations,
            "img_width": w,
            "img_height": h,
        }
        with open(output_dir / "grounded_sam2_results.json", "w", encoding="utf-8") as fp:
            json.dump(results, fp, ensure_ascii=False, indent=2)
        return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Instance segmentation from semantic labels")
    parser.add_argument("--image-dir", required=True, help="Input image directory")
    parser.add_argument("--semantic-table", required=True, help="semantic_label_table.json path")
    parser.add_argument("--prompt-path", required=True, help="semantic_prompt.txt path")
    parser.add_argument(
        "--tags-per-image",
        default=None,
        help="Optional per-image tags json (from step2 tags_per_image.json); if set, use per-image prompt",
    )
    parser.add_argument(
        "--moge-dir",
        default=None,
        help="Step3 output directory for normal.png (required when --mask-source normal)",
    )
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--gsam2-root", required=True, help="Grounded-SAM-2 repo root")
    parser.add_argument("--sam-config-dir", required=True, help="SAM config dir")
    parser.add_argument("--sam-config-name", required=True, help="SAM config name")
    parser.add_argument("--sam-checkpoint", required=True, help="SAM checkpoint path")
    parser.add_argument("--gdino-config", required=True, help="GroundingDINO config path")
    parser.add_argument("--gdino-checkpoint", required=True, help="GroundingDINO checkpoint path")
    parser.add_argument("--box-threshold", type=float, default=0.35)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--device", default="auto", help="auto/cuda/cpu")
    parser.add_argument(
        "--mask-source",
        choices=["rgb", "normal", "crop"],
        default="rgb",
        help="Choose the segmentation mask source: rgb, normal, or crop.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    images = collect_images(args.image_dir)
    if not images:
        raise RuntimeError(f"No RGB image found in: {args.image_dir}")
    if args.mask_source == "normal" and not args.moge_dir:
        raise RuntimeError("--moge-dir is required when --mask-source normal is set")

    class_id_map = load_class_id_map(args.semantic_table)
    prompt = Path(args.prompt_path).read_text(encoding="utf-8").strip()
    tags_per_image: dict[str, list[str]] = {}
    if args.tags_per_image:
        tags_path = Path(args.tags_per_image)
        if not tags_path.exists():
            raise RuntimeError(f"tags_per_image json not found: {tags_path}")
        tags_per_image = load_tags_per_image(str(tags_path))

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    runner = GroundedSam2Runner(
        SegmentConfig(
            gsam2_root=args.gsam2_root,
            sam_config_dir=args.sam_config_dir,
            sam_config_name=args.sam_config_name,
            sam_checkpoint=args.sam_checkpoint,
            gdino_config=args.gdino_config,
            gdino_checkpoint=args.gdino_checkpoint,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            device=args.device,
        )
    )

    index: dict[str, dict[str, Any]] = {}
    for image_path in images:
        image_id = image_path.stem.lower()
        image_out = out_root / image_id
        per_image_labels = tags_per_image.get(image_id, [])
        per_image_prompt = ". ".join(per_image_labels).strip()
        effective_prompt = per_image_prompt if per_image_prompt else prompt
        normal_image_path = None
        if args.mask_source == "normal" and args.moge_dir:
            cand = Path(args.moge_dir) / image_id / "normal.png"
            if cand.exists():
                normal_image_path = str(cand)
            else:
                print(f"[segment] skip {image_path.name}: normal.png not found under {args.moge_dir}")
                continue

        results = runner.run(
            str(image_path),
            effective_prompt,
            image_out,
            normal_image_path=normal_image_path,
            mask_source=args.mask_source,
        )
        mask_path = image_out / "instance_mask.npy"
        instances_path = image_out / "instances.json"
        viz_path = image_out / "instance_mask_viz.png"
        encode_instance_mask(results, class_id_map, mask_path, instances_path, viz_path)

        index[image_id] = {
            "image_path": str(image_path),
            "prompt_mode": "per_image" if per_image_prompt else "global",
            "prompt_used": effective_prompt,
            "mask_source": args.mask_source,
            "normal_image_path": normal_image_path,
            "results_json": str(image_out / "grounded_sam2_results.json"),
            "instance_mask": str(mask_path),
            "instances": str(instances_path),
            "mask_viz": str(viz_path),
        }
        print(f"[segment] {image_path.name} (mask_source={args.mask_source})")

    with open(out_root / "segmentation_index.json", "w", encoding="utf-8") as fp:
        json.dump(index, fp, ensure_ascii=False, indent=2)
    print(f"Segmentation output: {out_root}")


if __name__ == "__main__":
    main()
