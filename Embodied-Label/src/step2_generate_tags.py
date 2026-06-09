"""Build semantic labels with Qwen3-VL or RAM++, with optional integrated cleaning."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from PIL import Image


VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
SPLIT_PATTERN = re.compile(r"[,\n;/|，、]+")
NON_ALNUM = re.compile(r"[^a-z0-9 ]+")
MULTI_SPACE = re.compile(r"\s+")


def collect_images(image_dir: str) -> list[Path]:
    root = Path(image_dir)
    paths = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS]
    paths.sort()
    return paths


def normalize_label(text: str) -> str:
    token = str(text).strip().lower()
    token = NON_ALNUM.sub(" ", token)
    token = MULTI_SPACE.sub(" ", token).strip()
    if token.endswith("s") and len(token) > 3 and not token.endswith("ss"):
        token = token[:-1]
    return token


def parse_labels(text: str) -> list[str]:
    raw = [part.strip() for part in SPLIT_PATTERN.split(text)]
    labels = [normalize_label(x) for x in raw if x.strip()]
    labels = [x for x in labels if x]
    return sorted(set(labels))


def is_cuda_oom(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg and "cuda" in msg


def save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)


class QwenTagger:
    def __init__(self, model_dir: str, max_new_tokens: int = 200):
        try:
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        except ImportError as exc:
            raise ImportError(
                "Qwen tagging requires a transformers version that provides "
                "Qwen3VLForConditionalGeneration. Use a dedicated tags environment."
            ) from exc

        self.max_new_tokens = max_new_tokens
        self.model_dir = model_dir
        self._qwen_cls = Qwen3VLForConditionalGeneration
        self.processor = AutoProcessor.from_pretrained(model_dir)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.model = self._load_model(self.device)
        except Exception as exc:
            if self.device == "cuda":
                print(f"[tags][qwen] cuda load failed: {exc}; fallback to cpu")
                self.device = "cpu"
                self.model = self._load_model("cpu")
            else:
                raise

    def _load_model(self, device: str):
        if device == "cuda":
            return self._qwen_cls.from_pretrained(
                self.model_dir,
                torch_dtype=torch.bfloat16,
                device_map="cuda",
                trust_remote_code=True,
            )
        return self._qwen_cls.from_pretrained(
            self.model_dir,
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True,
        )

    def _fallback_to_cpu(self, reason: Exception) -> None:
        if self.device == "cpu":
            return
        print(f"[tags][qwen] fallback to cpu due to: {reason}")
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        self.device = "cpu"
        self.model = self._load_model("cpu")

    def predict_labels(self, image_path: str) -> list[str]:
        image = Image.open(image_path).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "List all visible objects in this image. Use simple noun phrases (e.g., 'robot arm', 'stove', 'banana'), separated by commas. No adjectives, no colors, no descriptions.",
                    },
                    {"type": "image", "image": image},
                ],
            }
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        try:
            from qwen_vl_utils import process_vision_info

            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
        except ImportError:
            inputs = self.processor(
                text=[text],
                images=[image],
                padding=True,
                return_tensors="pt",
            )

        inputs = inputs.to(self.device)
        try:
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                repetition_penalty=1.2,
            )
        except Exception as exc:
            if self.device == "cuda" and is_cuda_oom(exc):
                self._fallback_to_cpu(exc)
                inputs = inputs.to(self.device)
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    repetition_penalty=1.2,
                )
            else:
                raise
        generated_ids_trimmed = generated_ids[0][len(inputs.input_ids[0]) :]
        output_text = self.processor.decode(generated_ids_trimmed, skip_special_tokens=True)
        return parse_labels(output_text)


class RamTagger:
    def __init__(self, model_path: str, image_size: int = 384, vit: str = "swin_l"):
        try:
            from ram import get_transform
            from ram import inference_ram as inference
            from ram.models import ram_plus
        except ImportError as exc:
            raise ImportError(
                "RAM tagging requires the RAM package (e.g. ram-plus / recognize-anything)."
            ) from exc

        self.model_path = model_path
        self.image_size = image_size
        self.vit = vit
        self._ram_plus = ram_plus
        self._get_transform = get_transform
        self.transform = get_transform(image_size=image_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.model = self._build_model(self.device)
        except Exception as exc:
            if self.device.type == "cuda":
                print(f"[tags][ram] cuda load failed: {exc}; fallback to cpu")
                self.device = torch.device("cpu")
                self.model = self._build_model(self.device)
            else:
                raise
        self._inference = inference

    def _build_model(self, device: torch.device):
        model = self._ram_plus(pretrained=self.model_path, image_size=self.image_size, vit=self.vit)
        model.eval()
        model = model.to(device)
        return model

    def _fallback_to_cpu(self, reason: Exception) -> None:
        if self.device.type == "cpu":
            return
        print(f"[tags][ram] fallback to cpu due to: {reason}")
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        self.device = torch.device("cpu")
        self.model = self._build_model(self.device)

    def predict_labels(self, image_path: str) -> list[str]:
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        try:
            with torch.no_grad():
                result = self._inference(tensor, self.model)
        except Exception as exc:
            if self.device.type == "cuda" and is_cuda_oom(exc):
                self._fallback_to_cpu(exc)
                tensor = tensor.to(self.device)
                with torch.no_grad():
                    result = self._inference(tensor, self.model)
            else:
                raise

        english_tags = ""
        if isinstance(result, (tuple, list)) and len(result) > 0:
            english_tags = str(result[0])
        else:
            english_tags = str(result)
        return parse_labels(english_tags)


# ===== integrated optional clean logic =====


def load_support_from_table_rows(rows: list[dict[str, object]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        label = normalize_label(row.get("label", ""))
        if not label:
            continue
        support = int(row.get("support", 0))
        out[label] = max(support, out.get(label, 0))
    return out


def compute_support_from_tags(tags_by_image: dict[str, list[str]]) -> dict[str, int]:
    support = defaultdict(int)
    for labels in tags_by_image.values():
        for label in set(labels):
            support[label] += 1
    return dict(support)


def build_semantic_table(tags_by_image: dict[str, list[str]]) -> list[dict[str, object]]:
    support = compute_support_from_tags(tags_by_image)
    sorted_labels = sorted(support.items(), key=lambda item: (-item[1], item[0]))
    rows: list[dict[str, object]] = []
    for idx, (label, count) in enumerate(sorted_labels, start=1):
        rows.append({"class_id": idx, "label": label, "support": count})
    return rows


def extract_json_candidate(text: str) -> object | None:
    fenced = re.findall(r"```json\s*(\{.*?\}|\[.*?\])\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    for block in fenced:
        try:
            return json.loads(block)
        except json.JSONDecodeError:
            continue

    start_obj, end_obj = text.find("{"), text.rfind("}")
    if start_obj >= 0 and end_obj > start_obj:
        snippet = text[start_obj : end_obj + 1]
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            pass

    start_arr, end_arr = text.find("["), text.rfind("]")
    if start_arr >= 0 and end_arr > start_arr:
        snippet = text[start_arr : end_arr + 1]
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            pass
    return None


def parse_mapping_response(raw_text: str, input_labels: list[str]) -> dict[str, str | None]:
    allowed = set(input_labels)
    mapping: dict[str, str | None] = {label: label for label in input_labels}
    payload = extract_json_candidate(raw_text)
    if payload is None:
        return mapping

    entries: list[dict[str, Any]] = []
    if isinstance(payload, dict):
        if isinstance(payload.get("mapping"), list):
            entries = [x for x in payload["mapping"] if isinstance(x, dict)]
        elif all(isinstance(k, str) for k in payload.keys()):
            for key, value in payload.items():
                if isinstance(value, dict):
                    row = {"label": key}
                    row.update(value)
                    entries.append(row)
                else:
                    entries.append({"label": key, "canonical_label": value, "action": "keep"})
    elif isinstance(payload, list):
        entries = [x for x in payload if isinstance(x, dict)]

    for row in entries:
        label = normalize_label(row.get("label", ""))
        if label not in allowed:
            continue

        action = str(row.get("action", "keep")).strip().lower()
        canonical_raw = row.get("canonical_label", label)
        canonical = normalize_label(canonical_raw)

        if action in {"drop", "remove", "discard"}:
            mapping[label] = None
            continue
        if not canonical:
            mapping[label] = None
            continue
        mapping[label] = canonical
    return mapping


def build_single_pass_prompt(labels_with_support: list[tuple[str, int]]) -> str:
    payload = [{"label": label, "support": int(support)} for label, support in labels_with_support]
    return (
        "You are cleaning open-vocabulary object labels for visual detection.\n"
        "Process ALL input labels in ONE pass. Do not skip any label.\n"
        "Apply exactly these rules:\n"
        "1) Drop obvious generic or underspecified words.\n"
        "2) Normalize case, punctuation, and singular/plural forms.\n"
        "3) Semantically cluster same object with different modifiers into one canonical label.\n"
        "4) Merge different parts of an inseparable object into one object-level label.\n"
        "Output JSON only. Required schema:\n"
        "{\n"
        '  "mapping": [\n'
        '    {"label":"<input_label>","action":"keep|drop","canonical_label":"<final_label_or_empty_if_drop>"}\n'
        "  ]\n"
        "}\n"
        "Constraints:\n"
        "- mapping must include EVERY input label exactly once.\n"
        "- canonical_label must be a concise noun phrase in lowercase English.\n"
        "- If uncertain, keep instead of drop.\n"
        "Input labels:\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


class QwenTextClusterer:
    def __init__(self, model_dir: str, max_new_tokens: int = 4096, temperature: float = 0.0):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.backend = "unknown"

        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
        model_type = str(getattr(cfg, "model_type", "")).lower()
        is_vl_model = ("qwen3_vl" in model_type) or hasattr(cfg, "vision_config")

        if is_vl_model:
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

            self.vl_processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
            try:
                self.vl_model = Qwen3VLForConditionalGeneration.from_pretrained(
                    model_dir,
                    torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                    device_map="cuda" if self.device == "cuda" else "cpu",
                    trust_remote_code=True,
                )
            except Exception as exc:
                if self.device == "cuda":
                    print(f"[tags][clean] qwen3vl cuda load failed: {exc}; fallback to cpu")
                    self.device = "cpu"
                    self.vl_model = Qwen3VLForConditionalGeneration.from_pretrained(
                        model_dir,
                        torch_dtype=torch.float32,
                        device_map="cpu",
                        trust_remote_code=True,
                    )
                else:
                    raise
            self.backend = "qwen3vl"
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            device_map="cuda" if self.device == "cuda" else "cpu",
            trust_remote_code=True,
        )
        self.backend = "causal_lm"

    def generate(self, prompt: str) -> str:
        do_sample = self.temperature > 1e-6
        if self.backend == "qwen3vl":
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            chat_text = self.vl_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.vl_processor(text=[chat_text], padding=True, return_tensors="pt")
            inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}

            gen_kwargs: dict[str, Any] = {
                "max_new_tokens": self.max_new_tokens,
                "do_sample": do_sample,
            }
            if do_sample:
                gen_kwargs["temperature"] = self.temperature

            generated = self.vl_model.generate(**inputs, **gen_kwargs)
            trimmed = generated[0][len(inputs["input_ids"][0]) :]
            return self.vl_processor.decode(trimmed, skip_special_tokens=True)

        if hasattr(self.tokenizer, "apply_chat_template"):
            chat_text = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            chat_text = prompt

        inputs = self.tokenizer([chat_text], return_tensors="pt")
        inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        gen_kwargs = {"max_new_tokens": self.max_new_tokens, "do_sample": do_sample}
        if do_sample:
            gen_kwargs["temperature"] = self.temperature
        generated = self.model.generate(**inputs, **gen_kwargs)
        trimmed = generated[0][len(inputs["input_ids"][0]) :]
        return self.tokenizer.decode(trimmed, skip_special_tokens=True)


def apply_mapping(tags_by_image: dict[str, list[str]], mapping: dict[str, str | None]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for image_id, labels in tags_by_image.items():
        mapped: list[str] = []
        for label in labels:
            canonical = mapping.get(label, label)
            if canonical is None:
                continue
            mapped.append(canonical)
        out[image_id] = sorted(set(mapped))
    return out


def write_basic_outputs(out_root: Path, tags_by_image: dict[str, list[str]], semantic_table: list[dict[str, object]]) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    prompt = ". ".join([str(row["label"]) for row in semantic_table])
    class_id_map = {str(row["label"]): int(row["class_id"]) for row in semantic_table}

    save_json(out_root / "tags_per_image.json", tags_by_image)
    save_json(out_root / "semantic_label_table.json", semantic_table)
    save_json(out_root / "class_id_map.json", class_id_map)
    (out_root / "semantic_prompt.txt").write_text(prompt, encoding="utf-8")


def run_optional_clean(
    tags_by_image: dict[str, list[str]],
    semantic_table: list[dict[str, object]],
    args: argparse.Namespace,
) -> tuple[dict[str, list[str]], list[dict[str, object]], dict[str, object], dict[str, str | None], list[str], list[dict[str, object]]]:
    support = load_support_from_table_rows(semantic_table)
    support_from_tags = compute_support_from_tags(tags_by_image)
    for label, count in support_from_tags.items():
        if label not in support:
            support[label] = count

    labels = sorted(support.keys(), key=lambda x: (-support[x], x))
    if not labels:
        return tags_by_image, semantic_table, {
            "mode": args.clean_mode,
            "backend": "none",
            "num_input_labels": 0,
            "num_output_labels": 0,
            "num_dropped_labels": 0,
            "num_merged_labels": 0,
            "single_pass": True,
        }, {}, [], []

    if args.clean_mode in {"off", "identity"}:
        alias_to_canonical: dict[str, str | None] = {label: label for label in labels}
        backend = args.clean_mode
    else:
        clean_model_dir = args.clean_model_dir or args.qwen_model_dir
        if not clean_model_dir:
            raise RuntimeError("--clean-model-dir or --qwen-model-dir is required when --clean-mode qwen")
        clusterer = QwenTextClusterer(
            model_dir=clean_model_dir,
            max_new_tokens=args.clean_max_new_tokens,
            temperature=args.clean_temperature,
        )
        prompt = build_single_pass_prompt([(label, support[label]) for label in labels])
        response = clusterer.generate(prompt)
        alias_to_canonical = parse_mapping_response(response, labels)
        backend = clusterer.backend

    tags_cleaned = apply_mapping(tags_by_image, alias_to_canonical)
    semantic_cleaned = build_semantic_table(tags_cleaned)

    groups = defaultdict(list)
    dropped: list[str] = []
    for label in labels:
        canonical = alias_to_canonical.get(label, label)
        if canonical is None:
            dropped.append(label)
            continue
        groups[canonical].append(label)

    support_after = compute_support_from_tags(tags_cleaned)
    canonical_groups: list[dict[str, object]] = []
    for canonical in sorted(groups.keys(), key=lambda x: (-support_after.get(x, 0), x)):
        canonical_groups.append(
            {
                "canonical_label": canonical,
                "support": int(support_after.get(canonical, 0)),
                "aliases": sorted(groups[canonical]),
                "num_aliases": len(groups[canonical]),
            }
        )

    report: dict[str, object] = {
        "mode": args.clean_mode,
        "backend": backend,
        "num_input_labels": len(labels),
        "num_output_labels": len(semantic_cleaned),
        "num_dropped_labels": len(dropped),
        "num_merged_labels": len(labels) - len(dropped) - len(semantic_cleaned),
        "single_pass": True,
        "rules": {
            "drop_generic_terms": True,
            "normalize_case_plural_punctuation": True,
            "merge_modified_same_object": True,
            "merge_parts_of_inseparable_objects": True,
        },
    }

    return tags_cleaned, semantic_cleaned, report, alias_to_canonical, dropped, canonical_groups


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate semantic labels with Qwen3-VL or RAM++, with optional integrated cleaning"
    )
    parser.add_argument(
        "--mode",
        choices=["full", "clean_only"],
        default="full",
        help="full: run tagger and optional clean; clean_only: only run clean on an existing tags dir",
    )
    parser.add_argument("--image-dir", default=None, help="Input image directory (required for --mode full)")
    parser.add_argument("--output-dir", default=None, help="Output directory for raw tags (required for --mode full)")
    parser.add_argument(
        "--tags-dir",
        default=None,
        help="Existing tags directory containing tags_per_image.json and semantic_label_table.json (required for --mode clean_only)",
    )
    parser.add_argument(
        "--tagger",
        choices=["qwen", "ram"],
        default="qwen",
        help="Tagger backend",
    )
    parser.add_argument(
        "--qwen-model-dir",
        default=None,
        help="Qwen model directory",
    )
    parser.add_argument(
        "--ram-model-path",
        default=None,
        help="RAM++ checkpoint path (e.g. ram_plus_swin_large_14m.pth)",
    )
    parser.add_argument(
        "--ram-image-size",
        type=int,
        default=384,
        help="RAM image size",
    )
    parser.add_argument(
        "--ram-vit",
        default="swin_l",
        help="RAM backbone type",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=200,
        help="Generation length for Qwen (ignored for RAM)",
    )

    # integrated optional cleaning
    parser.add_argument(
        "--clean-mode",
        choices=["off", "identity", "qwen"],
        default="off",
        help="Optional integrated clean mode",
    )
    parser.add_argument(
        "--clean-output-dir",
        default=None,
        help="Optional output directory for cleaned tags",
    )
    parser.add_argument(
        "--clean-model-dir",
        default=None,
        help="Model directory used by clean-mode=qwen. Defaults to --qwen-model-dir.",
    )
    parser.add_argument(
        "--clean-max-new-tokens",
        type=int,
        default=4096,
        help="Generation length for integrated cleaning",
    )
    parser.add_argument(
        "--clean-temperature",
        type=float,
        default=0.0,
        help="Generation temperature for integrated cleaning",
    )

    parser.add_argument("--max-images", type=int, default=None, help="Optional test limit")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.mode == "clean_only":
        if not args.tags_dir:
            raise RuntimeError("--tags-dir is required when --mode clean_only")
        if not args.clean_output_dir:
            raise RuntimeError("--clean-output-dir is required when --mode clean_only")

        tags_root = Path(args.tags_dir)
        tags_per_image_path = tags_root / "tags_per_image.json"
        semantic_table_path = tags_root / "semantic_label_table.json"
        if not tags_per_image_path.exists():
            raise RuntimeError(f"Missing tags file: {tags_per_image_path}")
        if not semantic_table_path.exists():
            raise RuntimeError(f"Missing semantic table: {semantic_table_path}")

        tags_by_image = load_json(tags_per_image_path)
        semantic_table = load_json(semantic_table_path)
        clean_root = Path(args.clean_output_dir)

        (
            tags_cleaned,
            semantic_cleaned,
            report,
            alias_to_canonical,
            dropped,
            canonical_groups,
        ) = run_optional_clean(tags_by_image, semantic_table, args)

        write_basic_outputs(clean_root, tags_cleaned, semantic_cleaned)
        save_json(clean_root / "alias_to_canonical.json", alias_to_canonical)
        save_json(clean_root / "dropped_labels.json", sorted(dropped))
        save_json(clean_root / "canonical_groups.json", canonical_groups)
        save_json(clean_root / "clean_report.json", report)

        print(f"[tags][clean_only] mode={args.clean_mode}")
        print(f"[tags][clean_only] output labels: {len(semantic_cleaned)}")
        print(f"[tags][clean_only] report: {clean_root / 'clean_report.json'}")
        return

    if not args.image_dir:
        raise RuntimeError("--image-dir is required when --mode full")
    if not args.output_dir:
        raise RuntimeError("--output-dir is required when --mode full")

    images = collect_images(args.image_dir)
    if args.max_images is not None and args.max_images > 0:
        images = images[: args.max_images]
    if not images:
        raise RuntimeError(f"No RGB image found in: {args.image_dir}")

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.tagger == "qwen":
        if not args.qwen_model_dir:
            raise RuntimeError("--qwen-model-dir is required when --tagger qwen")
        tagger = QwenTagger(model_dir=args.qwen_model_dir, max_new_tokens=args.max_new_tokens)
    else:
        if not args.ram_model_path:
            raise RuntimeError("--ram-model-path is required when --tagger ram")
        tagger = RamTagger(
            model_path=args.ram_model_path,
            image_size=args.ram_image_size,
            vit=args.ram_vit,
        )

    tags_by_image: dict[str, list[str]] = {}
    for path in images:
        labels = tagger.predict_labels(str(path))
        tags_by_image[path.stem.lower()] = labels
        print(f"[tags] {path.name}: {len(labels)} labels")

    semantic_table = build_semantic_table(tags_by_image)
    write_basic_outputs(out_root, tags_by_image, semantic_table)

    print(f"Semantic table: {out_root / 'semantic_label_table.json'}")
    print(f"Prompt: {out_root / 'semantic_prompt.txt'}")
    print(f"Class map: {out_root / 'class_id_map.json'}")

    if args.clean_output_dir:
        clean_root = Path(args.clean_output_dir)
        (
            tags_cleaned,
            semantic_cleaned,
            report,
            alias_to_canonical,
            dropped,
            canonical_groups,
        ) = run_optional_clean(tags_by_image, semantic_table, args)

        write_basic_outputs(clean_root, tags_cleaned, semantic_cleaned)
        save_json(clean_root / "alias_to_canonical.json", alias_to_canonical)
        save_json(clean_root / "dropped_labels.json", sorted(dropped))
        save_json(clean_root / "canonical_groups.json", canonical_groups)
        save_json(clean_root / "clean_report.json", report)

        print(f"[tags][clean] mode={args.clean_mode}")
        print(f"[tags][clean] output labels: {len(semantic_cleaned)}")
        print(f"[tags][clean] report: {clean_root / 'clean_report.json'}")


if __name__ == "__main__":
    main()
