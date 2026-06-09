import os
import sys
from typing import Any

from omegaconf import OmegaConf

from verl.trainer.config import PPOConfig
from verl.utils.dataset import RLHFDataset
from verl.utils.tokenizer import get_processor, get_tokenizer


def _print_env():
    cache_dir = os.environ.get("ER15_VIDEO_CACHE_DIR", "").strip()
    cache_tag = os.environ.get("ER15_VIDEO_CACHE_TAG", "").strip()

    os.environ["ER15_VIDEO_CACHE_WRITE"] = "1"
    cache_write = os.environ.get("ER15_VIDEO_CACHE_WRITE", "").strip()

    print(f"[ENV] ER15_VIDEO_CACHE_DIR={cache_dir}")
    print(f"[ENV] ER15_VIDEO_CACHE_TAG={cache_tag}")
    print(f"[ENV] ER15_VIDEO_CACHE_WRITE={cache_write}")

    if not cache_dir:
        raise RuntimeError("ER15_VIDEO_CACHE_DIR is empty")

    if cache_write.strip().lower() not in ("1", "true", "yes"):
        raise RuntimeError("ER15_VIDEO_CACHE_WRITE must be one of: 1, true, yes")


def build_ppo_config() -> PPOConfig:
    cli_args = OmegaConf.from_cli()
    default_config = OmegaConf.structured(PPOConfig())

    if hasattr(cli_args, "config"):
        config_path = cli_args.pop("config", None)
        file_config = OmegaConf.load(config_path)
        default_config = OmegaConf.merge(default_config, file_config)

    merged = OmegaConf.merge(default_config, cli_args)
    cfg: PPOConfig = OmegaConf.to_object(merged)
    cfg.deep_post_init()
    return cfg


def main():
    _print_env()

    cfg = build_ppo_config()
    print("[INFO] Loaded PPOConfig")

    model_path = cfg.worker.actor.model.model_path
    tokenizer = get_tokenizer(
        model_path,
        override_chat_template=cfg.data.override_chat_template,
        trust_remote_code=cfg.worker.actor.model.trust_remote_code,
        use_fast=True,
    )
    processor = get_processor(
        model_path,
        override_chat_template=cfg.data.override_chat_template,
        trust_remote_code=cfg.worker.actor.model.trust_remote_code,
        use_fast=True,
    )

    if processor is None:
        raise RuntimeError("processor is required for video preprocessing")

    ds = RLHFDataset(
        data_path=cfg.data.train_files,
        tokenizer=tokenizer,
        processor=processor,
        prompt_key=cfg.data.prompt_key,
        answer_key=cfg.data.answer_key,
        image_key=cfg.data.image_key,
        video_key=cfg.data.video_key,
        problem_type_key=cfg.data.problem_type_key,
        problem_id_key=cfg.data.problem_id_key,
        options_key=cfg.data.options_key,
        data_type_key=cfg.data.data_type_key,
        data_source_key=cfg.data.data_source_key,
        image_dir=cfg.data.image_dir,
        max_frames=cfg.data.max_frames,
        max_prompt_length=cfg.data.max_prompt_length,
        truncation="right",
        format_prompt=cfg.data.format_prompt,
        min_pixels=cfg.data.min_pixels,
        max_pixels=cfg.data.max_pixels,
        filter_overlong_prompts=cfg.data.filter_overlong_prompts,
        filter_overlong_prompts_workers=cfg.data.filter_overlong_prompts_workers,
        video_fps=cfg.data.video_fps,
        debug=False,
    )

    total = len(ds)
    print(f"[INFO] Dataset size={total}")

    wrote = 0
    existed = 0
    failed = 0
    skipped_non_video = 0

    for i in range(total):
        raw: dict[str, Any] = ds.dataset[i]
        data_type = str(raw.get(cfg.data.data_type_key, "")).strip().lower()
        if data_type != "video":
            skipped_non_video += 1
            continue

        dataset_name = raw.get("dataset_name", None)
        problem_id = raw.get(cfg.data.problem_id_key, None)
        if dataset_name is None or problem_id is None:
            raise KeyError(
                f"Video cache key requires 'dataset_name' and '{cfg.data.problem_id_key}', got dataset_name={dataset_name}, {cfg.data.problem_id_key}={problem_id}"
            )

        cache_dir = os.environ.get("ER15_VIDEO_CACHE_DIR", "").strip()
        cache_tag = os.environ.get("ER15_VIDEO_CACHE_TAG", "").strip() or "default"
        cache_path = os.path.join(cache_dir, cache_tag, dataset_name, f"{problem_id}.pt")

        if os.path.exists(cache_path):
            existed += 1
        else:
            wrote += 1

        try:
            _ = ds[i]
        except Exception as e:
            failed += 1
            print(
                f"[ERROR] index={i} dataset_name={dataset_name} problem_id={problem_id} cache_path={cache_path} err={repr(e)}",
                file=sys.stderr,
            )

        if (i + 1) % 100 == 0:
            print(
                f"[PROGRESS] {i+1}/{total}  wrote~{wrote} existed~{existed} failed={failed} skipped_non_video={skipped_non_video}",
                flush=True,
            )

    print("\n[SUMMARY]")
    print(f"total={total}")
    print(f"video_cache_will_write_or_written={wrote}")
    print(f"video_cache_already_exists={existed}")
    print(f"skipped_non_video={skipped_non_video}")
    print(f"failed={failed}")


if __name__ == "__main__":
    main()
