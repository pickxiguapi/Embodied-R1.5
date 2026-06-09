import argparse

import torch
from verl.utils.dataset import RLHFDataset
from verl.utils.tokenizer import get_processor, get_tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-file",
        default="rft_train_datasets/ER1.5_Cosmos_video_qa.json",
        # default="rft_train_datasets/ER1.5_general_video_qa.json",
        type=str,
    )
    parser.add_argument(
        "--model-path",
        default="/path/to/Embodied-R1.5-SFT",
        type=str,
        help="Local model directory (HuggingFace format); avoids downloading from Hub",
    )
    parser.add_argument(
        "--image-dir",
        default="/path/to/rft/data",
        type=str,
        help="Root directory for images/videos (IMAGE_DIR in rft_train.sh)",
    )
    parser.add_argument("--n", default=100, type=int, help="Print first n samples")
    parser.add_argument("--start", default=0, type=int, help="Start from sample index")
    parser.add_argument("--max-frames", default=32, type=int)
    parser.add_argument("--video-fps", default=2.0, type=float)
    parser.add_argument("--max-prompt-length", default=3300, type=int)
    args = parser.parse_args()

    print(f"[INFO] data_file={args.data_file}")
    print(f"[INFO] model_path={args.model_path}")
    print(f"[INFO] image_dir={args.image_dir}")

    tokenizer = get_tokenizer(
        args.model_path,
        trust_remote_code=True,
        use_fast=True,
    )
    processor = get_processor(
        args.model_path,
        trust_remote_code=True,
        use_fast=True,
    )
    if processor is None:
        raise RuntimeError("processor is None: this test requires a Processor (Qwen3-VL should have one)")

    ds = RLHFDataset(
        data_path=args.data_file,
        tokenizer=tokenizer,
        processor=processor,
        image_dir=args.image_dir,
        max_frames=args.max_frames,
        video_fps=args.video_fps,
        max_prompt_length=args.max_prompt_length,
        filter_overlong_prompts=False,  # debug: disable filtering overhead
        debug=False,
    )

    video_pad_id = None
    try:
        video_pad_id = processor.tokenizer.convert_tokens_to_ids("<|video_pad|>")
    except Exception:
        pass

    for i in range(args.start, args.start + args.n):
        ex = ds[i]
        data_type = ex.get("data_type", "")
        mm = ex.get("multi_modal_data", None)
        mm_kwargs = ex.get("mm_processor_kwargs", None)

        print("\n" + "=" * 80)
        print(f"[SAMPLE {i}] data_type={data_type}")
        print(f"keys={list(ex.keys())}")

        input_ids: torch.Tensor = ex["input_ids"]
        attention_mask: torch.Tensor = ex["attention_mask"]
        print(f"input_ids.shape={tuple(input_ids.shape)}  total_len={input_ids.numel()}")
        print(f"attention_mask.sum()={int(attention_mask.sum().item())}")

        if video_pad_id is not None:
            num_video_pad = int((input_ids == video_pad_id).sum().item())
            print(f"num(<|video_pad|>)={num_video_pad}")

        print(f"mm_processor_kwargs={mm_kwargs}")

        if mm is None:
            print("multi_modal_data=None")
            continue

        if "videos" in mm:
            videos = mm["videos"]
            v0 = videos[0]
            if hasattr(v0, "shape"):
                print(f"videos[0].shape={tuple(v0.shape)}  nframes={v0.shape[0]}")
            else:
                print(f"videos[0] type={type(v0)}")
        else:
            print(f"multi_modal_data keys={list(mm.keys())}")


if __name__ == "__main__":
    main()
