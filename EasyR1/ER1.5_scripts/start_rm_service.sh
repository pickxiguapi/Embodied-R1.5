#!/bin/bash
set -x

export CUDA_VISIBLE_DEVICES=7
export no_proxy="$no_proxy,127.0.0.1,localhost,0.0.0.0"
export NO_PROXY="$NO_PROXY,127.0.0.1,localhost,0.0.0.0"
echo "Starting Skywork RM service on GPU 7..."

python -m sglang.launch_server \
    --served-model-name Skywork-Reward-V2-Qwen3-8B \
    --model-path /path/to/Skywork-Reward-V2-Qwen3-8B \
    --mem-fraction-static 0.8 \
    --tp 1 \
    --port 18889 \
    --host 0.0.0.0 \
    --is-embedding

echo "RM service started on port 18889"
