#!/bin/bash
set -x
set -e

# Video cache precompute launcher.
#
# You can either:
#   (A) Edit the defaults below, or
#   (B) Export ER15_VIDEO_CACHE_DIR / ER15_VIDEO_CACHE_TAG before running this script.
#
# Cache layout:
#   ${ER15_VIDEO_CACHE_DIR}/${ER15_VIDEO_CACHE_TAG}/${dataset_name}/${problem_id}.pt

# -----------------------------------------------------------------------------
# Cache settings (edit here if you want everything self-contained in this .sh)
# -----------------------------------------------------------------------------
# Default to a repo-local directory if not provided from environment.
export ER15_VIDEO_CACHE_DIR="${ER15_VIDEO_CACHE_DIR:-${PWD}/.er1_5_video_cache}"
# IMPORTANT: Change this tag whenever your yaml/sh parameters change.
export ER15_VIDEO_CACHE_TAG="${ER15_VIDEO_CACHE_TAG:-video_cache_v1_fps_2_frame_32}"


echo "[INFO] ER15_VIDEO_CACHE_DIR=${ER15_VIDEO_CACHE_DIR}"
echo "[INFO] ER15_VIDEO_CACHE_TAG=${ER15_VIDEO_CACHE_TAG}"

# -----------------------------------------------------------------------------
# Dataset / model settings
# -----------------------------------------------------------------------------
TRAIN_FILES="[rft_train_datasets/ER1.5_Cosmos_video_qa.json,rft_train_datasets/ER1.5_general_video_qa_50s_cleaned.json]"

# Reuse the same yaml as training (max_frames/video_fps/min/max_pixels, etc.)
CONFIG=ER1.5_scripts/Embodied-R1.5_config.yaml
MODEL_PATH="/path/to/Embodied-R1.5-SFT"          # edit: SFT model (used only for tokenizer/processor)
IMAGE_DIR="/path/to/rft/data"                     # edit: root directory for video files


python3 ER1.5_scripts/precompute_video_cache.py \
    config=${CONFIG} \
    worker.actor.model.model_path=${MODEL_PATH} \
    data.train_files=$TRAIN_FILES \
    data.image_dir=$IMAGE_DIR
