#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

IMAGE_DIR="${1:-$PROJECT_ROOT/examples}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
if [ -n "${2:-}" ]; then
    RUN_NAME="${2}_${TIMESTAMP}"
else
    RUN_NAME="demo_${TIMESTAMP}"
fi

EL_ENV="${EL_ENV:-Embodied-Label}"
TAGS_ENV="${TAGS_ENV:-qwen}"
RAM_ENV="${RAM_ENV:-/mnt/20T/jwt/conda/envs/recognize-anything}"
TAGGER="${TAGGER:-qwen}"

QWEN_MODEL_DIR="${QWEN_MODEL_DIR:-$PROJECT_ROOT/models/Qwen3-VL-4B-Instruct}"
RAM_MODEL_PATH="${RAM_MODEL_PATH:-$PROJECT_ROOT/models/recognize-anything-plus-model/ram_plus_swin_large_14m.pth}"
RAM_IMAGE_SIZE="${RAM_IMAGE_SIZE:-384}"
RAM_VIT="${RAM_VIT:-swin_l}"
TAG_CLEAN_MODEL_DIR="${TAG_CLEAN_MODEL_DIR:-$QWEN_MODEL_DIR}"
TAG_CLEAN_MODE="${TAG_CLEAN_MODE:-qwen}"
TAG_CLEAN_MAX_NEW_TOKENS="${TAG_CLEAN_MAX_NEW_TOKENS:-4096}"
TAG_CLEAN_TEMPERATURE="${TAG_CLEAN_TEMPERATURE:-0.0}"
MOGE_MODEL_PATH="${MOGE_MODEL_PATH:-$PROJECT_ROOT/models/moge-2-vitl-normal/model.pt}"
STEP5_DOWNSAMPLE_POINTS="${STEP5_DOWNSAMPLE_POINTS:-0}"
STEP5_EDGE_FILTER_THICKNESS="${STEP5_EDGE_FILTER_THICKNESS:-1}"
STEP5_EDGE_FILTER_TOL="${STEP5_EDGE_FILTER_TOL:-0.04}"
STEP4_MASK_SOURCE="${STEP4_MASK_SOURCE:-rgb}"
STEP8_BBOX_MODE="${STEP8_BBOX_MODE:-aabb}"
STEP8_SPLIT_PER_SCENE="${STEP8_SPLIT_PER_SCENE:-0}"

GSAM2_ROOT="${GSAM2_ROOT:-$PROJECT_ROOT/lib/Grounded-SAM-2}"
SAM_CONFIG_DIR="${SAM_CONFIG_DIR:-$GSAM2_ROOT/sam2/configs}"
SAM_CONFIG_NAME="${SAM_CONFIG_NAME:-sam2.1/sam2.1_hiera_l.yaml}"
GDINO_CONFIG="${GDINO_CONFIG:-$GSAM2_ROOT/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py}"
SAM_CHECKPOINT="${SAM_CHECKPOINT:-}"
GDINO_CHECKPOINT="${GDINO_CHECKPOINT:-}"

if [[ -z "$SAM_CHECKPOINT" ]]; then
  if [[ -f "$PROJECT_ROOT/models/sam_checkpoints/sam2.1_hiera_large.pt" ]]; then
    SAM_CHECKPOINT="$PROJECT_ROOT/models/sam_checkpoints/sam2.1_hiera_large.pt"
  else
    SAM_CHECKPOINT="$GSAM2_ROOT/checkpoints/sam2.1_hiera_large.pt"
  fi
fi

if [[ -z "$GDINO_CHECKPOINT" ]]; then
  if [[ -f "$PROJECT_ROOT/models/gdino_checkpoints/groundingdino_swint_ogc.pth" ]]; then
    GDINO_CHECKPOINT="$PROJECT_ROOT/models/gdino_checkpoints/groundingdino_swint_ogc.pth"
  else
    GDINO_CHECKPOINT="$GSAM2_ROOT/gdino_checkpoints/groundingdino_swint_ogc.pth"
  fi
fi


OUT_ROOT="${OUT_ROOT:-$PROJECT_ROOT/logs/$RUN_NAME}"
RESIZE_DIR="$OUT_ROOT/01_resized"
TAGS_DIR="$OUT_ROOT/02_tags"
# Keep cleaned tags in the same stage directory as raw tags (no separate 02_5 dir).
TAGS_CLEAN_DIR="$TAGS_DIR"
MOGE_DIR="$OUT_ROOT/03_moge"
SEG2D_DIR="$OUT_ROOT/04_seg2d"
BACKPROJECT_DIR="$OUT_ROOT/05_backproject"
DOMINANT_NORMAL_DIR="$OUT_ROOT/06_dominant_normal"
WORLD_DIR="$OUT_ROOT/07_world"
PACKAGE_DIR="$OUT_ROOT/08_package"

require_exists() {
  local path="$1"
  local label="$2"
  if [[ ! -e "$path" ]]; then
    echo "Error: $label not found: $path" >&2
    exit 1
  fi
}

conda_run_env() {
  local env_spec="$1"
  shift
  if [[ "$env_spec" == /* ]]; then
    conda run -p "$env_spec" "$@"
  else
    conda run -n "$env_spec" "$@"
  fi
}

is_truthy() {
  local v="${1:-}"
  [[ "$v" == "1" || "$v" == "true" || "$v" == "TRUE" ]]
}

require_exists "$IMAGE_DIR" "image directory"
if [[ "$TAGGER" == "qwen" ]]; then
  require_exists "$QWEN_MODEL_DIR" "Qwen model directory"
elif [[ "$TAGGER" == "ram" ]]; then
  require_exists "$RAM_MODEL_PATH" "RAM model checkpoint"
  if [[ "$RAM_ENV" == /* ]]; then
    require_exists "$RAM_ENV" "RAM conda environment path"
  fi
else
  echo "Error: TAGGER must be qwen or ram, got: $TAGGER" >&2
  exit 1
fi
if [[ "$TAG_CLEAN_MODE" == "qwen" ]]; then
  require_exists "$TAG_CLEAN_MODEL_DIR" "tag-clean model directory"
elif [[ "$TAG_CLEAN_MODE" == "identity" || "$TAG_CLEAN_MODE" == "off" ]]; then
  :
else
  echo "Error: TAG_CLEAN_MODE must be qwen/identity/off, got: $TAG_CLEAN_MODE" >&2
  exit 1
fi
if [[ "$STEP4_MASK_SOURCE" != "rgb" && "$STEP4_MASK_SOURCE" != "normal" && "$STEP4_MASK_SOURCE" != "crop" ]]; then
  echo "Error: STEP4_MASK_SOURCE must be rgb/normal/crop, got: $STEP4_MASK_SOURCE" >&2
  exit 1
fi
require_exists "$MOGE_MODEL_PATH" "MoGE model checkpoint"
require_exists "$GSAM2_ROOT" "Grounded-SAM-2 root"
require_exists "$SAM_CONFIG_DIR" "SAM config directory"
require_exists "$SAM_CHECKPOINT" "SAM checkpoint"
require_exists "$GDINO_CONFIG" "GroundingDINO config"
require_exists "$GDINO_CHECKPOINT" "GroundingDINO checkpoint"

mkdir -p "$OUT_ROOT"

echo "[1/8] resize"
conda_run_env "$EL_ENV" python -m src.step1_resize_images \
  --input-dir "$IMAGE_DIR" \
  --output-dir "$RESIZE_DIR" \
  --multiple 14

echo "[2/8] tags"
if [[ "$TAGGER" == "ram" && "$TAG_CLEAN_MODE" == "qwen" ]]; then
  echo "[2/8] tags raw (RAM env)"
  conda_run_env "$RAM_ENV" python -m src.step2_generate_tags \
    --mode full \
    --image-dir "$RESIZE_DIR" \
    --output-dir "$TAGS_DIR" \
    --tagger ram \
    --ram-model-path "$RAM_MODEL_PATH" \
    --ram-image-size "$RAM_IMAGE_SIZE" \
    --ram-vit "$RAM_VIT" \
    --clean-mode off

  echo "[2/8] tags clean_only (Qwen env)"
  conda_run_env "$TAGS_ENV" python -m src.step2_generate_tags \
    --mode clean_only \
    --tags-dir "$TAGS_DIR" \
    --clean-mode qwen \
    --clean-output-dir "$TAGS_CLEAN_DIR" \
    --clean-model-dir "$TAG_CLEAN_MODEL_DIR" \
    --clean-max-new-tokens "$TAG_CLEAN_MAX_NEW_TOKENS" \
    --clean-temperature "$TAG_CLEAN_TEMPERATURE"
else
  if [[ "$TAGGER" == "ram" ]]; then
    STEP2_ENV="$RAM_ENV"
  else
    STEP2_ENV="$TAGS_ENV"
  fi
  STEP2_CMD=(python -m src.step2_generate_tags \
    --mode full \
    --image-dir "$RESIZE_DIR" \
    --output-dir "$TAGS_DIR" \
    --tagger "$TAGGER" \
    --clean-mode "$TAG_CLEAN_MODE" \
    --clean-output-dir "$TAGS_CLEAN_DIR" \
    --clean-max-new-tokens "$TAG_CLEAN_MAX_NEW_TOKENS" \
    --clean-temperature "$TAG_CLEAN_TEMPERATURE")
  if [[ "$TAGGER" == "qwen" ]]; then
    STEP2_CMD+=(--qwen-model-dir "$QWEN_MODEL_DIR")
  else
    STEP2_CMD+=(--ram-model-path "$RAM_MODEL_PATH" --ram-image-size "$RAM_IMAGE_SIZE" --ram-vit "$RAM_VIT")
  fi
  if [[ "$TAG_CLEAN_MODE" == "qwen" ]]; then
    STEP2_CMD+=(--clean-model-dir "$TAG_CLEAN_MODEL_DIR")
  fi
  conda_run_env "$STEP2_ENV" "${STEP2_CMD[@]}"
fi

echo "[3/8] estimate_geometry"
conda_run_env "$EL_ENV" python -m src.step3_estimate_geometry \
  --image-dir "$RESIZE_DIR" \
  --output-dir "$MOGE_DIR" \
  --moge-model-path "$MOGE_MODEL_PATH" \
  --device cuda

echo "[4/8] segment"
STEP4_CMD=(python -m src.step4_segment_2d \
  --image-dir "$RESIZE_DIR" \
  --semantic-table "$TAGS_CLEAN_DIR/semantic_label_table.json" \
  --prompt-path "$TAGS_CLEAN_DIR/semantic_prompt.txt" \
  --tags-per-image "$TAGS_CLEAN_DIR/tags_per_image.json" \
  --moge-dir "$MOGE_DIR" \
  --output-dir "$SEG2D_DIR" \
  --gsam2-root "$GSAM2_ROOT" \
  --sam-config-dir "$SAM_CONFIG_DIR" \
  --sam-config-name "$SAM_CONFIG_NAME" \
  --sam-checkpoint "$SAM_CHECKPOINT" \
  --gdino-config "$GDINO_CONFIG" \
  --gdino-checkpoint "$GDINO_CHECKPOINT" \
  --box-threshold 0.28 \
  --text-threshold 0.28 \
  --mask-source "$STEP4_MASK_SOURCE")
conda_run_env "$EL_ENV" "${STEP4_CMD[@]}"

echo "[5/8] backproject_3d"
conda_run_env "$EL_ENV" python -m src.step5_backproject_3d \
  --image-dir "$RESIZE_DIR" \
  --moge-dir "$MOGE_DIR" \
  --seg-dir "$SEG2D_DIR" \
  --output-dir "$BACKPROJECT_DIR" \
  --downsample-points "$STEP5_DOWNSAMPLE_POINTS" \
  --edge-filter-thickness "$STEP5_EDGE_FILTER_THICKNESS" \
  --edge-filter-tol "$STEP5_EDGE_FILTER_TOL"

echo "[6/8] dominant_normal"
conda_run_env "$EL_ENV" python -m src.step6_estimate_dominant_normal \
  --backproject-dir "$BACKPROJECT_DIR" \
  --output-dir "$DOMINANT_NORMAL_DIR" \
  --iterations 256 \
  --angle-threshold-deg 5.0

echo "[7/8] build_world_frame"
conda_run_env "$EL_ENV" python -m src.step7_build_world_frame \
  --backproject-dir "$BACKPROJECT_DIR" \
  --normal-dir "$DOMINANT_NORMAL_DIR" \
  --output-dir "$WORLD_DIR" \
  --n-bins 500 \
  --table-threshold 0.03

echo "[8/8] package_metadata"
STEP8_CMD=(python -m src.step8_package_metadata \
  --image-dir "$RESIZE_DIR" \
  --tags-dir "$TAGS_DIR" \
  --tags-clean-dir "$TAGS_CLEAN_DIR" \
  --seg-dir "$SEG2D_DIR" \
  --moge-dir "$MOGE_DIR" \
  --backproject-dir "$BACKPROJECT_DIR" \
  --normal-dir "$DOMINANT_NORMAL_DIR" \
  --world-dir "$WORLD_DIR" \
  --output-dir "$PACKAGE_DIR" \
  --dataset embodied-label \
  --bbox-mode "$STEP8_BBOX_MODE")
if is_truthy "$STEP8_SPLIT_PER_SCENE"; then
  STEP8_CMD+=(--split-per-scene)
fi
conda_run_env "$EL_ENV" "${STEP8_CMD[@]}"

echo "Done. Outputs: $OUT_ROOT"
