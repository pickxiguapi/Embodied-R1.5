## Scripts Overview

### `run_pipeline.bash`

Runs the full pipeline in 8 stages:

1. `src.step1_resize_images`
2. `src.step2_generate_tags` (includes optional integrated clean)
3. `src.step3_estimate_geometry`
4. `src.step4_segment_2d`
5. `src.step5_backproject_3d`
6. `src.step6_estimate_dominant_normal`
7. `src.step7_build_world_frame`
8. `src.step8_package_metadata`

All default paths are resolved relative to repo root.

### Usage

Run from project root:

```bash
bash scripts/run_pipeline.bash [IMAGE_DIR] [RUN_NAME]
```

- `IMAGE_DIR`: optional, default `./examples`
- `RUN_NAME`: optional, default `demo`; output path appends timestamp

### Environment variables

- `EL_ENV` (default: `Embodied-Label`): core runtime env
- `TAGS_ENV` (default: `qwen`): Qwen runtime for tagging/cleaning
- `TAGGER` (default: `qwen`; choices: `qwen`, `ram`)
- `QWEN_MODEL_DIR` (default: `./models/Qwen3-VL-4B-Instruct`)
- `RAM_ENV` (default: `/mnt/20T/jwt/conda/envs/recognize-anything`)
- `RAM_MODEL_PATH` (default: `./models/recognize-anything-plus-model/ram_plus_swin_large_14m.pth`)
- `RAM_IMAGE_SIZE` (default: `384`)
- `RAM_VIT` (default: `swin_l`)
- `TAG_CLEAN_MODEL_DIR` (default: same as `QWEN_MODEL_DIR`)
- `TAG_CLEAN_MODE` (default: `qwen`; use `identity` or `off` if needed)
- `TAG_CLEAN_MAX_NEW_TOKENS` (default: `4096`)
- `TAG_CLEAN_TEMPERATURE` (default: `0.0`)
- `MOGE_MODEL_PATH` (default: `./models/moge-2-vitl-normal/model.pt`)
- `GSAM2_ROOT` (default: `./lib/Grounded-SAM-2`)
- `SAM_CONFIG_DIR` (default: `$GSAM2_ROOT/sam2/configs`)
- `SAM_CONFIG_NAME` (default: `sam2.1/sam2.1_hiera_l.yaml`)
- `SAM_CHECKPOINT` (auto-detect; prefer `./models/sam_checkpoints/sam2.1_hiera_large.pt`, fallback `$GSAM2_ROOT/checkpoints/sam2.1_hiera_large.pt`)
- `GDINO_CONFIG` (default: `$GSAM2_ROOT/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py`)
- `GDINO_CHECKPOINT` (auto-detect; prefer `./models/gdino_checkpoints/groundingdino_swint_ogc.pth`, fallback `$GSAM2_ROOT/gdino_checkpoints/groundingdino_swint_ogc.pth`)
- `OUT_ROOT` (default: `./logs/<RUN_NAME>_<timestamp>`)
- `STEP5_DOWNSAMPLE_POINTS` (default: `0`)
- `STEP4_MASK_SOURCE` (default: `rgb`; choices: `rgb`, `normal`, `crop`)
- `STEP8_BBOX_MODE` (default: `aabb`; set `obb` for PCA-oriented boxes)
- `STEP8_SPLIT_PER_SCENE` (default: `0`; set `1` to split per-scene JSON)

### Example

```bash
CUDA_VISIBLE_DEVICES=1 \
EL_ENV=Embodied-Label \
TAGS_ENV=qwen \
TAGGER=qwen \
QWEN_MODEL_DIR=./models/Qwen3-VL-4B-Instruct \
bash scripts/run_pipeline.bash ./examples smoke
```

RAM-based tags example:

```bash
EL_ENV=Embodied-Label \
TAGGER=ram \
RAM_ENV=/mnt/20T/jwt/conda/envs/recognize-anything \
RAM_MODEL_PATH=./models/recognize-anything-plus-model/ram_plus_swin_large_14m.pth \
bash scripts/run_pipeline.bash ./examples smoke_ram
```

### `serve_ply_web.py`

Serve a browser-based PLY viewer for headless servers:

```bash
python scripts/serve_ply_web.py --root-dir ./logs --host 0.0.0.0 --port 8765
```

Then use SSH tunnel from local machine:

```bash
ssh -N -L 8765:127.0.0.1:8765 <user>@<server>
```

Open `http://127.0.0.1:8765`.

### `random_pick_first_images.py`

Randomly sample first images from leaf image folders:

```bash
python scripts/random_pick_first_images.py \
  --source-root /path/to/source \
  --target-path /any/path/sample_name \
  --num-samples 50 \
  --mode leaf_first_image
```

Outputs to `examples/<sample_name>_<timestamp>/` with `mapping.json`.
