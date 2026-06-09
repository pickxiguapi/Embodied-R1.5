#!/bin/bash
# =============================================================================
# Embodied-R1.5 RFT Training Script (2 nodes × 8 GPUs = 16 GPUs)
#
# Run from the EasyR1/ directory:
#   cd EasyR1
#   bash ER1.5_scripts/rft_train.sh
#
# Edit the variables in Section 1 before running.
# =============================================================================

# ==================== 1. Training Configuration ====================
MODEL_PATH="/path/to/Embodied-R1.5-SFT"          # SFT checkpoint (HuggingFace format)
IMAGE_DIR="/path/to/rft/data"                      # root directory for all training images/videos
EXP_NAME="Embodied-R1.5-RFT"                       # experiment name (used for checkpoint dir and logging)

# ==================== 2. Dataset ====================
# Full 26-dataset mix used for the official Embodied-R1.5 RFT run.
# Paths are relative to the EasyR1/ directory.
TRAIN_FILES="[rft_train_datasets/ER1.5_Droid-Trace_image_trace.json,\
rft_train_datasets/ER1.5_EO_image_qa.json,\
rft_train_datasets/ER1.5_ER1-point_image_point.json,\
rft_train_datasets/ER1.5_ER1-trace_image_trace.json,\
rft_train_datasets/ER1.5_ERQA2_image_qa.json,\
rft_train_datasets/ER1.5_ERQA_Rush_image_qa.json,\
rft_train_datasets/ER1.5_ERQA_Rush_image_qa.json,\
rft_train_datasets/ER1.5_ERQA_Rush_image_qa.json,\
rft_train_datasets/ER1.5_general_image_qa_filtered.json,\
rft_train_datasets/ER1.5_HandAL_image_point.json,\
rft_train_datasets/ER1.5_HOI4D-Trace_image_trace.json,\
rft_train_datasets/ER1.5_InstructPart_image_point.json,\
rft_train_datasets/ER1.5_InternData-Trace_image_trace.json,\
rft_train_datasets/ER1.5_Ref_L4_image_point.json,\
rft_train_datasets/ER1.5_Refspatial_image_point.json,\
rft_train_datasets/ER1.5_regular_simulation_image_point.json,\
rft_train_datasets/ER1.5_regular_synthetic_image_point.json,\
rft_train_datasets/ER1.5_Robo2VLM_image_qa.json,\
rft_train_datasets/ER1.5_robocasa_partnet_2d_image_trace.json,\
rft_train_datasets/ER1.5_robocasa_partnet_3d_image_trace.json,\
rft_train_datasets/ER1.5_Roborefit_image_point.json,\
rft_train_datasets/ER1.5_RoboVQA_image.json,\
rft_train_datasets/ER1.5_SAT_image_qa.json,\
rft_train_datasets/ER1.5_spatialssrl_image_qa.json,\
rft_train_datasets/ER1.5_CoSyn-point_image_point.json,\
rft_train_datasets/ER1.5_EmbSpatial_image_qa.json]"

TEST_FILES="[rft_test_datasets/erqa.json,\
rft_test_datasets/refspatial.json,\
rft_test_datasets/sat.json,\
rft_test_datasets/vabench_p.json,\
rft_test_datasets/where2place.json]"

# ==================== 3. Hyperparameters ====================
CONFIG=ER1.5_scripts/Embodied-R1.5_config.yaml
ROLLOUT_BS=1024
GEN_BS=256
GLOBAL_BS=512
MB_PER_UPDATE=4
MB_PER_EXP=8
REWARD=ER1.5_scripts/reward_function/embodied_reward.py:compute_score

# ==================== 4. Log Setup ====================
LOG_DIR="${PWD}/logs"
mkdir -p "${LOG_DIR}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/rft_${TIMESTAMP}.log"

echo "==================== Training Config ===================="
echo "Log File:    ${LOG_FILE}"
echo "Model Path:  ${MODEL_PATH}"
echo "Exp Name:    ${EXP_NAME}"
echo "Nodes*GPUs:  2 × 8 = 16"
echo "Time:        $(date)"
echo "========================================================"

python3 -m verl.trainer.main \
    config=${CONFIG} \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.clip_ratio_low=0.2 \
    worker.actor.clip_ratio_high=0.28 \
    worker.rollout.n=8 \
    algorithm.adv_estimator=mbpo \
    trainer.experiment_name=${EXP_NAME} \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=2 \
    trainer.val_freq=30 \
    trainer.save_freq=15 \
    trainer.save_limit=10 \
    trainer.total_epochs=2 \
    data.rollout_batch_size="${ROLLOUT_BS}" \
    data.mini_rollout_batch_size="${GEN_BS}" \
    data.train_files=$TRAIN_FILES \
    data.val_files=$TEST_FILES \
    data.image_dir=$IMAGE_DIR \
    worker.actor.global_batch_size="${GLOBAL_BS}" \
    worker.actor.micro_batch_size_per_device_for_update="${MB_PER_UPDATE}" \
    worker.actor.micro_batch_size_per_device_for_experience="${MB_PER_EXP}" \
    worker.rollout.tensor_parallel_size=4 \
    worker.reward.reward_function=${REWARD} 2>&1 | tee "${LOG_FILE}"
