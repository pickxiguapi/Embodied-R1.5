#!/bin/bash
# =============================================================================
# Embodied-R1.5 SFT Training Script (4 nodes × 8 GPUs = 32 GPUs)
#
# Edit the variables in Section 1 and 2 before running.
# Then launch with: bash scripts/train/sft_train.sh
# =============================================================================

# ==================== 1. Node Configuration ====================
# IP of the head node used for NCCL communication (must be reachable from all nodes)
MASTER_ADDR="192.168.1.100"
MASTER_PORT="29582"

# SSH IPs for each node (used to dispatch training commands)
NODE0_SSH_IP="192.168.1.100"   # head node
NODE1_SSH_IP="192.168.1.101"   # worker 1
NODE2_SSH_IP="192.168.1.102"   # worker 2
NODE3_SSH_IP="192.168.1.103"   # worker 3

# ==================== 2. Training Configuration ====================
DATASET="Your Datasets"
WORK_DIR="/path/to/Embodied-R1.5"              # repo root on each node (must be the same path)
CONDA_ENV="llama"                              # conda environment with LLaMA-Factory installed
MODEL_PATH="Qwen/Qwen3-VL-8B-Instruct"         # HuggingFace model ID or local path
CONFIG_FILE="scripts/train/sft_config.yaml"    # training config
OUTPUT_DIR="/path/to/output/checkpoints"       # checkpoint output directory
NNODES=4
CURRENT_DATE=$(date +%Y%m%d)
RUN_NAME="Embodied-R1.5-SFT-${NNODES}node-$((NNODES * 8))gpu-${CURRENT_DATE}"
DATASET_DIR="dataset"
RESUME_FROM_CHECKPOINT=""  # leave empty to train from scratch; set to checkpoint path to resume

# ==================== 3. Experiment Tracking ====================
SWANLAB_API_KEY=""  # your SwanLab API key; leave empty to disable

# ==================== 4. Log Setup ====================
LOG_DIR="logs"
mkdir -p "${LOG_DIR}"

# ==================== 5. Signal Handler ====================
cleanup() {
    echo -e "\n\033[31mInterrupted — terminating background processes...\033[0m"
    kill $(jobs -p) 2>/dev/null
    exit
}
trap cleanup SIGINT SIGTERM

# ==================== 6. Node Launch Function ====================
run_node() {
    local NODE_RANK=$1
    local SSH_IP=$2
    local LOG_FILE="${LOG_DIR}/ER1.5_SFT_${CURRENT_DATE}_node_${NODE_RANK}.log"

    echo "🚀 [Node ${NODE_RANK}] Launching via SSH (${SSH_IP})..."

    # Build optional resume argument
    local RESUME_ARG=""
    if [ -n "${RESUME_FROM_CHECKPOINT}" ]; then
        RESUME_ARG="resume_from_checkpoint=${RESUME_FROM_CHECKPOINT}"
    fi

    # Build optional SwanLab API key export
    local SWANLAB_EXPORT=""
    if [ -n "${SWANLAB_API_KEY}" ]; then
        SWANLAB_EXPORT="export SWANLAB_API_KEY=${SWANLAB_API_KEY} &&"
    fi

    CMD="
    source ~/.bashrc && \
    conda activate ${CONDA_ENV} && \
    cd ${WORK_DIR} && \

    ${SWANLAB_EXPORT}

    export NCCL_IB_GID_INDEX=3 && \
    export NCCL_IB_SL=3 && \
    export NCCL_CHECK_DISABLE=1 && \
    export NCCL_P2P_DISABLE=0 && \
    export NCCL_IB_DISABLE=0 && \
    export NCCL_LL_THRESHOLD=16384 && \
    export NCCL_IB_CUDA_SUPPORT=1 && \
    export NCCL_SOCKET_IFNAME=bond1 && \
    export UCX_NET_DEVICES=bond1 && \
    export NCCL_IB_HCA=mlx5_bond_1,mlx5_bond_5,mlx5_bond_3,mlx5_bond_7,mlx5_bond_4,mlx5_bond_8,mlx5_bond_2,mlx5_bond_6 && \
    export NCCL_COLLNET_ENABLE=0 && \
    export SHARP_COLL_ENABLE_SAT=0 && \
    export NCCL_NET_GDR_LEVEL=2 && \
    export NCCL_IB_QPS_PER_CONNECTION=4 && \
    export NCCL_IB_TC=160 && \
    export NCCL_PXN_DISABLE=0 && \
    export NCCL_NVLS_ENABLE=0 && \
    export NCCL_PROFILE_PRIMS_ENABLE=1 && \
    export NCCL_DEBUG=INFO && \
    export NCCL_TIMEOUT=18000000 && \

    export FORCE_TORCHRUN=1 && \
    export NNODES=${NNODES} && \
    export NODE_RANK=${NODE_RANK} && \
    export MASTER_ADDR=${MASTER_ADDR} && \
    export MASTER_PORT=${MASTER_PORT} && \

    echo 'START' && \

    llamafactory-cli train ${CONFIG_FILE} \
        model_name_or_path=${MODEL_PATH} \
        output_dir=${OUTPUT_DIR} \
        dataset=${DATASET} \
        dataset_dir=${DATASET_DIR} \
        swanlab_run_name=${RUN_NAME} \
        ${RESUME_ARG}
    "

    ssh "${SSH_IP}" "${CMD}" > "${LOG_FILE}" 2>&1 &
}

# ==================== 6. Launch ====================
echo "=========================================="
echo "  Embodied-R1.5 SFT Training"
echo "  Nodes:       ${NNODES} × 8 GPUs = $((NNODES * 8)) GPUs total"
echo "  Master:      ${MASTER_ADDR}:${MASTER_PORT}"
echo "  Model:       ${MODEL_PATH}"
echo "  Output:      ${OUTPUT_DIR}"
echo "  Config:      ${CONFIG_FILE}"
if [ -n "${RESUME_FROM_CHECKPOINT}" ]; then
    echo "  Resuming from: ${RESUME_FROM_CHECKPOINT}"
else
    echo "  Training from scratch"
fi
echo "=========================================="

run_node 0 "${NODE0_SSH_IP}"
run_node 1 "${NODE1_SSH_IP}"
run_node 2 "${NODE2_SSH_IP}"
run_node 3 "${NODE3_SSH_IP}"

# ==================== 7. Log Monitoring ====================
echo "✅ Jobs submitted. Waiting for logs..."
sleep 4

echo "=========================================="
echo "📺 Tailing logs from all nodes..."
echo "👉 Press Ctrl+C to stop monitoring (training continues in background)"
echo "=========================================="

tail -f "${LOG_DIR}/ER1.5_SFT_${CURRENT_DATE}_node_0.log" \
        "${LOG_DIR}/ER1.5_SFT_${CURRENT_DATE}_node_1.log" \
        "${LOG_DIR}/ER1.5_SFT_${CURRENT_DATE}_node_2.log" \
        "${LOG_DIR}/ER1.5_SFT_${CURRENT_DATE}_node_3.log" &
wait
