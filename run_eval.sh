#!/bin/bash

# ============================================================================
# BLO PRM Model Evaluation Script
# ============================================================================
# This script evaluates a BLO-trained PRM model on test data
# ============================================================================

set -e  # Exit on error

# Use only one GPU to avoid device conflicts
export CUDA_VISIBLE_DEVICES=3

# Create logs directory if it doesn't exist
mkdir -p logs

# Generate timestamp for log file
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/eval_blo_${TIMESTAMP}.log"

# ============================================================================
# Configuration
# ============================================================================

# Model configuration
MODEL_NAME="Qwen/Qwen2.5-Coder-3B"  # Must match training

# Checkpoint configuration
CHECKPOINT_DIR="YOUR CKPT DIR"  # UPDATE THIS
CHECKPOINT_STEP=10000  # Which step to evaluate 

# Data paths
EVAL_DATA="YOUR EVAL DATA"

# Output path
OUTPUT_JSON="prm-results/prm_blo_results_$(date +%Y%m%d_%H%M%S).json"

# Model configuration (must match training)
TEXT_MAX_LEN=2048
DROPOUT=0.01  # Must match training

# LoRA configuration (MUST match training configuration exactly!)
# Check your training script or logs to verify these values
LORA_R=8  # LoRA rank (default from training script)
LORA_ALPHA=16  # LoRA alpha
LORA_DROPOUT=0.05  # LoRA dropout
LORA_TARGET_MODULES="q_proj,v_proj,k_proj,o_proj"  # Default from training script

# Loss function: use --use_bce flag for BCE mode (must match training)
USE_BCE_FLAG=""  # Set to "--use_bce" if needed

# Evaluation configuration
USE_PREFIX_EVAL_FLAG="--use_prefix_eval"  # Use prefix-based evaluation (typical PRM)
USE_PREFIX_EVAL_FLAG=""
PREFIX_AGGREGATION="last"  # Options: min, mean, last, product

# Batch size
BATCH_SIZE=8

# ============================================================================
# Validation
# ============================================================================

if [ ! -d "${CHECKPOINT_DIR}" ]; then
    echo "Error: Checkpoint directory not found: ${CHECKPOINT_DIR}"
    echo "Please update CHECKPOINT_DIR in this script"
    exit 1
fi

if [ ! -f "${EVAL_DATA}" ]; then
    echo "Error: Evaluation data not found: ${EVAL_DATA}"
    exit 1
fi

# Check if checkpoint exists
CHECKPOINT_FILE="${CHECKPOINT_DIR}/reward_model_step_${CHECKPOINT_STEP}.pt"
if [ ! -f "${CHECKPOINT_FILE}" ]; then
    echo "Error: Checkpoint file not found: ${CHECKPOINT_FILE}"
    echo ""
    echo "Available checkpoints in ${CHECKPOINT_DIR}:"
    ls -lh "${CHECKPOINT_DIR}"/reward_model_step_*.pt 2>/dev/null || echo "  (none found)"
    echo ""
    echo "Please update CHECKPOINT_STEP in this script"
    exit 1
fi

# ============================================================================
# Evaluation
# ============================================================================

echo "============================================================================"
echo "BLO PRM Model Evaluation"
echo "============================================================================"
echo "Starting evaluation in background..."
echo "Log file: ${LOG_FILE}"
echo "Use 'tail -f ${LOG_FILE}' to monitor progress"
echo "============================================================================"
echo ""

# Run with nohup and redirect all output to log file
nohup python evaluate_prm_blo.py \
    --model_name "${MODEL_NAME}" \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --checkpoint_step ${CHECKPOINT_STEP} \
    --eval_data "${EVAL_DATA}" \
    --output_json "${OUTPUT_JSON}" \
    --text_max_len ${TEXT_MAX_LEN} \
    --dropout ${DROPOUT} \
    --batch_size ${BATCH_SIZE} \
    --lora_r ${LORA_R} \
    --lora_alpha ${LORA_ALPHA} \
    --lora_dropout ${LORA_DROPOUT} \
    --lora_target_modules "${LORA_TARGET_MODULES}" \
    ${USE_BCE_FLAG} \
    ${USE_PREFIX_EVAL_FLAG} \
    --prefix_aggregation ${PREFIX_AGGREGATION} \
    > "${LOG_FILE}" 2>&1 &

# Save the PID
EVAL_PID=$!
echo "Evaluation process started with PID: ${EVAL_PID}"
echo ""
echo "To monitor progress:"
echo "  tail -f ${LOG_FILE}"
echo ""
echo "To check if still running:"
echo "  ps -p ${EVAL_PID}"
