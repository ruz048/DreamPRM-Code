#!/bin/bash

# ============================================================================
# PRM Training with Bilevel Optimization (Data Reweighting) + Evaluation
# Using Distributed Data Parallel (DDP) with Betty Framework
# ============================================================================
# This script trains a PRM model using bilevel optimization with Betty framework
# for automatic data reweighting based on meta-learning, then evaluates it.
# Supports multi-GPU distributed training via torchrun.
# ============================================================================

set -e  # Exit on error

# ============================================================================
# DDP Configuration
# ============================================================================

# Number of GPUs to use (adjust based on your available GPUs)
NUM_GPUS=1

# Specify which GPUs to use (comma-separated list)
export CUDA_VISIBLE_DEVICES=3

# Master address and port for distributed training
export MASTER_ADDR=localhost
export MASTER_PORT=29500

# ============================================================================
# Configuration
# ============================================================================

# Model configuration
MODEL_NAME="Qwen/Qwen2.5-Coder-3B"  # Options: Qwen/Qwen2.5-Coder-1.5B, Qwen/Qwen2.5-Coder-7B, etc.

# Data paths
TRAIN_DATA="YOUR TRAIN DATA"
META_DATA="YOUR META DATA"
EVAL_DATA="YOUR EVAL DATA"

# Output paths
OUTPUT_DIR="checkpoints_blo_ddp/$(date +%Y%m%d_%H%M%S)"
OUTPUT_JSON="prm-results/prm_blo_ddp_results_$(date +%Y%m%d_%H%M%S).json"

# Training hyperparameters (optimized for DDP training)
BATCH_SIZE=1  # Per-GPU batch size (effective batch size = BATCH_SIZE * NUM_GPUS * GRADIENT_ACCUM_STEPS)
GRADIENT_ACCUM_STEPS=8  # Gradient accumulation steps
LEARNING_RATE=1e-4  # Very low LR to prevent overfitting
WEIGHT_DECAY=0.01  # L2 regularization to prevent overfitting
META_LR=1e-2  # Low meta LR to prevent overfitting
META_WEIGHT_DECAY=0.002  # Weight decay for meta-network
DROPOUT=0.01  # Dropout for regularization
GRAD_CLIP=1.0  # Gradient clipping to prevent exploding gradients
TEXT_MAX_LEN=2048

# Bilevel optimization hyperparameters
TRAIN_ITERS=10000  # Training iterations
VALID_STEP=1000  # Validation frequency
UNROLL_STEPS=1  # Unroll steps for implicit differentiation

# Warm-up stage hyperparameters (pre-BLO training on full dataset)
WARMUP_EPOCHS=5  # Number of warm-up epochs before BLO (0 to disable)
WARMUP_BATCH_SIZE=1  # Batch size per GPU for warm-up stage
WARMUP_GRADIENT_ACCUM_STEPS=32  # Gradient accumulation for warm-up
WARMUP_LR=1e-4  # Learning rate for warm-up stage

# Loss function: use --use_bce flag for BCEWithLogits (scores in [0,1]), omit for MSE
USE_BCE_FLAG=""  # Set to "--use_bce" if needed

# Training mode
BASELINE_FLAG=""  # Set to "--baseline" to disable meta-learning (for comparison)

# Precision and strategy
PRECISION="fp32"  # Options: fp32, fp16, bf16
STRATEGY="distributed"  # Use "distributed" for DDP, "zero" for ZeRO, "fsdp" for FSDP (experimental)

# Random seed
SEED=42

# LoRA Configuration
LORA_R=8
LORA_ALPHA=16
LORA_DROPOUT=0.05
LORA_TARGET_MODULES="q_proj,v_proj,k_proj,o_proj"

# Evaluation configuration
USE_PREFIX_EVAL_FLAG=""  # Use prefix-based evaluation (typical PRM)
PREFIX_AGGREGATION="last"  # Options: min, mean, last, product

# Warm-up output directory
WARMUP_DIR="${OUTPUT_DIR}_warmup"
WARMUP_CHECKPOINT="${WARMUP_DIR}/warmup_checkpoint.pt"

# ============================================================================
# Step 1: Warm-up Training (if enabled)
# ============================================================================

if [ ${WARMUP_EPOCHS} -gt 0 ]; then
    echo "============================================================================"
    echo "Starting Warm-up Training"
    echo "============================================================================"
    echo "Model: ${MODEL_NAME}"
    echo "Training data: ${TRAIN_DATA}"
    echo "Warmup output directory: ${WARMUP_DIR}"
    echo "Number of GPUs: ${NUM_GPUS}"
    echo "GPUs used: ${CUDA_VISIBLE_DEVICES}"
    echo "============================================================================"
    echo ""
    echo "Warm-up Hyperparameters:"
    echo "  - Epochs: ${WARMUP_EPOCHS}"
    echo "  - Batch size per GPU: ${WARMUP_BATCH_SIZE}"
    echo "  - Gradient accumulation: ${WARMUP_GRADIENT_ACCUM_STEPS}"
    echo "  - Effective batch size: $((WARMUP_BATCH_SIZE * NUM_GPUS * WARMUP_GRADIENT_ACCUM_STEPS))"
    echo "  - Learning rate: ${WARMUP_LR}"
    echo "  - LoRA rank: ${LORA_R}"
    echo "  - LoRA alpha: ${LORA_ALPHA}"
    echo "============================================================================"
    echo ""

    # Launch distributed warm-up training using torchrun
    torchrun \
        --nproc_per_node=${NUM_GPUS} \
        --nnodes=1 \
        --node_rank=0 \
        --master_addr=${MASTER_ADDR} \
        --master_port=${MASTER_PORT} \
        train_prm_warmup.py \
        --model_name "${MODEL_NAME}" \
        --train_data "${TRAIN_DATA}" \
        --meta_data "${META_DATA}" \
        --output_dir "${WARMUP_DIR}" \
        --text_max_len ${TEXT_MAX_LEN} \
        --batch_size ${WARMUP_BATCH_SIZE} \
        --gradient_accumulation_steps ${WARMUP_GRADIENT_ACCUM_STEPS} \
        --learning_rate ${WARMUP_LR} \
        --weight_decay ${WEIGHT_DECAY} \
        --dropout ${DROPOUT} \
        --grad_clip ${GRAD_CLIP} \
        --epochs ${WARMUP_EPOCHS} \
        --seed ${SEED} \
        --lora_r ${LORA_R} \
        --lora_alpha ${LORA_ALPHA} \
        --lora_dropout ${LORA_DROPOUT} \
        --lora_target_modules ${LORA_TARGET_MODULES} \
        ${USE_BCE_FLAG}

    echo ""
    echo "============================================================================"
    echo "Warm-up training completed successfully!"
    echo "============================================================================"
    echo "Checkpoint saved to: ${WARMUP_CHECKPOINT}"
    echo "============================================================================"
    echo ""

    # Wait for GPU memory to be released before BLO training
    echo "Waiting for GPU memory to be released before BLO training..."
    sleep 10

    # Function to check if GPUs are free
    wait_for_gpu_memory_warmup() {
        local max_wait=300  # Maximum wait time in seconds (5 minutes)
        local wait_time=0
        local check_interval=5

        while [ $wait_time -lt $max_wait ]; do
            # Get GPU memory usage for the specified GPUs
            local gpu_busy=0
            for gpu_id in $(echo $CUDA_VISIBLE_DEVICES | tr ',' ' '); do
                local mem_used=$(nvidia-smi -i $gpu_id --query-gpu=memory.used --format=csv,noheader,nounits)
                # If memory used is more than 1000 MB, consider GPU busy
                if [ "$mem_used" -gt 1000 ]; then
                    gpu_busy=1
                    break
                fi
            done

            if [ $gpu_busy -eq 0 ]; then
                echo "GPUs are free. Proceeding with BLO training..."
                return 0
            fi

            echo "GPUs still in use. Waiting... (${wait_time}s/${max_wait}s)"
            sleep $check_interval
            wait_time=$((wait_time + check_interval))
        done

        echo "WARNING: Timeout waiting for GPU memory to be released. Proceeding anyway..."
        return 1
    }

    # Wait for GPUs to be free
    wait_for_gpu_memory_warmup
    echo ""
else
    echo "============================================================================"
    echo "Skipping warm-up training (WARMUP_EPOCHS=0)"
    echo "============================================================================"
    echo ""
fi

# ============================================================================
# Step 2: Training with Bilevel Optimization using DDP
# ============================================================================

echo "============================================================================"
echo "Starting PRM Training with Bilevel Optimization (DDP)"
echo "============================================================================"
echo "Model: ${MODEL_NAME}"
echo "Training data: ${TRAIN_DATA}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Training mode: $([ -z "${BASELINE_FLAG}" ] && echo 'Meta-learning (BLO)' || echo 'Baseline (no BLO)')"
echo "Distributed strategy: ${STRATEGY}"
echo "Number of GPUs: ${NUM_GPUS}"
echo "GPUs used: ${CUDA_VISIBLE_DEVICES}"
echo "============================================================================"
echo ""
echo "Hyperparameters:"
echo "  - Batch size per GPU: ${BATCH_SIZE}"
echo "  - Number of GPUs: ${NUM_GPUS}"
echo "  - Effective batch size: $((BATCH_SIZE * NUM_GPUS * GRADIENT_ACCUM_STEPS))"
echo "  - Gradient accumulation steps: ${GRADIENT_ACCUM_STEPS}"
echo "  - Learning rate: ${LEARNING_RATE}"
echo "  - Weight decay: ${WEIGHT_DECAY}"
echo "  - Dropout: ${DROPOUT}"
echo "  - Gradient clipping: ${GRAD_CLIP}"
echo "  - Meta learning rate: ${META_LR}"
echo "  - Training iterations: ${TRAIN_ITERS}"
echo "  - Validation step: ${VALID_STEP}"
echo "  - Unroll steps: ${UNROLL_STEPS}"
echo "  - LoRA rank: ${LORA_R}"
echo "  - LoRA alpha: ${LORA_ALPHA}"
echo "============================================================================"
echo ""

# Launch distributed training using torchrun
torchrun \
    --nproc_per_node=${NUM_GPUS} \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    train_prm_blo.py \
    --model_name "${MODEL_NAME}" \
    --train_data "${TRAIN_DATA}" \
    --meta_data "${META_DATA}" \
    --output_dir "${OUTPUT_DIR}" \
    --text_max_len ${TEXT_MAX_LEN} \
    --batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACCUM_STEPS} \
    --learning_rate ${LEARNING_RATE} \
    --weight_decay ${WEIGHT_DECAY} \
    --meta_lr ${META_LR} \
    --meta_weight_decay ${META_WEIGHT_DECAY} \
    --dropout ${DROPOUT} \
    --grad_clip ${GRAD_CLIP} \
    --train_iters ${TRAIN_ITERS} \
    --valid_step ${VALID_STEP} \
    --unroll_steps ${UNROLL_STEPS} \
    --precision ${PRECISION} \
    --strategy ${STRATEGY} \
    --seed ${SEED} \
    --lora_r ${LORA_R} \
    --lora_alpha ${LORA_ALPHA} \
    --lora_dropout ${LORA_DROPOUT} \
    --lora_target_modules ${LORA_TARGET_MODULES} \
    $([ ${WARMUP_EPOCHS} -gt 0 ] && echo "--warmup_checkpoint ${WARMUP_CHECKPOINT}") \
    ${USE_BCE_FLAG} \
    ${BASELINE_FLAG}

echo ""
echo "============================================================================"
echo "Training completed successfully!"
echo "============================================================================"
echo "Checkpoint directory: ${OUTPUT_DIR}"
echo "============================================================================"
echo ""

# Wait for GPU memory to be released
echo "Waiting for GPU memory to be released..."
sleep 10

# Function to check if GPUs are free
wait_for_gpu_memory() {
    local max_wait=300  # Maximum wait time in seconds (5 minutes)
    local wait_time=0
    local check_interval=5

    while [ $wait_time -lt $max_wait ]; do
        # Get GPU memory usage for the specified GPUs
        local gpu_busy=0
        for gpu_id in $(echo $CUDA_VISIBLE_DEVICES | tr ',' ' '); do
            local mem_used=$(nvidia-smi -i $gpu_id --query-gpu=memory.used --format=csv,noheader,nounits)
            # If memory used is more than 1000 MB, consider GPU busy
            if [ "$mem_used" -gt 1000 ]; then
                gpu_busy=1
                break
            fi
        done

        if [ $gpu_busy -eq 0 ]; then
            echo "GPUs are free. Proceeding with evaluation..."
            return 0
        fi

        echo "GPUs still in use. Waiting... (${wait_time}s/${max_wait}s)"
        sleep $check_interval
        wait_time=$((wait_time + check_interval))
    done

    echo "WARNING: Timeout waiting for GPU memory to be released. Proceeding anyway..."
    return 1
}

# Wait for GPUs to be free
wait_for_gpu_memory

# ============================================================================
# Step 3: Evaluation
# ============================================================================

echo ""
echo "============================================================================"
echo "Starting PRM Evaluation"
echo "============================================================================"
echo "Model: ${MODEL_NAME}"
echo "Checkpoint directory: ${OUTPUT_DIR}"
echo "Evaluation data: ${EVAL_DATA}"
echo "Output JSON: ${OUTPUT_JSON}"
echo "============================================================================"
echo ""

# Find the best checkpoint (the first one saved, which has the best validation loss)
FIRST_CHECKPOINT=$(find "${OUTPUT_DIR}" -name "reward_model_step_*.pt" -type f | sort -V | head -n 1)

if [ -n "${FIRST_CHECKPOINT}" ]; then
    CHECKPOINT_FILE="${FIRST_CHECKPOINT}"
    # Extract step number from filename
    BEST_STEP=$(echo "${CHECKPOINT_FILE}" | grep -oP 'step_\K[0-9]+')
    echo "Using best checkpoint: ${CHECKPOINT_FILE} (step ${BEST_STEP})"
else
    echo "============================================================================"
    echo "ERROR: No checkpoints found in ${OUTPUT_DIR}"
    echo "============================================================================"
    echo "Available files:"
    ls -lh "${OUTPUT_DIR}" 2>/dev/null || echo "  (directory is empty or not accessible)"
    echo ""
    echo "Training completed but evaluation skipped."
    echo "Please check training logs for errors."
    echo "============================================================================"
    exit 1
fi

echo ""
echo "Running evaluation..."
echo ""

# Run evaluation (on single GPU)
python evaluate_prm_blo.py \
    --model_name "${MODEL_NAME}" \
    --checkpoint_dir "${OUTPUT_DIR}" \
    --checkpoint_step ${BEST_STEP} \
    --eval_data "${EVAL_DATA}" \
    --output_json "${OUTPUT_JSON}" \
    --text_max_len ${TEXT_MAX_LEN} \
    --dropout ${DROPOUT} \
    --batch_size 8 \
    ${USE_BCE_FLAG} \
    ${USE_PREFIX_EVAL_FLAG} \
    --prefix_aggregation ${PREFIX_AGGREGATION} \
    --lora_r ${LORA_R} 

EVAL_EXIT_CODE=$?

echo ""
if [ ${EVAL_EXIT_CODE} -eq 0 ]; then
    echo "============================================================================"
    echo "Pipeline completed successfully!"
    echo "============================================================================"
    echo "Training checkpoint: ${OUTPUT_DIR}"
    echo "Best checkpoint step: ${BEST_STEP}"
    echo "Evaluation results: ${OUTPUT_JSON}"
    echo "============================================================================"

    # Display results summary if jq is available
    if command -v jq &> /dev/null && [ -f "${OUTPUT_JSON}" ]; then
        echo ""
        echo "Evaluation Summary:"
        echo "============================================================================"
        jq '.summary' "${OUTPUT_JSON}" 2>/dev/null || echo "Unable to parse results"
        echo "============================================================================"
    fi
else
    echo "============================================================================"
    echo "Evaluation failed with exit code: ${EVAL_EXIT_CODE}"
    echo "============================================================================"
    echo "Training checkpoint: ${OUTPUT_DIR}"
    echo "Please check evaluation logs for errors."
    echo "============================================================================"
    exit ${EVAL_EXIT_CODE}
fi

echo ""
