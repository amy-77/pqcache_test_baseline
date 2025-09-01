#!/bin/bash

# SCBench Generic Evaluation Script for PQCache
# This script tests different task categories to analyze attention drift patterns

# Set environment variables
export CUDA_VISIBLE_DEVICES=4
export SUBVEC=2
export SUBBITS=4
export METRIC="euc"

# PQCache parameters
COMPRESS_RATIO=0.1
IMPORTANT_RATIO=0.5
RECENT_RATIO=0.5
SUBVEC=2
SUBBITS=4
DEVICE_ID=4
PP_SIZE=1

# Model and data parameters
MODEL_NAME_OR_PATH="/home/pai/data/model/Llama-3.1-8B-Instruct"
MAX_SEQ_LENGTH=131072
NUM_EVAL_EXAMPLES=2  # Start with small number for testing
MAX_TURNS=-1  # -1 means use all turns from dataset
OUTPUT_DIR="pred_generic"

# Task categories to test
TASK_CATEGORIES=(
    "long_generation_decode_shift"  # En.Sum, Mix.Sum+NIAH
    "strong_drift"                  # En.QA, Zh.QA, En.MultiChoice, Math.Find
    "multiturn_kv_drift"           # Retr.KV, Retr.Prefix-Suffix, Retr.MultiHop
)

# Function to run a single task category
run_task_category() {
    local category=$1
    echo "=================================="
    echo "Running task category: $category"
    echo "=================================="
    
    python vq_pred_scbench_generic.py \
        --model "llama-3.1" \
        --task_category "$category" \
        --num_eval_examples $NUM_EVAL_EXAMPLES \
        --max_turns $MAX_TURNS \
        --max_seq_length $MAX_SEQ_LENGTH \
        --compress_ratio $COMPRESS_RATIO \
        --important_ratio $IMPORTANT_RATIO \
        --recent_ratio $RECENT_RATIO \
        --n_subvec_per_head $SUBVEC \
        --n_subbits $SUBBITS \
        --pp-size $PP_SIZE \
        --device_id $DEVICE_ID \
        --output_dir "$OUTPUT_DIR" \
        --enable_vq_cache \
        --verbose
    
    if [ $? -eq 0 ]; then
        echo "Task category $category completed successfully"
    else
        echo "Task category $category failed"
        return 1
    fi
}

# Function to run a single task
run_single_task() {
    local task=$1
    echo "=================================="
    echo "Running single task: $task"
    echo "=================================="
    
    python vq_pred_scbench_generic.py \
        --model "llama-3.1" \
        --task "$task" \
        --num_eval_examples $NUM_EVAL_EXAMPLES \
        --max_turns $MAX_TURNS \
        --max_seq_length $MAX_SEQ_LENGTH \
        --compress_ratio $COMPRESS_RATIO \
        --important_ratio $IMPORTANT_RATIO \
        --recent_ratio $RECENT_RATIO \
        --n_subvec_per_head $SUBVEC \
        --n_subbits $SUBBITS \
        --pp-size $PP_SIZE \
        --device_id $DEVICE_ID \
        --output_dir "$OUTPUT_DIR" \
        --enable_vq_cache \
        --verbose
}

# Main execution
echo "Starting SCBench Generic evaluation for PQCache"
echo "Model: $MODEL_NAME_OR_PATH"
echo "Device: CUDA:$DEVICE_ID"
echo "Compress ratio: $COMPRESS_RATIO"
echo "Important ratio: $IMPORTANT_RATIO"
echo "Recent ratio: $RECENT_RATIO"
echo "Subvec: $SUBVEC, Subbits: $SUBBITS"
echo "Max examples: $NUM_EVAL_EXAMPLES"

# Check if user specified a specific task category or task
if [ $# -eq 1 ]; then
    case $1 in
        "long_generation_decode_shift"|"strong_drift"|"multiturn_kv_drift")
            echo "Running specific task category: $1"
            run_task_category $1
            ;;
        "scbench_"*)
            echo "Running specific task: $1"
            run_single_task $1
            ;;
        "test")
            echo "Running test with scbench_summary..."
            run_single_task "scbench_summary"
            ;;
        *)
            echo "Unknown task or category: $1"
            echo "Available categories: ${TASK_CATEGORIES[@]}"
            echo "Available tasks: scbench_summary, scbench_qa_eng, scbench_kv, etc."
            exit 1
            ;;
    esac
else
    echo "Running all task categories..."
    
    # Run each category
    for category in "${TASK_CATEGORIES[@]}"; do
        echo ""
        echo "Starting category: $category"
        run_task_category "$category"
        
        if [ $? -ne 0 ]; then
            echo "Failed on category $category, but continuing..."
        fi
        
        # Give some time between categories
        sleep 5
    done
fi

echo ""
echo "SCBench Generic evaluation completed!"
echo "Results saved in: $OUTPUT_DIR/"
