#!/bin/bash

# Run the inference script (parallel mode)
# Use multithreading for parallel processing to significantly improve speed

# Argument parsing
METHOD="${1:-all_at_once}"
MAX_WORKERS="${2:-10}"
API_KEY="${3:-sk-xxx}"
MODEL="${4:-gpt-4o-mini}"
DATA_PATH="${5:-../../data}"
OUTPUT_DIR="${6:-outputs}"
BASE_URL="${7:-https://api.xxx.com/v1}"



echo "========================================="
echo "Running inference (parallel mode)"
echo "========================================="
echo "Method: $METHOD"
echo "Model: $MODEL"
echo "BASE URL: $BASE_URL"
echo "Data path: $DATA_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "Number of parallel workers: $MAX_WORKERS"
echo "========================================="
echo ""

# Create output directory (if it does not exist)
mkdir -p "$OUTPUT_DIR"

# Run inference (parallel mode, now always parallel)
python inference.py \
    --method "$METHOD" \
    --model "$MODEL" \
    --directory_path "$DATA_PATH" \
    --api_key "$API_KEY" \
    --base_url "$BASE_URL" \
    --max_tokens 1024 \
    --max_workers "$MAX_WORKERS" \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "========================================="
echo "Inference completed!"
echo "Output file: ${OUTPUT_DIR}/${METHOD}_${MODEL}.jsonl"
echo "========================================="
