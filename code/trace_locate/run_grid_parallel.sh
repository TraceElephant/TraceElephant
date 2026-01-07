#!/bin/bash

# One-click execution of 4 methods × 2 Ground Truth combinations (parallel inference + evaluation)
# Usage: ./run_grid_parallel.sh [options]
#
# Options:
#   --api-key KEY          LLM API Key (required)
#   --model MODEL          Model name (default: gpt-4o)
#   --base-url URL         API Base URL (default: https://api.xxx.com/v1)
#   --data-dir DIR         Data directory path (default: ../../data_dir)
#   --output-dir DIR       Output directory path (default: outputs)
#   --max-workers NUM      Number of parallel workers (default: 10)
#   --methods METHODS      Methods to run, comma-separated (default: all_at_once,step_by_step,binary_search)
#   --help                 Show this help message

# Default values
API_KEY=""
MODEL="gpt-4o"
BASE_URL="https://api.xxx.com/v1"
DATA_DIR="../../data_dir"
OUTPUT_DIR="outputs"
MAX_WORKERS="20"
# METHODS_STR="all_at_once,step_by_step,binary_search"
METHODS_STR="all_at_once"


# Show help message
show_help() {
    head -n 16 "$0" | tail -n 11
    exit 0
}

# Argument parsing
while [[ $# -gt 0 ]]; do
    case $1 in
        --api-key)
            API_KEY="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --base-url)
            BASE_URL="$2"
            shift 2
            ;;
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --max-workers)
            MAX_WORKERS="$2"
            shift 2
            ;;
        --methods)
            METHODS_STR="$2"
            shift 2
            ;;
        --help|-h)
            show_help
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Use --help to view the help message"
            exit 1
            ;;
    esac
done

# Check required arguments
if [ -z "$API_KEY" ]; then
    echo "Error: --api-key must be provided"
    echo "Use --help to view the help message"
    exit 1
fi

# Convert methods string to array
IFS=',' read -ra METHODS <<< "$METHODS_STR"

echo "========================================"
echo "Starting combined runs (parallel inference)"
echo "========================================"
echo "Model: $MODEL"
echo "BASE URL: $BASE_URL"
echo "Data directory: $DATA_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Parallel workers: $MAX_WORKERS"
echo "Methods: ${METHODS[*]}"
echo ""

printf "%-16s %-8s %-14s %-14s\n" "method" "agent_acc(%)" "step_acc(%)"
printf "%-16s %-8s %-14s %-14s\n" "------" "------------" "-----------"

for METHOD in "${METHODS[@]}"; do
    ./run_inference_parallel.sh "$METHOD" "$MAX_WORKERS" "$API_KEY" "$MODEL" "$DATA_DIR" "$OUTPUT_DIR" "$BASE_URL"
    if [ $? -ne 0 ]; then
      echo "Error: inference failed ($METHOD)"
      exit 1
    fi

    OUTPUT=$(./run_evaluation.sh "$METHOD" "$DATA_DIR" "$OUTPUT_DIR" "$MODEL")
    if [ $? -ne 0 ]; then
      echo "Error: evaluation failed ($METHOD)"
      exit 1
    fi

    AGENT_ACC=$(echo "$OUTPUT" | awk -F': ' '/Agent Accuracy:/ {print $2}' | tr -d '%')
    STEP_ACC=$(echo "$OUTPUT" | awk -F': ' '/Step Accuracy:/ {print $2}' | tr -d '%')

    printf "%-16s %-8s %-14s %-14s\n" "$METHOD" "${AGENT_ACC:-N/A}" "${STEP_ACC:-N/A}"
done

echo ""
echo "========================================"
echo "Combined runs completed"
echo "========================================"
