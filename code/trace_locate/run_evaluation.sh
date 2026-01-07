#!/bin/bash

# Run the evaluation script
# Evaluate the accuracy of inference results

# Argument parsing
METHOD="${1:-all_at_once}"
DATA_PATH="${2:-../../data}"
OUTPUT_DIR="${3:-outputs}"
MODEL="${4:-gpt-4o}"
 
# Evaluation file
EVAL_FILE="${OUTPUT_DIR}/${METHOD}_${MODEL}.jsonl"

echo "========================================="
echo "Running Evaluation"
echo "========================================="
echo "Method: $METHOD"
echo "Model: $MODEL"
echo "Data Path: $DATA_PATH"
echo "Output Directory: $OUTPUT_DIR"
echo "Evaluation File: $EVAL_FILE"
echo "========================================="
echo ""

# Check whether the evaluation file exists
if [ ! -f "$EVAL_FILE" ]; then
    echo "Error: Evaluation file does not exist: $EVAL_FILE"
    echo "Please run inference first"
    exit 1
fi

# Run evaluation
python evaluate.py \
    --data_path "$DATA_PATH" \
    --eval_file "$EVAL_FILE"

echo ""
echo "========================================="
echo "Evaluation completed!"
echo "========================================="
