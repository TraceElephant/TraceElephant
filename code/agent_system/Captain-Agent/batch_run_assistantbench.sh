#!/usr/bin/env bash
set -e

MODEL="gpt-4o"
START=x
END=x
REPEAT=3

for idx in $(seq $START $((END-1))); do
  for r in $(seq 1 $REPEAT); do
    echo "Running task-index $idx, round $r..."
    python scripts/run_assistantbench.py --mode build_and_run --task-index "$idx" --model "$MODEL" --max-round 30 || { echo "Task $idx round $r failed after retries, skipping to next."; continue; }
  done
done
