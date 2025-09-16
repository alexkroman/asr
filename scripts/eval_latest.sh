#!/bin/bash

# Script to evaluate the latest checkpoint in a directory
# Usage: ./scripts/eval_latest.sh <output_dir> <config_file>

OUTPUT_DIR=${1:-"./production_output"}
CONFIG_FILE=${2:-"configs/experiments/production.json"}

# Find the latest checkpoint directory
LATEST_CHECKPOINT=$(find "$OUTPUT_DIR" -name "checkpoint-*" -type d | sort -V | tail -1)

if [ -z "$LATEST_CHECKPOINT" ]; then
    echo "Error: No checkpoint found in $OUTPUT_DIR"
    exit 1
fi

echo "Evaluating latest checkpoint: $LATEST_CHECKPOINT"

# Run evaluation
python src/train.py --eval --checkpoint "$LATEST_CHECKPOINT" --config "$CONFIG_FILE"