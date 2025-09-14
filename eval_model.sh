#!/bin/bash

# Script to evaluate a saved model
# Usage: ./eval_model.sh [config_file]
# Example: ./eval_model.sh configs/test_config.json

# Set environment variables for Mac
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export TOKENIZERS_PARALLELISM=false

# Use provided config or default to test_config.json
CONFIG_FILE=${1:-configs/test_config.json}

echo "📊 Evaluating saved model..."
echo "================================"
echo "Using config: $CONFIG_FILE"
echo ""

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Config file not found: $CONFIG_FILE"
    exit 1
fi

# Run evaluation with the specified config
accelerate launch --config_file accelerate_configs/accelerate_config_mac.yaml \
    train.py --eval-only --config "$CONFIG_FILE"