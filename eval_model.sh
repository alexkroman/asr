#!/bin/bash

# Script to evaluate a saved model

# Set environment variables for Mac
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export TOKENIZERS_PARALLELISM=false

echo "📊 Evaluating saved model..."
echo "================================"

# Run evaluation with the small test config
accelerate launch --config_file accelerate_configs/accelerate_config_mac.yaml \
    train.py --eval-only --config configs/small_test.json