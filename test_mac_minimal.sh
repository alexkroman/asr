#!/bin/bash

# Set environment variables for Mac with minimal memory usage
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export TOKENIZERS_PARALLELISM=false
export MPS_ALLOCATOR_POLICY=default

# Run with minimal configuration
accelerate launch --config_file accelerate_configs/accelerate_config_mac.yaml train.py --config configs/mac_minimal.json