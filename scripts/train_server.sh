#!/bin/bash
# Training server deployment script

set -e

# Configuration
EXPERIMENT=${1:-production}
GPU_IDS=${GPU_IDS:-0,1,2,3}
BATCH_SIZE=${BATCH_SIZE:-8}

echo "🚀 Starting training on server"
echo "Experiment: $EXPERIMENT"
echo "GPUs: $GPU_IDS"

# Option 1: Using Hatch (if installed)
if command -v hatch &> /dev/null; then
    echo "Using Hatch environment..."
    CUDA_VISIBLE_DEVICES=$GPU_IDS hatch run cuda:train-production
    exit 0
fi

# Option 2: Using Conda
if command -v conda &> /dev/null; then
    echo "Using Conda environment..."
    conda activate asr || conda create -n asr python=3.11 -y && conda activate asr
    pip install -e .
    CUDA_VISIBLE_DEVICES=$GPU_IDS accelerate launch \
        --num_processes=$(echo $GPU_IDS | tr ',' '\n' | wc -l) \
        --multi_gpu \
        src/train.py +experiments=$EXPERIMENT
    exit 0
fi

# Option 3: Using venv
echo "Using Python venv..."
python -m venv venv
source venv/bin/activate
pip install -e .
CUDA_VISIBLE_DEVICES=$GPU_IDS python src/train.py +experiments=$EXPERIMENT