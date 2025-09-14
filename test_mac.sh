#!/bin/bash

echo "🚀 Testing ASR training on Mac..."
echo "================================"

# Check if accelerate is installed
if ! command -v accelerate &> /dev/null; then
    echo "❌ Accelerate not found. Installing..."
    pip install accelerate
fi

# Check if required packages are installed
echo "📦 Checking dependencies..."
python -c "import torch; import transformers; import datasets; import evaluate; import einops; import peft; import torchaudio" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  Some dependencies are missing. Installing..."
    pip install torch transformers datasets evaluate einops peft torchaudio
fi

# Set environment variables for Mac
export PYTORCH_ENABLE_MPS_FALLBACK=1
export TOKENIZERS_PARALLELISM=false

# Create necessary directories
mkdir -p test_output test_logs ~/datasets

echo ""
echo "🏃 Running training with test configuration..."
echo "----------------------------------------------"

# Run with accelerate using the Mac config
accelerate launch --config_file accelerate_config_mac.yaml train.py --config test_config.json

echo ""
echo "✅ Test complete!"
echo ""
echo "📊 Check the following for results:"
echo "   - Training logs: ./test_logs/"
echo "   - Model output: ./test_output/"
echo "   - TensorBoard: tensorboard --logdir ./test_logs"