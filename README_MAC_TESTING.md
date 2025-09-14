# Testing ASR Training on Mac

This setup allows you to test the ASR training pipeline on your Mac with minimal resources.

## Configuration Files

### 1. `test_config.json`
A minimal configuration for testing that:
- Uses tiny model sizes (128 d_model, 2 layers, 4 attention heads)
- Uses the smallest SmolLM2 model (135M parameters)
- Trains on only 20 samples, validates on 10
- Runs for 1 epoch with batch size 2
- Saves checkpoints every 20 steps
- Evaluates every 10 steps

### 2. `accelerate_config_mac.yaml`
Accelerate configuration optimized for Mac:
- Single machine, single process
- No distributed training
- CPU/MPS backend (will use MPS if available on M1/M2 Macs)

## Quick Start

### Option 1: Using the Test Script
```bash
./test_mac.sh
```

This script will:
- Check and install dependencies if needed
- Set up environment variables for Mac
- Create necessary directories
- Run the training with test configuration

### Option 2: Manual Run

1. Install dependencies:
```bash
pip install torch transformers datasets evaluate einops peft torchaudio accelerate tensorboard
```

2. Set environment variables:
```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
export TOKENIZERS_PARALLELISM=false
```

3. Run training:
```bash
accelerate launch --config_file accelerate_config_mac.yaml train.py --config test_config.json
```

## Monitoring Progress

1. **Console Output**: Watch the terminal for training progress
2. **TensorBoard**:
   ```bash
   tensorboard --logdir ./test_logs
   ```
   Then open http://localhost:6006 in your browser

## Expected Performance

On a Mac (M1/M2 or Intel):
- Training should complete in 5-10 minutes
- Memory usage should stay under 4GB
- The model won't converge (it's too small) but should run without errors

## Customization

To adjust for your Mac's capabilities:

### For M1/M2 Macs with more memory:
- Increase `per_device_train_batch_size` to 4 or 8
- Increase `d_model` to 256 or 512
- Increase `num_layers` to 4 or 6

### For faster testing:
- Reduce `train_split` to `"train.100[:10]"`
- Reduce `eval_split` to `"validation[:5]"`

### For better model performance (longer training):
- Use `"HuggingFaceTB/SmolLM2-360M-Instruct"` as decoder
- Increase `num_train_epochs` to 3-5
- Increase dataset size by using `"train.100[:100]"`

## Troubleshooting

### If you get CUDA errors:
The config is set to use CPU/MPS. Make sure you're using the correct accelerate config.

### If you get memory errors:
- Reduce `per_device_train_batch_size` to 1
- Reduce `d_model` to 64
- Enable `gradient_checkpointing: true` in config

### If datasets download is slow:
The first run will download the LibriSpeech dataset. This can take a few minutes. The data is cached in `~/datasets` for future runs.

## Cleanup

To remove test outputs:
```bash
rm -rf test_output test_logs ~/datasets
```