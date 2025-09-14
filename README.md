# 🎙️ Conformer-SmolLM2 ASR Training

A production-ready Automatic Speech Recognition (ASR) training pipeline combining Conformer encoder with SmolLM2 decoder, powered by Hugging Face Transformers and Accelerate.

## Features

- **🚀 Hardware Agnostic**: Automatic optimization for any GPU (A40, A100, H100, etc.) via Accelerate
- **📊 Flexible Configuration**: Dataclass-based config with JSON support and command-line overrides
- **🔧 Production Ready**: LoRA fine-tuning, gradient checkpointing, mixed precision training
- **📈 Multi-GPU Support**: Seamless distributed training with a single command
- **🤗 Hub Integration**: Direct push to Hugging Face Hub
- **📉 WandB Logging**: Automatic experiment tracking

## Quick Start

### Installation

```bash
# Install Hatch (if not already installed)
pip install hatch

# Install the project and dependencies
hatch env create

# Or install directly with pip
pip install .

# For development:
pip install -e ".[dev]"

# With optimizations:
pip install -e ".[optimized]"

# For CUDA support:
pip install -e ".[cuda]"
```

### Basic Training

**A configuration file is required to run training.** We provide several example configs:

#### Using Hatch (Recommended)

```bash
# Quick test run (1000 steps)
hatch run train

# Standard training with specific config
hatch run train configs/experiments/small_test.json

# Production training
hatch run train-prod

# Multi-GPU training
hatch run train-multi experiment_config.json

# Mac-specific training
hatch run mac:train
hatch run mac:train-minimal

# CUDA environments
hatch run cuda:train-a40
hatch run cuda:train-a100
```

#### Direct Commands

```bash
# Quick test run (1000 steps)
accelerate launch src/train.py --config configs/experiments/small_test.json

# Standard training
accelerate launch src/train.py --config experiment_config.json

# Production training with larger model
accelerate launch src/train.py --config configs/experiments/production.json

# Multi-GPU training
accelerate launch --multi_gpu src/train.py --config experiment_config.json
```

## Configuration System

All parameters must be specified in a JSON configuration file. The training script uses three groups of parameters:

### 1. Model Parameters
Controls model architecture (Conformer encoder, SmolLM2 decoder, projector):

```json
"n_mels": 80,                    // Number of Mel bands
"d_model": 512,                  // Model dimension
"n_head": 8,                     // Attention heads
"num_layers": 12,                // Encoder layers
"decoder_model_name": "HuggingFaceTB/SmolLM2-360M-Instruct",
"use_lora": true,                // Enable LoRA fine-tuning
"lora_r": 16,                    // LoRA rank
"lora_alpha": 16,                // LoRA alpha scaling
```

### 2. Data Parameters
Controls dataset and preprocessing:

```json
"dataset_name": "librispeech_asr",     // Dataset to use
"dataset_config_name": "clean",        // Dataset configuration
"train_split": "train.100",            // Training split
"eval_split": "validation",            // Evaluation split
"max_audio_seconds": 20.0,             // Max audio length
"max_text_words": 150,                 // Max text length
"dataset_cache_dir": "/workspace/datasets"
```

### 3. Training Parameters
Standard Hugging Face training parameters:

```json
"output_dir": "/workspace/checkpoints",
"per_device_train_batch_size": 96,
"learning_rate": 8e-4,
"max_steps": 50000,
"eval_steps": 250,
"warmup_steps": 1000,
"bf16": true,
"gradient_checkpointing": true,
"push_to_hub": false,
"hub_model_id": "your-org/model-name"
```

## Hardware Setup with Accelerate

### Configure Your Environment

```bash
# Interactive setup (recommended for first time)
accelerate config

# Questions it will ask:
# - In which compute environment are you running? (This machine)
# - Which type of machine are you using? (No distributed training / multi-GPU)
# - Do you want to run your training on CPU only? (NO)
# - Do you wish to use FP16 or BF16? (bf16 for modern GPUs)
```

### Pre-configured Templates

We provide optimized configs in `configs/accelerate/`:

```bash
# Single GPU
accelerate launch --config_file configs/accelerate/single_gpu.yaml src/train.py

# Multi-GPU on single machine
accelerate launch --config_file configs/accelerate/multi_gpu.yaml src/train.py

# A40 optimized (48GB VRAM, 9 vCPUs)
OMP_NUM_THREADS=9 accelerate launch \
  --config_file configs/accelerate/a40_optimized.yaml \
  src/train.py --per_device_train_batch_size 96
```

## Example Configurations

We provide three complete configuration files:

### 1. `experiment_config.json` - Standard Training
- SmolLM2-360M decoder with LoRA
- 50,000 training steps
- Batch size 96 (optimized for A40)
- LibriSpeech train.100 dataset

### 2. `configs/experiments/small_test.json` - Quick Testing
- SmolLM2-135M decoder
- 1,000 steps for quick validation
- Smaller batch size (16)
- Reduced model layers (6)

### 3. `configs/experiments/production.json` - Production Training
- SmolLM2-1.7B decoder
- 100,000 training steps
- Larger batch size (128)
- Full LibriSpeech train.360 dataset
- More LoRA parameters

### Creating Your Own Config

Copy one of the example configs and modify as needed. All parameters must be specified:

```json
{
  // Training parameters (required)
  "output_dir": "/path/to/checkpoints",
  "per_device_train_batch_size": 32,
  "max_steps": 10000,
  "learning_rate": 1e-4,

  // Model parameters (required)
  "n_mels": 80,
  "d_model": 512,
  "decoder_model_name": "HuggingFaceTB/SmolLM2-360M-Instruct",

  // Data parameters (required)
  "dataset_name": "librispeech_asr",
  "train_split": "train.100",

  // ... (see example configs for complete list)
}
```

## Multi-GPU and Distributed Training

### Single Machine, Multiple GPUs

```bash
# Use all available GPUs
accelerate launch --multi_gpu src/train.py --config experiment_config.json

# Specify number of GPUs
accelerate launch --num_processes 4 src/train.py --config experiment_config.json
```

### Multiple Machines

First, configure for multi-node:
```bash
accelerate config
# Choose multi-node setup
```

Then launch on each machine:
```bash
# Main node (rank 0)
accelerate launch --config_file multi_node.yaml src/train.py --config experiment_config.json

# Worker nodes
accelerate launch --config_file multi_node.yaml src/train.py --config experiment_config.json
```

### Gradient Accumulation

For larger effective batch sizes with limited memory, modify your config file:
```json
{
  "gradient_accumulation_steps": 4,
  "per_device_train_batch_size": 24,
  // ... other parameters
}
```

## Hatch Environments

The project includes pre-configured Hatch environments for different use cases:

### Default Environment
Standard development environment with all core dependencies:
```bash
hatch run train                    # Run training
hatch run lint                     # Check code style
hatch run format                   # Auto-format code
```

### Mac Environment
Optimized for Mac with automatic MPS/CPU fallback:
```bash
hatch env create mac               # Create Mac environment
hatch run mac:train                # Run with Mac optimizations
hatch run mac:eval                 # Evaluation mode
```

### CUDA Environment
For GPU training with CUDA optimizations:
```bash
hatch env create cuda              # Create CUDA environment
hatch run cuda:train-a40           # A40-optimized training
hatch run cuda:train-a100          # A100-optimized training
```

### Test Environment
Dedicated testing environment:
```bash
hatch env create test              # Create test environment
hatch run test:run                 # Run tests
hatch run test:cov                 # Coverage report
hatch run test:parallel            # Parallel test execution
```

## Environment Variables

```bash
# Hugging Face Hub
export HF_TOKEN="your_token"           # For model upload
export HF_READ_TOKEN="your_read_token" # For private model access

# Weights & Biases (optional)
export WANDB_API_KEY="your_wandb_key"

# Cache directories
export HF_HOME="/workspace"
export HF_DATASETS_CACHE="/workspace/datasets"
```

## Hardware-Specific Optimizations

The script automatically optimizes for your hardware:

| GPU | VRAM | Suggested Batch Size | Notes |
|-----|------|---------------------|-------|
| A40 | 48GB | 96 | Ampere architecture, good for production |
| A100 | 40GB/80GB | 128-256 | Tensor cores, excellent performance |
| H100/H200 | 80GB+ | 512+ | Latest architecture, FP8 support |
| RTX 4090 | 24GB | 32-64 | Consumer GPU, good for development |
| RTX 3090 | 24GB | 32 | Older but capable |

## Monitoring Training

### TensorBoard

```bash
tensorboard --logdir /workspace/ASR_Conformer_SmolLM2_Optimized/checkpoints
```

### Weights & Biases

Set your API key for automatic logging:
```bash
export WANDB_API_KEY="your_key"
accelerate launch train.py
```

## Advanced Features

### Mixed Precision Training

```bash
# BF16 (recommended for A40, A100, H100)
accelerate launch --mixed_precision bf16 src/train.py --config experiment_config.json

# FP16 (for older GPUs)
accelerate launch --mixed_precision fp16 src/train.py --config experiment_config.json

# No mixed precision
accelerate launch --mixed_precision no src/train.py --config experiment_config.json
```

### DeepSpeed Integration

For very large models or extreme batch sizes:
```bash
accelerate config  # Choose DeepSpeed when prompted
accelerate launch src/train.py --config configs/experiments/production.json
```

### Model Compilation (PyTorch 2.0+)

The script automatically compiles models when possible for ~10-30% speedup.

## Performance Tips

1. **Start with moderate batch sizes**: Let Accelerate optimize for your hardware
2. **Use gradient accumulation**: Better than reducing batch size for memory constraints
3. **Enable gradient checkpointing**: Set `"gradient_checkpointing": true` in your config
4. **Use BF16**: Set `"bf16": true` for modern GPUs (more stable than FP16)
5. **Profile your training**:
   ```bash
   accelerate launch --dynamo_backend inductor src/train.py --config experiment_config.json
   ```

## Troubleshooting

### Out of Memory

Modify your config file to reduce memory usage:
```json
{
  "per_device_train_batch_size": 16,  // Reduce batch size
  "gradient_accumulation_steps": 4,    // Or use gradient accumulation
  "gradient_checkpointing": true,      // Enable checkpointing
  // ... other parameters
}
```

### Slow Training

- Verify mixed precision is enabled: `"bf16": true` in config
- Increase dataloader workers: `"dataloader_num_workers": 8` in config
- Ensure you're using the right accelerate config for your hardware

### Multi-GPU Issues

```bash
# Test your setup
accelerate test

# Check environment
accelerate env

# Verify GPUs are visible
nvidia-smi
```

## Model Architecture

### Conformer Encoder
- 12 transformer layers with Conformer blocks
- 512-dimensional embeddings
- 8 attention heads
- 4x subsampling via CNN layers
- SpecAugment for robustness

### SmolLM2 Decoder
- 360M parameter language model (default)
- LoRA fine-tuning (rank 8-16)
- Causal attention with audio prefix

### Audio-Text Projector
- Cross-attention mechanism
- 24 learnable queries
- Projects audio features to text embedding space

## Training Output

The training script saves:
- **Checkpoints**: `/workspace/ASR_Conformer_SmolLM2_Optimized/checkpoints/`
- **Final model**: `/workspace/ASR_Conformer_SmolLM2_Optimized/models/final_model/`
- **Logs**: TensorBoard and WandB (if configured)
- **Hub upload**: Automatic push to Hugging Face Hub (if configured)

## License

This project uses:
- Hugging Face Transformers (Apache 2.0)
- SmolLM2 models (Apache 2.0)
- LibriSpeech dataset (CC BY 4.0)

## Acknowledgments

Built with:
- 🤗 Hugging Face Transformers & Accelerate
- PyTorch and torchaudio
- NVIDIA Apex (when available)
- Weights & Biases for experiment tracking

## Getting Help

```bash
# Test accelerate setup
accelerate test

# Check environment
accelerate env

# View example config structure
cat experiment_config.json
```

## Usage Summary

### With Hatch (Recommended)

```bash
# Training commands
hatch run train                    # Quick test with default config
hatch run train-prod               # Production training
hatch run train-test               # Test configuration
hatch run train-multi              # Multi-GPU training

# Mac environment
hatch run mac:train                # Mac-optimized training
hatch run mac:train-minimal        # Minimal Mac config
hatch run mac:eval                 # Evaluation mode

# CUDA environment
hatch run cuda:train-a40           # A40-optimized
hatch run cuda:train-a100          # A100-optimized

# Development tools
hatch run lint                     # Run linters
hatch run format                   # Format code
hatch run typecheck               # Type checking
hatch run test                     # Run tests
hatch run test-cov                # Test coverage

# Utilities
hatch run tensorboard              # Launch TensorBoard
hatch run clean                    # Clean outputs
hatch run pod-copy                # Deploy to pod
```

### Direct Commands

```bash
# Required: always provide a config file
accelerate launch src/train.py --config <config_file.json>

# Examples:
accelerate launch src/train.py --config configs/experiments/small_test.json     # Quick test
accelerate launch src/train.py --config experiment_config.json                  # Standard run
accelerate launch src/train.py --config configs/experiments/production.json     # Production

# Multi-GPU:
accelerate launch --multi_gpu src/train.py --config experiment_config.json

# Custom hardware config:
accelerate launch --config_file configs/accelerate/a40_optimized.yaml \
  src/train.py --config experiment_config.json
```

For issues or questions, please check the example configuration files or create an issue in the repository.

## Testing ASR Training on Mac

This setup allows you to test the ASR training pipeline on your Mac with minimal resources.

### Mac Configuration Files

#### 1. `configs/experiments/test_config.json`
A minimal configuration for testing that:
- Uses tiny model sizes (128 d_model, 2 layers, 4 attention heads)
- Uses the smallest SmolLM2 model (135M parameters)
- Trains on only 20 samples, validates on 10
- Runs for 1 epoch with batch size 2
- Saves checkpoints every 20 steps
- Evaluates every 10 steps

#### 2. `configs/accelerate/accelerate_config_mac.yaml`
Accelerate configuration optimized for Mac:
- Single machine, single process
- No distributed training
- CPU/MPS backend (will use MPS if available on M1/M2 Macs)

### Mac Quick Start

#### Option 1: Using the Test Script
```bash
cd scripts
./test_mac.sh
```

This script will:
- Check and install dependencies if needed
- Set up environment variables for Mac
- Create necessary directories
- Run the training with test configuration

#### Option 2: Manual Run

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
accelerate launch --config_file configs/accelerate/accelerate_config_mac.yaml src/train.py --config configs/experiments/test_config.json
```

### Mac Monitoring Progress

1. **Console Output**: Watch the terminal for training progress
2. **TensorBoard**:
   ```bash
   tensorboard --logdir ./test_logs
   ```
   Then open http://localhost:6006 in your browser

### Expected Mac Performance

On a Mac (M1/M2 or Intel):
- Training should complete in 5-10 minutes
- Memory usage should stay under 4GB
- The model won't converge (it's too small) but should run without errors

### Mac Customization

To adjust for your Mac's capabilities:

#### For M1/M2 Macs with more memory:
- Increase `per_device_train_batch_size` to 4 or 8
- Increase `d_model` to 256 or 512
- Increase `num_layers` to 4 or 6

#### For faster testing:
- Reduce `train_split` to `"train.100[:10]"`
- Reduce `eval_split` to `"validation[:5]"`

#### For better model performance (longer training):
- Use `"HuggingFaceTB/SmolLM2-360M-Instruct"` as decoder
- Increase `num_train_epochs` to 3-5
- Increase dataset size by using `"train.100[:100]"`

### Mac Troubleshooting

#### If you get CUDA errors:
The config is set to use CPU/MPS. Make sure you're using the correct accelerate config.

#### If you get memory errors:
- Reduce `per_device_train_batch_size` to 1
- Reduce `d_model` to 64
- Enable `gradient_checkpointing: true` in config

#### If datasets download is slow:
The first run will download the LibriSpeech dataset. This can take a few minutes. The data is cached in `~/datasets` for future runs.

### Mac Cleanup

To remove test outputs:
```bash
rm -rf test_output test_logs ~/datasets
```