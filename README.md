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
pip install -r requirements.txt
```

### Basic Training

**A configuration file is required to run training.** We provide several example configs:

```bash
# Quick test run (1000 steps)
accelerate launch train.py --config configs/small_test.json

# Standard training
accelerate launch train.py --config experiment_config.json

# Production training with larger model
accelerate launch train.py --config configs/production.json

# Multi-GPU training
accelerate launch --multi_gpu train.py --config experiment_config.json
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

We provide optimized configs in `accelerate_configs/`:

```bash
# Single GPU
accelerate launch --config_file accelerate_configs/single_gpu.yaml train.py

# Multi-GPU on single machine
accelerate launch --config_file accelerate_configs/multi_gpu.yaml train.py

# A40 optimized (48GB VRAM, 9 vCPUs)
OMP_NUM_THREADS=9 accelerate launch \
  --config_file accelerate_configs/a40_optimized.yaml \
  train.py --per_device_train_batch_size 96
```

## Example Configurations

We provide three complete configuration files:

### 1. `experiment_config.json` - Standard Training
- SmolLM2-360M decoder with LoRA
- 50,000 training steps
- Batch size 96 (optimized for A40)
- LibriSpeech train.100 dataset

### 2. `configs/small_test.json` - Quick Testing
- SmolLM2-135M decoder
- 1,000 steps for quick validation
- Smaller batch size (16)
- Reduced model layers (6)

### 3. `configs/production.json` - Production Training
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
accelerate launch --multi_gpu train.py --config experiment_config.json

# Specify number of GPUs
accelerate launch --num_processes 4 train.py --config experiment_config.json
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
accelerate launch --config_file multi_node.yaml train.py --config experiment_config.json

# Worker nodes
accelerate launch --config_file multi_node.yaml train.py --config experiment_config.json
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
accelerate launch --mixed_precision bf16 train.py --config experiment_config.json

# FP16 (for older GPUs)
accelerate launch --mixed_precision fp16 train.py --config experiment_config.json

# No mixed precision
accelerate launch --mixed_precision no train.py --config experiment_config.json
```

### DeepSpeed Integration

For very large models or extreme batch sizes:
```bash
accelerate config  # Choose DeepSpeed when prompted
accelerate launch train.py --config configs/production.json
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
   accelerate launch --dynamo_backend inductor train.py --config experiment_config.json
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

```bash
# Required: always provide a config file
accelerate launch train.py --config <config_file.json>

# Examples:
accelerate launch train.py --config configs/small_test.json     # Quick test
accelerate launch train.py --config experiment_config.json       # Standard run
accelerate launch train.py --config configs/production.json      # Production

# Multi-GPU:
accelerate launch --multi_gpu train.py --config experiment_config.json

# Custom hardware config:
accelerate launch --config_file accelerate_configs/a40_optimized.yaml \
  train.py --config experiment_config.json
```

For issues or questions, please check the example configuration files or create an issue in the repository.