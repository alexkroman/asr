# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
This is an ASR (Automatic Speech Recognition) training pipeline that combines a Whisper encoder with SmolLM2/Qwen decoder using PyTorch and Hugging Face transformers. The project uses Hydra for configuration management and supports both local Mac development and cloud GPU training on RunPod.

## Key Commands

### Development & Linting
- **Lint code**: `hatch run lint` (runs ruff check and mypy on src/)
- **Format code**: `hatch run format` (runs ruff format and black on src/)
- **Run specific test**: `pytest tests/test_file.py::test_function`
- **Run all tests**: `pytest tests/`

### Training
- **Train locally on Mac**: `hatch run train-mac` or `python src/train.py +experiments=mac_minimal`
- **Train with custom config**: `python src/train.py +experiments=your_experiment`
- **Train in production (RunPod)**: `hatch run cuda:train-prod`
- **Evaluate checkpoint**: `python src/train.py +experiments=production eval_checkpoint=./outputs/production_model/checkpoint-1000`

### Docker & Deployment
- **Build Docker image**: `hatch run docker-build`
- **Build local Docker**: `hatch run docker-build-local`
- **Push to registry**: `hatch run docker-push`
- **Deploy locally**: `hatch run deploy-docker-local`

### RunPod Management
- **Create RunPod instance**: `hatch run create-a20`
- **Sync code to RunPod**: `hatch run sync-to-runpod`

## Architecture

### Core Components
1. **WhisperEncoder** (`src/train.py`): Wraps OpenAI's Whisper-small model for audio encoding. Frozen weights, outputs 768-dim features.

2. **AudioProjector** (`src/train.py`): Projects Whisper features to match decoder's input dimension using a 2-layer MLP with GELU activation and RMSNorm.

3. **ASRModel** (`src/train.py`): Main model combining encoder, projector, and decoder (SmolLM2 or Qwen). Supports LoRA for efficient fine-tuning.

### Configuration System (Hydra)
- **Base config**: `configs/hydra/config.yaml` - defines defaults and output directories
- **Model configs**: `configs/hydra/model/` - defines model architecture (small/medium/large)
- **Data configs**: `configs/hydra/data/` - dataset settings
- **Training configs**: `configs/hydra/training/` - training hyperparameters
- **Experiments**: `configs/hydra/experiments/` - predefined experiment configurations combining the above
- **Accelerate configs**: `configs/accelerate/` - hardware-specific acceleration settings

### Training Pipeline
The training script (`src/train.py`) uses Hugging Face's Trainer API with custom data collation for handling variable-length audio inputs. Key features:
- Dynamic batching with proper padding/masking
- Mixed precision training (bf16 on compatible hardware)
- Gradient checkpointing for memory efficiency
- TensorBoard logging
- Checkpoint management with best model selection

## Environment Variables
For Mac development:
- `PYTORCH_ENABLE_MPS_FALLBACK=1` - Enable fallback for unsupported MPS operations
- `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` - Optimize MPS memory usage
- `TOKENIZERS_PARALLELISM=false` - Avoid tokenizer warnings

For RunPod/CUDA:
- `HF_HOME=/workspace/.cache/huggingface` - Cache directory
- `HF_DATASETS_CACHE=/workspace/datasets` - Dataset cache
- `HF_HUB_ENABLE_HF_TRANSFER=1` - Enable fast downloads

## Dataset
The project uses the LibriSpeech dataset from Hugging Face. Data processing includes:
- Audio resampling to 16kHz
- Mel-spectrogram extraction (80 bins)
- Dynamic sequence length handling with proper masking
- Tokenization using the decoder's tokenizer

## Key Dependencies
- PyTorch ecosystem: torch, torchaudio, torchvision
- Hugging Face: transformers, datasets, accelerate, peft
- Configuration: hydra-core, omegaconf
- Audio: librosa, soundfile
- Evaluation: jiwer (WER calculation), evaluate