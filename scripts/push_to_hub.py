#!/usr/bin/env python3
"""
Push ASR model checkpoint to Hugging Face Hub.

Usage:
    python push_to_hub.py --checkpoint_dir /path/to/checkpoint --repo_name username/model-name
"""

import argparse
import os
import sys
import torch
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_folder
from transformers import AutoTokenizer, AutoModelForCausalLM

def parse_args():
    parser = argparse.ArgumentParser(description="Push ASR model checkpoint to Hugging Face Hub")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Path to checkpoint directory (e.g., /workspace/production_model/checkpoint-500)"
    )
    parser.add_argument(
        "--repo_name",
        type=str,
        required=True,
        help="Repository name on HF Hub (e.g., username/asr-conformer-smollm2)"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the repository private"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face API token (or set HF_TOKEN env variable)"
    )
    parser.add_argument(
        "--push_code",
        action="store_true",
        help="Also push the model code (train.py) to the repository"
    )
    return parser.parse_args()


def create_model_card(repo_name: str, checkpoint_dir: str) -> str:
    """Create a model card for the repository."""
    return f"""---
license: apache-2.0
tags:
- automatic-speech-recognition
- asr
- conformer
- smollm2
- audio
- speech
language:
- en
datasets:
- librispeech_asr
metrics:
- wer
model-index:
- name: {repo_name.split('/')[-1]}
  results:
  - task:
      type: automatic-speech-recognition
      name: Automatic Speech Recognition
    dataset:
      name: LibriSpeech
      type: librispeech_asr
---

# Conformer-SmolLM2 ASR Model

This model combines a Conformer encoder for audio processing with SmolLM2 decoder for text generation, trained for automatic speech recognition (ASR).

## Model Architecture
- **Encoder**: Conformer (12 layers, 512 hidden dim)
- **Decoder**: SmolLM2-1.7B with LoRA adapters
- **Audio Projector**: Deep projector with cross-attention

## Training Details
- **Dataset**: LibriSpeech (clean + other)
- **Training steps**: 10,000
- **Batch size**: 128 (64 * 2 gradient accumulation)
- **Learning rate**: 1e-4 with cosine schedule
- **Mixed precision**: bfloat16

## Usage

```python
import torch
from transformers import AutoTokenizer
from train import ASRModel, ModelArguments

# Load model
model = ASRModel.from_pretrained("{repo_name}")
tokenizer = AutoTokenizer.from_pretrained("{repo_name}")

# Transcribe audio
# audio: numpy array or torch tensor (16kHz)
transcription = model.transcribe(audio)
```

## Files
- `pytorch_model.bin`: Full model weights
- `config.json`: Model configuration
- `tokenizer_*`: Tokenizer files
- `train.py`: Model implementation (if included)

## Performance
- Evaluation Loss: Check tensorboard logs
- WER: Calculated on validation set

## Citation
If you use this model, please cite:
```
@misc{{conformer-smollm2-asr,
  author = {{Your Name}},
  title = {{Conformer-SmolLM2 ASR}},
  year = {{2024}},
  publisher = {{Hugging Face}},
  url = {{https://huggingface.co/{repo_name}}}
}}
```
"""


def prepare_checkpoint_for_hub(checkpoint_dir: str, output_dir: str):
    """Prepare checkpoint files for upload."""
    os.makedirs(output_dir, exist_ok=True)

    # Files to copy directly
    direct_copy_patterns = [
        "*.json",  # config files
        "*.txt",   # tokenizer vocab
        "*.model", # sentencepiece model
        "tokenizer*",
        "special_tokens_map.json",
        "chat_template.jinja",
    ]

    # Copy necessary files
    checkpoint_path = Path(checkpoint_dir)
    output_path = Path(output_dir)

    for pattern in direct_copy_patterns:
        for file in checkpoint_path.glob(pattern):
            if file.is_file():
                target = output_path / file.name
                print(f"Copying {file.name}...")
                import shutil
                shutil.copy2(file, target)

    # Handle model weights
    if (checkpoint_path / "pytorch_model.bin").exists():
        print("Copying pytorch_model.bin...")
        import shutil
        shutil.copy2(checkpoint_path / "pytorch_model.bin", output_path / "pytorch_model.bin")
    elif (checkpoint_path / "model.safetensors").exists():
        print("Copying model.safetensors...")
        import shutil
        shutil.copy2(checkpoint_path / "model.safetensors", output_path / "model.safetensors")

    # Handle decoder (LoRA adapters or full model)
    decoder_path = checkpoint_path / "decoder"
    if decoder_path.exists():
        print("Copying decoder weights...")
        import shutil
        shutil.copytree(decoder_path, output_path / "decoder", dirs_exist_ok=True)

    # Copy encoder and projector if they exist
    for component in ["encoder.bin", "projector.bin", "audio_projector.bin"]:
        if (checkpoint_path / component).exists():
            print(f"Copying {component}...")
            import shutil
            shutil.copy2(checkpoint_path / component, output_path / component)


def main():
    args = parse_args()

    # Validate checkpoint directory
    if not os.path.exists(args.checkpoint_dir):
        print(f"Error: Checkpoint directory '{args.checkpoint_dir}' does not exist!")
        sys.exit(1)

    # Get token
    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        print("Warning: No HF token provided. You may need to login with 'huggingface-cli login'")

    # Create repository
    print(f"\n🚀 Creating repository: {args.repo_name}")
    try:
        api = HfApi()
        repo_url = create_repo(
            repo_id=args.repo_name,
            repo_type="model",
            private=args.private,
            token=token,
            exist_ok=True
        )
        print(f"✅ Repository created/found: {repo_url}")
    except Exception as e:
        print(f"❌ Error creating repository: {e}")
        sys.exit(1)

    # Prepare files for upload
    print("\n📦 Preparing checkpoint files...")
    temp_dir = "/tmp/hf_upload_temp"
    prepare_checkpoint_for_hub(args.checkpoint_dir, temp_dir)

    # Create and save model card
    print("📝 Creating model card...")
    model_card = create_model_card(args.repo_name, args.checkpoint_dir)
    with open(f"{temp_dir}/README.md", "w") as f:
        f.write(model_card)

    # Copy training script if requested
    if args.push_code:
        train_script = Path(__file__).parent / "src" / "train.py"
        if train_script.exists():
            print("📄 Copying training script...")
            import shutil
            shutil.copy2(train_script, f"{temp_dir}/train.py")

    # Upload to hub
    print(f"\n⬆️ Uploading to {args.repo_name}...")
    try:
        upload_folder(
            folder_path=temp_dir,
            repo_id=args.repo_name,
            repo_type="model",
            token=token,
            commit_message=f"Upload checkpoint from {args.checkpoint_dir}"
        )
        print(f"\n✅ Successfully uploaded to: https://huggingface.co/{args.repo_name}")
    except Exception as e:
        print(f"❌ Error uploading to hub: {e}")
        sys.exit(1)

    # Cleanup
    print("🧹 Cleaning up temporary files...")
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)

    print("\n🎉 Done! Your model is now available at:")
    print(f"   https://huggingface.co/{args.repo_name}")
    print("\nTo use it:")
    print(f"   from transformers import AutoModel")
    print(f"   model = AutoModel.from_pretrained('{args.repo_name}')")


if __name__ == "__main__":
    main()