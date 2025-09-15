#!/usr/bin/env python3
"""Convert checkpoint to Trainer-compatible format."""

import torch
import os
from pathlib import Path

def convert_checkpoint(checkpoint_dir):
    """Convert custom checkpoint format to Trainer-compatible format."""
    checkpoint_dir = Path(checkpoint_dir)

    print(f"📦 Converting checkpoint: {checkpoint_dir}")

    # Load individual components
    encoder_path = checkpoint_dir / "encoder.bin"
    projector_path = checkpoint_dir / "projector.bin"
    decoder_path = checkpoint_dir / "decoder"

    if not encoder_path.exists() or not projector_path.exists():
        print(f"❌ Error: Missing checkpoint components in {checkpoint_dir}")
        return False

    # Load state dicts
    encoder_state = torch.load(encoder_path, map_location='cpu')
    projector_state = torch.load(projector_path, map_location='cpu')

    # Load decoder state dict
    decoder_model_path = decoder_path / "model.safetensors"
    if decoder_model_path.exists():
        from safetensors.torch import load_file
        decoder_state = load_file(decoder_model_path)
    else:
        # Try pytorch format
        decoder_model_path = decoder_path / "pytorch_model.bin"
        if decoder_model_path.exists():
            decoder_state = torch.load(decoder_model_path, map_location='cpu')
        else:
            print(f"❌ Error: Cannot find decoder model in {decoder_path}")
            return False

    # Combine all state dicts with correct prefixes
    combined_state = {}

    # Add encoder states
    for key, value in encoder_state.items():
        combined_state[f"encoder.{key}"] = value

    # Add projector states
    for key, value in projector_state.items():
        combined_state[f"audio_projector.{key}"] = value

    # Add decoder states
    for key, value in decoder_state.items():
        combined_state[f"decoder.model.{key}"] = value

    # Save combined state dict as pytorch_model.bin
    output_path = checkpoint_dir / "pytorch_model.bin"
    torch.save(combined_state, output_path)

    print(f"✅ Saved combined model to {output_path}")
    print(f"   Total parameters: {len(combined_state)}")

    return True

if __name__ == "__main__":
    import sys

    checkpoint_dir = sys.argv[1] if len(sys.argv) > 1 else "./mac_minimal_output/checkpoint-2000"

    if convert_checkpoint(checkpoint_dir):
        print("✅ Checkpoint conversion complete!")
    else:
        print("❌ Checkpoint conversion failed!")