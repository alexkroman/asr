#!/usr/bin/env python3
"""Test script to validate data loading and Whisper feature extraction."""

import json
import numpy as np
import torch
import torchaudio
from datasets import load_dataset
from transformers import WhisperModel, WhisperProcessor
import os
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available, skipping visualizations")

def test_data_loading():
    """Test LibriSpeech data loading."""
    print("=" * 50)
    print("Testing Data Loading")
    print("=" * 50)

    # Load a small sample
    dataset = load_dataset(
        "librispeech_asr",
        "clean",
        split="train.100[:5]",
        cache_dir="./datasets_cache"
    )

    print(f"Loaded {len(dataset)} samples")
    print(f"Dataset features: {dataset.features}")

    for i, sample in enumerate(dataset):
        audio = sample["audio"]
        text = sample["text"]

        print(f"\nSample {i+1}:")
        print(f"  Text: {text[:100]}...")
        print(f"  Audio shape: {audio['array'].shape}")
        print(f"  Audio sampling rate: {audio['sampling_rate']} Hz")
        print(f"  Audio duration: {len(audio['array']) / audio['sampling_rate']:.2f} seconds")
        print(f"  Audio min: {audio['array'].min():.4f}")
        print(f"  Audio max: {audio['array'].max():.4f}")
        print(f"  Audio mean: {audio['array'].mean():.4f}")
        print(f"  Audio std: {audio['array'].std():.4f}")

        # Check for NaN or Inf
        if np.isnan(audio['array']).any():
            print("  WARNING: Audio contains NaN values!")
        if np.isinf(audio['array']).any():
            print("  WARNING: Audio contains Inf values!")

    return dataset

def test_whisper_extraction():
    """Test Whisper feature extraction."""
    print("\n" + "=" * 50)
    print("Testing Whisper Feature Extraction")
    print("=" * 50)

    # Load Whisper model and processor
    model = WhisperModel.from_pretrained(
        "openai/whisper-small",
        torch_dtype=torch.float32
    )
    processor = WhisperProcessor.from_pretrained("openai/whisper-small")

    # Create synthetic audio for testing
    sample_rate = 16000
    duration = 3  # seconds
    t = np.linspace(0, duration, sample_rate * duration)

    # Create a simple sine wave
    frequency = 440  # A4 note
    audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)

    print(f"Test audio shape: {audio.shape}")
    print(f"Test audio range: [{audio.min():.4f}, {audio.max():.4f}]")

    # Extract features using processor
    inputs = processor(
        audio,
        sampling_rate=sample_rate,
        return_tensors="pt"
    )

    print(f"\nProcessor outputs:")
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
            print(f"    min={value.min():.4f}, max={value.max():.4f}")
            print(f"    mean={value.mean():.4f}, std={value.std():.4f}")

    # Test with real data
    dataset = load_dataset(
        "librispeech_asr",
        "clean",
        split="train.100[:1]",
        cache_dir="./datasets_cache"
    )

    real_audio = dataset[0]["audio"]["array"]
    real_sr = dataset[0]["audio"]["sampling_rate"]

    print(f"\nReal audio sample:")
    print(f"  Shape: {real_audio.shape}")
    print(f"  Sampling rate: {real_sr} Hz")

    # Process real audio
    real_inputs = processor(
        real_audio,
        sampling_rate=real_sr,
        return_tensors="pt"
    )

    print(f"\nReal audio features:")
    for key, value in real_inputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")

            # Check for anomalies
            if torch.isnan(value).any():
                print(f"    WARNING: {key} contains NaN!")
            if torch.isinf(value).any():
                print(f"    WARNING: {key} contains Inf!")

    # Test model forward pass
    with torch.no_grad():
        encoder_outputs = model.encoder(real_inputs.input_features)
        print(f"\nEncoder output shape: {encoder_outputs.last_hidden_state.shape}")
        print(f"Encoder output range: [{encoder_outputs.last_hidden_state.min():.4f}, {encoder_outputs.last_hidden_state.max():.4f}]")

        if torch.isnan(encoder_outputs.last_hidden_state).any():
            print("WARNING: Encoder outputs contain NaN!")
        if torch.isinf(encoder_outputs.last_hidden_state).any():
            print("WARNING: Encoder outputs contain Inf!")

    return model, processor, real_inputs

def test_mel_spectrogram():
    """Test mel-spectrogram computation."""
    print("\n" + "=" * 50)
    print("Testing Mel-Spectrogram Computation")
    print("=" * 50)

    # Load a sample
    dataset = load_dataset(
        "librispeech_asr",
        "clean",
        split="train.100[:1]",
        cache_dir="./datasets_cache"
    )

    audio = dataset[0]["audio"]["array"]
    sr = dataset[0]["audio"]["sampling_rate"]

    # Create mel-spectrogram transform (matching training config)
    n_fft = 400
    hop_length = 160
    n_mels = 80

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )

    audio_tensor = torch.from_numpy(audio).unsqueeze(0)
    mel_spec = mel_transform(audio_tensor)

    print(f"Audio shape: {audio_tensor.shape}")
    print(f"Mel-spectrogram shape: {mel_spec.shape}")
    print(f"Mel-spectrogram range: [{mel_spec.min():.4f}, {mel_spec.max():.4f}]")
    print(f"Mel-spectrogram mean: {mel_spec.mean():.4f}")
    print(f"Mel-spectrogram std: {mel_spec.std():.4f}")

    # Log mel-spectrogram
    log_mel = torch.log(mel_spec + 1e-9)
    print(f"\nLog mel-spectrogram range: [{log_mel.min():.4f}, {log_mel.max():.4f}]")

    # Check for issues
    if torch.isnan(mel_spec).any():
        print("WARNING: Mel-spectrogram contains NaN!")
    if torch.isinf(mel_spec).any():
        print("WARNING: Mel-spectrogram contains Inf!")

    # Save visualization if matplotlib available
    if MATPLOTLIB_AVAILABLE:
        plt.figure(figsize=(12, 8))

        plt.subplot(3, 1, 1)
        plt.plot(audio[:sr])  # First second
        plt.title("Waveform (first second)")
        plt.ylabel("Amplitude")

        plt.subplot(3, 1, 2)
        plt.imshow(mel_spec[0].numpy(), aspect='auto', origin='lower')
        plt.title("Mel-Spectrogram")
        plt.ylabel("Mel bins")
        plt.colorbar()

        plt.subplot(3, 1, 3)
        plt.imshow(log_mel[0].numpy(), aspect='auto', origin='lower')
        plt.title("Log Mel-Spectrogram")
        plt.ylabel("Mel bins")
        plt.xlabel("Time frames")
        plt.colorbar()

        plt.tight_layout()
        plt.savefig("mel_spectrogram_test.png")
        print("\nSaved mel-spectrogram visualization to mel_spectrogram_test.png")
    else:
        print("\nSkipping visualization (matplotlib not available)")

    return mel_spec, log_mel

def test_audio_integrity():
    """Test audio data integrity and processing."""
    print("\n" + "=" * 50)
    print("Testing Audio Data Integrity")
    print("=" * 50)

    # Load multiple samples
    dataset = load_dataset(
        "librispeech_asr",
        "clean",
        split="train.100[:20]",
        cache_dir="./datasets_cache"
    )

    print(f"Testing {len(dataset)} samples for data integrity\n")

    issues_found = []

    for i, sample in enumerate(dataset):
        audio = sample["audio"]["array"]
        sr = sample["audio"]["sampling_rate"]
        text = sample["text"]

        # Check for various data issues
        issues = []

        # 1. Check for NaN/Inf
        if np.isnan(audio).any():
            issues.append("Contains NaN")
        if np.isinf(audio).any():
            issues.append("Contains Inf")

        # 2. Check audio range
        if audio.min() < -1.5 or audio.max() > 1.5:
            issues.append(f"Extreme values: [{audio.min():.2f}, {audio.max():.2f}]")

        # 3. Check for silence
        if audio.std() < 0.001:
            issues.append("Possibly silent")

        # 4. Check duration
        duration = len(audio) / sr
        if duration < 0.5:
            issues.append(f"Very short: {duration:.2f}s")
        if duration > 30:
            issues.append(f"Very long: {duration:.2f}s")

        # 5. Check text
        if len(text.strip()) == 0:
            issues.append("Empty text")
        if len(text) > 1000:
            issues.append(f"Very long text: {len(text)} chars")

        if issues:
            issues_found.append((i, issues))
            print(f"Sample {i}: Issues found - {', '.join(issues)}")

    if not issues_found:
        print("✓ No data integrity issues found in samples")
    else:
        print(f"\n⚠ Found issues in {len(issues_found)}/{len(dataset)} samples")

    # Test Whisper processing on samples with different characteristics
    print("\n" + "=" * 50)
    print("Testing Whisper Processing on Various Samples")
    print("=" * 50)

    processor = WhisperProcessor.from_pretrained("openai/whisper-small")

    # Test short, medium, long samples
    durations = []
    for i in range(min(10, len(dataset))):
        sample = dataset[i]
        audio = sample["audio"]["array"]
        sr = sample["audio"]["sampling_rate"]
        duration = len(audio) / sr
        durations.append(duration)

        try:
            inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
            print(f"✓ Processed {duration:.1f}s audio -> features shape: {inputs.input_features.shape}")

            # Check feature statistics
            features = inputs.input_features
            if torch.isnan(features).any() or torch.isinf(features).any():
                print(f"  ⚠ WARNING: Features contain NaN/Inf!")

        except Exception as e:
            print(f"✗ Failed to process {duration:.1f}s audio: {e}")

    print(f"\nDuration statistics: min={min(durations):.1f}s, max={max(durations):.1f}s, mean={np.mean(durations):.1f}s")

    return dataset

if __name__ == "__main__":
    # Test each component
    print("Starting comprehensive data loading and feature extraction tests...\n")

    # 1. Test data loading
    dataset = test_data_loading()

    # 2. Test Whisper extraction
    model, processor, inputs = test_whisper_extraction()

    # 3. Test mel-spectrogram
    mel_spec, log_mel = test_mel_spectrogram()

    # 4. Test audio integrity
    dataset_integrity = test_audio_integrity()

    print("\n" + "=" * 50)
    print("All tests completed!")
    print("=" * 50)
    print("\nSummary:")
    print("✓ Data loading working")
    print("✓ Whisper feature extraction working")
    print("✓ Mel-spectrogram computation working")
    print("✓ Audio integrity validated")
    if MATPLOTLIB_AVAILABLE:
        print("\nCheck mel_spectrogram_test.png for visualization")
    else:
        print("\n(Visualization skipped - matplotlib not available)")