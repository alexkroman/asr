#!/usr/bin/env python3
"""
🎙️ ASR Model Evaluation Script
Evaluates a trained checkpoint on a validation dataset and calculates WER.
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import evaluate
import torch
from accelerate import Accelerator
from datasets import Audio, load_dataset
from torch.utils.data import DataLoader
from transformers import WhisperFeatureExtractor

# Add parent directory to path to import ASRModel
sys.path.append(str(Path(__file__).parent.parent))
from src.train import ASRModel


def load_model_and_processors(
    checkpoint_path: str,
) -> Tuple[Any, Any, WhisperFeatureExtractor]:
    """Load the model from checkpoint using the from_pretrained method."""
    print(f"Loading model from: {checkpoint_path}")

    # Use the ASRModel's from_pretrained method
    model = ASRModel.from_pretrained(checkpoint_path)
    model.eval()

    # Get tokenizer and feature extractor
    tokenizer = model.decoder.tokenizer
    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")

    return model, tokenizer, feature_extractor


def load_validation_dataset(
    dataset_name: str = "librispeech_asr",
    config: str = "clean",
    split: str = "validation",
    num_samples: int = 10,
) -> Any:
    """Load a validation dataset for evaluation using streaming."""
    print(f"Loading {dataset_name} dataset (config={config}, split={split}) in streaming mode...")

    # Use streaming to avoid downloading the entire dataset
    dataset = load_dataset(
        dataset_name,
        config,
        split=split,
        streaming=True,
        cache_dir="/workspace/datasets" if Path("/workspace").exists() else None,
    )

    # Cast audio to correct sampling rate
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    # Take only the specified number of samples
    if num_samples:
        dataset = dataset.take(num_samples)
        # Convert to list to materialize only these samples
        samples = list(dataset)
        # Convert back to a regular dataset for compatibility with DataLoader
        from datasets import Dataset

        dataset = Dataset.from_list(samples)
        print(f"Loaded {len(dataset)} samples for evaluation")
    else:
        print("Loaded streaming dataset for evaluation")

    return dataset


def evaluate_model(
    model: Any,
    tokenizer: Any,
    feature_extractor: WhisperFeatureExtractor,
    dataset: Any,
    batch_size: int = 1,
) -> Dict[str, Any]:
    """Evaluate the model on the dataset and calculate WER."""

    # Initialize accelerator
    accelerator = Accelerator(mixed_precision="fp16")

    # Create a simple data collator
    class SimpleDataCollator:
        def __init__(self, tokenizer, feature_extractor):
            self.tokenizer = tokenizer
            self.feature_extractor = feature_extractor

        def __call__(
            self, features: List[Dict[str, Any]]
        ) -> Dict[str, Union[torch.Tensor, Optional[torch.Tensor], List[str]]]:
            # Process audio
            audio_arrays = [f["audio"]["array"] for f in features]
            audio_features = self.feature_extractor(
                audio_arrays,
                sampling_rate=16000,
                return_tensors="pt",
                return_attention_mask=True,
                padding="max_length",
                max_length=480000,
            )

            # Process text (for reference)
            texts = [f.get("text", f.get("sentence", "")) for f in features]

            return {
                "input_values": audio_features.input_features,
                "encoder_attention_mask": (
                    audio_features.attention_mask
                    if hasattr(audio_features, "attention_mask")
                    else None
                ),
                "texts": texts,
            }

    collator = SimpleDataCollator(tokenizer, feature_extractor)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator)

    # Prepare model and dataloader with accelerator
    model, dataloader = accelerator.prepare(model, dataloader)

    print(f"Using device: {accelerator.device}")
    if accelerator.device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    all_predictions = []
    all_references = []

    print("\nGenerating predictions...")
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            # Inputs are automatically on the right device with accelerator
            input_values = batch["input_values"]
            encoder_attention_mask = batch["encoder_attention_mask"]

            # Unwrap the model to access custom methods
            unwrapped_model = accelerator.unwrap_model(model)

            # Generate predictions with the unwrapped model
            generated_ids = unwrapped_model.generate(
                input_values=input_values,
                encoder_attention_mask=encoder_attention_mask,
                max_new_tokens=100,
                do_sample=False,  # Greedy decoding for evaluation
                temperature=1.0,
            )

            # Decode predictions
            predictions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            references = batch["texts"]

            all_predictions.extend(predictions)
            all_references.extend(references)

            # Print progress
            print(f"Processed batch {i+1}/{len(dataloader)}")

    # Calculate WER
    print("\nCalculating WER...")
    wer_metric = evaluate.load("wer")
    wer = wer_metric.compute(predictions=all_predictions, references=all_references)

    # Print sample predictions
    print("\n" + "=" * 80)
    print("SAMPLE PREDICTIONS (Truth | Prediction)")
    print("=" * 80)

    for i, (ref, pred) in enumerate(zip(all_references, all_predictions)):
        print(f"\nSample {i+1}:")
        print(f"Truth:      {ref}")
        print(f"Prediction: {pred}")
        print("-" * 80)

    print(f"\n{'='*80}")
    print("EVALUATION RESULTS")
    print(f"{'='*80}")
    print(f"Total samples evaluated: {len(all_predictions)}")
    print(f"Word Error Rate (WER): {wer:.2%}")

    return {
        "wer": wer,
        "num_samples": len(all_predictions),
        "predictions": list(zip(all_references, all_predictions)),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate an ASR model checkpoint")
    parser.add_argument("checkpoint", type=str, help="Path to model checkpoint directory")
    parser.add_argument(
        "--dataset",
        type=str,
        default="librispeech_asr",
        help="Dataset name (default: librispeech_asr)",
    )
    parser.add_argument(
        "--config", type=str, default="clean", help="Dataset config (default: clean)"
    )
    parser.add_argument(
        "--split", type=str, default="validation", help="Dataset split (default: validation)"
    )
    parser.add_argument(
        "--num-samples", type=int, default=10, help="Number of samples to evaluate (default: 10)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Batch size for evaluation (default: 1)"
    )

    args = parser.parse_args()

    # Check if checkpoint exists
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint path does not exist: {checkpoint_path}")
        sys.exit(1)

    # Load model
    model, tokenizer, feature_extractor = load_model_and_processors(str(checkpoint_path))

    # Load dataset
    dataset = load_validation_dataset(
        dataset_name=args.dataset,
        config=args.config,
        split=args.split,
        num_samples=args.num_samples,
    )

    # Evaluate
    evaluate_model(
        model=model,
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        dataset=dataset,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
