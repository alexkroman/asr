#!/usr/bin/env python3
"""
🎙️ ASR Training
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import hydra
import numpy as np
import torch
from datasets import Dataset, load_dataset
from omegaconf import DictConfig, OmegaConf
from transformers import (
    AutoConfig,
    AutoModel,
    Trainer,
    TrainingArguments,
    WhisperFeatureExtractor,
)

# Import model classes from modeling.py
from modeling import (
    ASRModel,
    ASRModelConfig,
)


# Model classes have been moved to modeling.py for reusability
# Create ASRModel with proper config handling for training
def create_asr_model(config: DictConfig) -> ASRModel:
    """Create ASRModel with Hydra config."""
    # Convert Hydra config to ASRModelConfig
    asr_config = ASRModelConfig(
        decoder_model_name=config.model.decoder_model_name,
        lora_r=config.model.lora_r,
        lora_alpha=config.model.lora_alpha,
        lora_target_modules=list(config.model.lora_target_modules),
        lora_dropout=config.model.lora_dropout,
    )

    # Create model with the config
    model = ASRModel(asr_config)

    # Store the hydra config for backward compatibility
    model.hydra_config = config

    return model


def evaluate_samples(model, tokenizer, feature_extractor, eval_samples, device=None):
    """Evaluate model on samples and return predictions, references, and WER.

    This is a reusable function for both evaluation mode and callbacks.
    """
    import evaluate

    if device is None:
        device = model.device

    predictions = []
    references = []

    with torch.no_grad():
        for sample in eval_samples:
            # Process audio using feature extractor
            audio_array = sample["audio"]["array"]
            inputs = feature_extractor(
                audio_array,
                sampling_rate=16000,
                return_tensors="pt",
                return_attention_mask=True,
            )

            # Move to device
            input_features = inputs.input_features.to(device)
            attention_mask = inputs.attention_mask.to(device) if hasattr(inputs, 'attention_mask') else None

            # Use model's generate method which handles the full audio->text pipeline
            generated_ids = model.generate(
                input_features=input_features,
                attention_mask=attention_mask,
                max_new_tokens=100,
                do_sample=False,  # Greedy decoding for consistent evaluation
            )

            # Decode the generated tokens
            prediction = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            reference = sample.get("text", sample.get("sentence", ""))

            predictions.append(prediction)
            references.append(reference)

    # Calculate WER
    wer_metric = evaluate.load("wer")
    wer = wer_metric.compute(predictions=predictions, references=references)

    return predictions, references, wer


def clean_gigaspeech_text(text: str) -> Optional[str]:
    """Clean GigaSpeech text by replacing punctuation tags and filtering garbage utterances.

    Returns None if the text contains only garbage utterance tags.
    """
    # Check if text contains only garbage utterance tags
    garbage_tags = ["<SIL>", "<MUSIC>", "<NOISE>", "<OTHER>"]
    for tag in garbage_tags:
        if tag in text:
            # If the text is just a garbage tag or mostly garbage, skip it
            cleaned = text.replace(tag, "").strip()
            if not cleaned or len(cleaned) < 3:  # Very short after removing garbage
                return None

    # Replace punctuation tags with actual punctuation
    text = text.replace("<COMMA>", ",")
    text = text.replace("<PERIOD>", ".")
    text = text.replace("<QUESTIONMARK>", "?")
    text = text.replace("<EXCLAMATIONPOINT>", "!")

    # Remove any remaining garbage tags (in case they're mixed with real text)
    for tag in garbage_tags:
        text = text.replace(tag, " ")

    # Clean up multiple spaces and strip
    text = " ".join(text.split())
    text = text.strip()

    # Return None if empty after cleaning
    return text if text else None


class DataCollator:
    """Data collator that performs instruction formatting and masking."""

    def __init__(
        self,
        tokenizer: Any,
        feature_extractor: WhisperFeatureExtractor,
        config: DictConfig,
        model: Optional[ASRModel] = None,
    ):
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.sample_rate = config.data.sample_rate
        self.max_audio_seconds = config.data.max_audio_seconds
        self.model = model

    def __call__(
        self, features: List[Dict[str, Any]]
    ) -> Dict[str, Union[torch.Tensor, Optional[torch.Tensor], np.ndarray]]:
        # Filter samples that are too long
        valid_features = []
        for f in features:
            try:
                audio_array = f["audio"]["array"]
                audio_len_sec = len(audio_array) / self.sample_rate

                # Handle different field names for transcription text
                # LibriSpeech uses "text", GigaSpeech uses "text" or "sentence"
                text = f.get("text") or f.get("sentence") or ""

                # Clean GigaSpeech text (handles punctuation and garbage tags)
                cleaned_text = clean_gigaspeech_text(text)

                # Skip samples with only garbage utterances
                if cleaned_text is None:
                    continue

                text = cleaned_text

                if audio_len_sec <= self.max_audio_seconds:
                    # Normalize to use "text" field
                    if "text" not in f:
                        f["text"] = text
                    else:
                        f["text"] = text
                    valid_features.append(f)
            except Exception:
                continue

        if not valid_features:
            # Return a dummy batch if all samples were filtered
            return {
                "input_ids": torch.zeros((1, 10), dtype=torch.long),
                "labels": torch.zeros((1, 10), dtype=torch.long),
                "attention_mask": torch.zeros((1, 10), dtype=torch.long),
                "input_features": torch.zeros((1, 80, 3000)),
                "audio_attention_mask": torch.ones((1, 1500)),
            }

        # Process audio with feature extractor
        audio_arrays = [f["audio"]["array"] for f in valid_features]
        audio_features = self.feature_extractor(
            audio_arrays,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            return_attention_mask=True,
            padding="max_length",
            max_length=480000,
        )

        # Define instruction template
        instruction = (
            "User: Please transcribe the following audio recording.\n<|audio_chunk|>\nAssistant: "
        )

        # Format texts with instruction
        texts = [instruction + f["text"] for f in valid_features]

        # Tokenize the full instruction-tuned texts
        batch = self.tokenizer(
            texts,
            padding="longest",
            truncation=True,
            return_tensors="pt",
        )

        # Create labels by cloning input_ids
        labels = batch["input_ids"].clone()

        # Find the start of the assistant's response for masking
        # Tokenize the instruction prompt without the response
        instruction_prompt_tokens = self.tokenizer(instruction, return_tensors="pt")["input_ids"]
        prompt_length = instruction_prompt_tokens.shape[1]

        # Mask all tokens that are part of the prompt (user instruction)
        labels[:, :prompt_length] = -100

        # Also mask any pad tokens in the labels
        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100

        batch["labels"] = labels

        return {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "labels": batch["labels"],
            "input_features": audio_features.input_features,
            "audio_attention_mask": (
                audio_features.attention_mask if hasattr(audio_features, "attention_mask") else None
            ),
        }


def initialize_model(config: DictConfig) -> Tuple[ASRModel, Any, WhisperFeatureExtractor]:
    """Initialize the ASR model, tokenizer, and feature extractor."""
    model = create_asr_model(config)
    tokenizer = model.decoder.tokenizer
    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
    return model, tokenizer, feature_extractor


def load_datasets(config: DictConfig) -> Tuple[Dataset, Dataset]:
    """Load training and validation datasets in streaming mode."""
    from datasets import Audio, interleave_datasets

    train_datasets = []
    val_datasets = []

    # Process each dataset in the configuration
    for dataset_info in config.data.datasets:
        if dataset_info.name == "librispeech_asr":
            # Load LibriSpeech
            for i, dataset_config in enumerate(dataset_info.configs):
                train_split = dataset_info.train_splits[i]
                eval_split = dataset_info.eval_splits[i]

                # Streaming mode - loads data on-the-fly
                train_ds = load_dataset(
                    "librispeech_asr",
                    dataset_config,
                    split=train_split,
                    streaming=True,
                    cache_dir=config.data.dataset_cache_dir,
                )
                val_ds = load_dataset(
                    "librispeech_asr",
                    dataset_config,
                    split=eval_split,
                    streaming=True,
                    cache_dir=config.data.dataset_cache_dir,
                )

                train_datasets.append(train_ds)
                val_datasets.append(val_ds)

        elif dataset_info.name == "gigaspeech":
            # Load GigaSpeech (English only)
            import os

            token = os.environ.get("HUGGING_FACE_HUB_TOKEN", None)
            subset = dataset_info.subset if hasattr(dataset_info, "subset") else "xs"

            train_ds = load_dataset(
                "speechcolab/gigaspeech",
                subset,
                split=dataset_info.train_split,
                streaming=True,
                cache_dir=config.data.dataset_cache_dir,
                trust_remote_code=True,
                token=token,
            )
            val_ds = load_dataset(
                "speechcolab/gigaspeech",
                subset,
                split=dataset_info.eval_split,
                streaming=True,
                cache_dir=config.data.dataset_cache_dir,
                trust_remote_code=True,
                token=token,
            )

            # Filter out garbage utterances from GigaSpeech
            def filter_gigaspeech(example):
                text = example.get("text", "")
                cleaned = clean_gigaspeech_text(text)
                if cleaned:
                    example["text"] = cleaned
                    return True
                return False

            train_ds = train_ds.filter(filter_gigaspeech)
            val_ds = val_ds.filter(filter_gigaspeech)

            train_datasets.append(train_ds)
            val_datasets.append(val_ds)

        elif dataset_info.name == "common_voice":
            # Load Common Voice dataset
            import os

            token = os.environ.get("HUGGING_FACE_HUB_TOKEN", None)
            language = dataset_info.language if hasattr(dataset_info, "language") else "en"

            train_ds = load_dataset(
                "mozilla-foundation/common_voice_17_0",
                language,
                split=dataset_info.train_split,
                streaming=True,
                cache_dir=config.data.dataset_cache_dir,
                trust_remote_code=True,
                token=token,
            )
            val_ds = load_dataset(
                "mozilla-foundation/common_voice_17_0",
                language,
                split=dataset_info.eval_split,
                streaming=True,
                cache_dir=config.data.dataset_cache_dir,
                trust_remote_code=True,
                token=token,
            )
            # Rename 'sentence' to 'text' for Common Voice
            train_ds = train_ds.rename_column("sentence", "text")
            val_ds = val_ds.rename_column("sentence", "text")

            train_datasets.append(train_ds)
            val_datasets.append(val_ds)

    # Cast audio column to the correct sampling rate BEFORE interleaving
    # This ensures all datasets have compatible features
    for i in range(len(train_datasets)):
        train_datasets[i] = train_datasets[i].cast_column(
            "audio", Audio(sampling_rate=config.data.sample_rate)
        )
    for i in range(len(val_datasets)):
        val_datasets[i] = val_datasets[i].cast_column(
            "audio", Audio(sampling_rate=config.data.sample_rate)
        )

    # Interleave datasets for better mixing if we have multiple
    if len(train_datasets) > 1:
        train_dataset = interleave_datasets(
            train_datasets, stopping_strategy="first_exhausted"  # Stop when shortest dataset ends
        )
        val_dataset = interleave_datasets(val_datasets, stopping_strategy="first_exhausted")
    else:
        train_dataset = train_datasets[0]
        val_dataset = val_datasets[0]

    # Handle dataset size limits
    if config.data.max_train_samples:
        train_dataset = train_dataset.take(config.data.max_train_samples)
    if config.data.max_eval_samples:
        # Take limited samples for evaluation
        val_dataset = val_dataset.take(config.data.max_eval_samples)
        # Convert to regular dataset to avoid streaming issues during evaluation
        # This materializes only max_eval_samples into memory
        val_samples = list(val_dataset)
        val_dataset = Dataset.from_list(val_samples)

    return train_dataset, val_dataset


@hydra.main(version_base=None, config_path="../configs/hydra", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function using Hydra configuration."""
    import os

    from huggingface_hub import HfFolder

    print(OmegaConf.to_yaml(cfg))

    # Enable TensorFloat32 for better performance on Ampere and newer GPUs
    torch.set_float32_matmul_precision("high")

    # Normal training mode
    model, tokenizer, feature_extractor = initialize_model(cfg)
    train_dataset, val_dataset = load_datasets(cfg)

    # Handle checkpoint resuming
    if cfg.resume_from_checkpoint:
        print(f"Resuming training from checkpoint: {cfg.resume_from_checkpoint}")
        # The Trainer will handle loading optimizer and scheduler states

    training_args_dict = OmegaConf.to_container(cfg.training, resolve=True)
    assert isinstance(training_args_dict, dict), "Training args must be a dict"

    # Handle Hub token from environment if not in config
    if training_args_dict.get("push_to_hub", False):
        if not training_args_dict.get("hub_token"):
            # Try to get token from environment variables
            hub_token = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
            if hub_token:
                training_args_dict["hub_token"] = hub_token
            else:
                # Try to get from HfFolder (huggingface-cli login)
                hub_token = HfFolder.get_token()
                if hub_token:
                    training_args_dict["hub_token"] = hub_token

        # Log hub configuration
        print("\n📤 Hub push enabled:")
        print(f"   Repository: {training_args_dict.get('hub_model_id', 'auto-generated')}")
        print(f"   Strategy: {training_args_dict.get('hub_strategy', 'every_save')}")
        print(f"   Private: {training_args_dict.get('hub_private_repo', False)}")

    training_args = TrainingArguments(**training_args_dict)

    # Create custom callbacks
    import shutil
    from pathlib import Path

    from torch.utils.tensorboard import SummaryWriter
    from transformers import TrainerCallback

    class ModelingFileCopyCallback(TrainerCallback):
        """Copy modeling.py to the output directory so it gets pushed to Hub."""

        def on_save(self, args, state, control, **kwargs):
            modeling_src = Path(__file__).parent / "modeling.py"
            if modeling_src.exists():
                modeling_dst = Path(args.output_dir) / "modeling_asr.py"
                shutil.copy2(modeling_src, modeling_dst)
                print(f"✅ Copied modeling.py to {modeling_dst} for Hub upload")

    class PredictionLoggingCallback(TrainerCallback):
        """Log predictions and WER to TensorBoard periodically during training."""

        def __init__(self, eval_dataset, tokenizer, feature_extractor, cfg, log_predictions_every_n_steps=500):
            self.eval_dataset = eval_dataset
            self.tokenizer = tokenizer
            self.feature_extractor = feature_extractor
            self.cfg = cfg
            self.log_predictions_every_n_steps = log_predictions_every_n_steps
            self.writer = None

        def on_step_end(self, args, state, control, model=None, **kwargs):
            # Log predictions every N steps
            if state.global_step % self.log_predictions_every_n_steps == 0 and state.global_step > 0:
                self._log_predictions(args, state, model)

        def _log_predictions(self, args, state, model):
            """Generate predictions on a sample of validation data and log to TensorBoard."""
            if model is None:
                return

            # Initialize TensorBoard writer if needed
            if self.writer is None:
                self.writer = SummaryWriter(log_dir=args.logging_dir)

            model.eval()

            # Get a sample of validation data (configurable, default 10 samples)
            num_samples = getattr(self.cfg, 'log_predictions_samples', 10)

            # If eval_dataset is streaming, take samples
            if hasattr(self.eval_dataset, 'take'):
                eval_samples = list(self.eval_dataset.take(num_samples))
            else:
                # For regular dataset, take first N samples
                eval_samples = [self.eval_dataset[i] for i in range(min(num_samples, len(self.eval_dataset)))]

            print(f"\n📊 Generating predictions for logging (step {state.global_step})...")

            # Use the shared evaluation function
            predictions, references, wer = evaluate_samples(
                model, self.tokenizer, self.feature_extractor, eval_samples, device=model.device
            )

            # Log to TensorBoard
            self.writer.add_scalar("eval/wer", wer, state.global_step)

            # Log sample predictions as text
            predictions_text = []
            for i, (ref, pred) in enumerate(zip(references[:5], predictions[:5])):
                predictions_text.append(f"**Sample {i+1}**\n\nTruth: {ref}\n\nPrediction: {pred}\n\n---")

            full_text = "\n".join(predictions_text)
            self.writer.add_text("eval/predictions", full_text, state.global_step)

            # Also print to console
            print(f"📈 WER at step {state.global_step}: {wer:.2%}")
            print("\nSample predictions:")
            for _i, (ref, pred) in enumerate(zip(references[:3], predictions[:3])):
                print(f"  Truth:      {ref[:80]}...")
                print(f"  Prediction: {pred[:80]}...")
                print()

            model.train()

    # Determine which callbacks to use
    callbacks = []
    if training_args.push_to_hub:
        callbacks.append(ModelingFileCopyCallback())

    # Add prediction logging callback if we have eval dataset and TensorBoard is enabled
    if val_dataset and "tensorboard" in training_args.report_to:
        log_predictions_every_n_steps = getattr(cfg, 'log_predictions_every_n_steps', 500)
        callbacks.append(PredictionLoggingCallback(
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
            cfg=cfg,
            log_predictions_every_n_steps=log_predictions_every_n_steps
        ))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollator(
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
            config=cfg,
            model=model,
        ),
        processing_class=tokenizer,
        callbacks=callbacks,
        # No compute_metrics - only eval_loss will be calculated
    )

    # Train the model
    trainer.train(resume_from_checkpoint=cfg.resume_from_checkpoint)
    trainer.save_model()

    # Push final model to Hub if enabled
    if training_args.push_to_hub:
        print("\n🚀 Pushing final model to Hub...")
        trainer.push_to_hub()


# Register the model with AutoModel
AutoConfig.register("asr_model", ASRModelConfig)
AutoModel.register(ASRModelConfig, ASRModel)


if __name__ == "__main__":
    main()
