#!/usr/bin/env python3
"""
🎙️ ASR Training - Hydra Version
Training script using Hydra for configuration management with Whisper encoder and Qwen decoder.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import evaluate
import hydra
import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset, load_dataset
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    EvalPrediction,
    PreTrainedModel,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    WhisperFeatureExtractor,
    WhisperModel,
)
from transformers.models.llama.modeling_llama import LlamaRMSNorm as RMSNorm

# Minimal environment setup - Accelerate handles the rest
workspace_dir = "/workspace" if Path("/workspace").exists() else str(Path("~/.cache").expanduser())
os.environ["HF_HOME"] = os.environ.get("HF_HOME", workspace_dir)
os.environ["HF_DATASETS_CACHE"] = os.environ.get(
    "HF_DATASETS_CACHE", str(Path(workspace_dir) / "datasets")
)

class WhisperEncoder(nn.Module):

    def __init__(self, config: DictConfig):
        super().__init__()
        # Load Whisper-small model
        self.whisper = WhisperModel.from_pretrained(
            "openai/whisper-small", dtype=torch.bfloat16, token=False
        )

        for param in self.whisper.parameters():
            param.requires_grad = False

        self.d_model = self.whisper.config.d_model
        self.whisper.eval()

    def forward(self, x: torch.Tensor, input_lengths: torch.Tensor) -> torch.Tensor:
        batch_size, n_mels, time_frames = x.shape
        original_time_frames = time_frames

        expected_frames = 3000

        if time_frames < expected_frames:
            pad_amount = expected_frames - time_frames
            x = torch.nn.functional.pad(x, (0, pad_amount), mode="constant", value=0)
        elif time_frames > expected_frames:
            x = x[:, :, :expected_frames]
            original_time_frames = expected_frames

        with torch.no_grad():
            x = x.to(self.whisper.dtype)
            outputs = self.whisper.encoder(x)
            encoder_outputs = outputs.last_hidden_state

        if original_time_frames < expected_frames:
            actual_output_frames = (original_time_frames + 1) // 2
            encoder_outputs = encoder_outputs[:, :actual_output_frames, :]

        return encoder_outputs  # type: ignore[no-any-return]


class AudioProjector(nn.Module):
    def __init__(self, audio_dim: int, text_dim: int, config: DictConfig):
        super().__init__()
        self.norm = RMSNorm(audio_dim, eps=1e-6)

        self.linear_1 = nn.Linear(audio_dim, text_dim, bias=True)
        self.act = nn.GELU()
        self.linear_2 = nn.Linear(text_dim, text_dim, bias=True)

        nn.init.normal_(self.linear_1.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.linear_1.bias)
        nn.init.normal_(self.linear_2.weight, mean=0.0, std=0.002)
        nn.init.zeros_(self.linear_2.bias)

    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        hidden_states = self.norm(audio_features)
        hidden_states = self.linear_1(hidden_states)
        hidden_states = self.act(hidden_states)
        return self.linear_2(hidden_states)  # type: ignore[no-any-return]

class LLMDecoder(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.model: nn.Module = AutoModelForCausalLM.from_pretrained(
            config.model.decoder_model_name,
            dtype=torch.bfloat16,
            token=False,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(config.model.decoder_model_name, token=False)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.bos_token_id = self.tokenizer.bos_token_id
        self.model.config.eos_token_id = self.tokenizer.eos_token_id

        if hasattr(self.model, "generation_config") and self.model.generation_config is not None:
            self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
            if self.tokenizer.bos_token_id is not None:
                self.model.generation_config.bos_token_id = self.tokenizer.bos_token_id
            self.model.generation_config.eos_token_id = self.tokenizer.eos_token_id

        lora_config = LoraConfig(
            r=config.model.lora_r,
            lora_alpha=config.model.lora_alpha,
            target_modules=list(config.model.lora_target_modules),
            lora_dropout=config.model.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        self.model = get_peft_model(self.model, lora_config)

    def forward(self, **kwargs):
        return self.model(**kwargs)


class ASRModel(PreTrainedModel):
    base_model_prefix = "asr"
    supports_gradient_checkpointing = True
    _no_split_modules = ["WhisperEncoder", "LLMDecoder", "AudioProjector"]

    def __init__(self, config: DictConfig) -> None:
        from transformers import PretrainedConfig

        minimal_config = PretrainedConfig()
        super().__init__(minimal_config)

        self.hydra_config = config
        self.encoder = WhisperEncoder(config)
        self.decoder = LLMDecoder(config)
        self.config = self.decoder.model.config
        text_dim = getattr(self.decoder.model.config, "hidden_size", 768)
        audio_dim = self.encoder.d_model
        self.audio_projector = AudioProjector(audio_dim, text_dim, config)
        self.add_audio_special_tokens()

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if hasattr(self.decoder.model, "gradient_checkpointing_enable"):
            self.decoder.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        if hasattr(self.decoder.model, "gradient_checkpointing_disable"):
            self.decoder.model.gradient_checkpointing_disable()

    def add_audio_special_tokens(self):
        """Add audio-specific special tokens for better audio-text alignment."""
        special_tokens = {
            "additional_special_tokens": [
                "<|audio_start|>",
                "<|audio_end|>",
                "<|audio_pad|>",
                "<|audio_sep|>",
            ]
        }

        num_added = self.decoder.tokenizer.add_special_tokens(special_tokens)

        if num_added > 0:
            self.decoder.model.resize_token_embeddings(len(self.decoder.tokenizer))

            with torch.no_grad():
                embeddings = self.decoder.model.get_input_embeddings()
                if embeddings is not None and hasattr(embeddings, "weight"):
                    std_embedding = embeddings.weight[:-num_added].std()

                    for i in range(num_added):
                        embeddings.weight[-num_added + i] = torch.randn_like(
                            embeddings.weight[0]
                        ) * (std_embedding * 0.02)

        self.audio_start_id = self.decoder.tokenizer.convert_tokens_to_ids("<|audio_start|>")
        self.audio_end_id = self.decoder.tokenizer.convert_tokens_to_ids("<|audio_end|>")
        self.audio_pad_id = self.decoder.tokenizer.convert_tokens_to_ids("<|audio_pad|>")

    def forward(
        self,
        input_values: Optional[torch.Tensor] = None,
        input_lengths: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if input_values is None:
            input_values = kwargs.get("input_values")
        if input_lengths is None:
            input_lengths = kwargs.get("input_lengths")

        if input_values is None or input_lengths is None:
            raise ValueError("input_values and input_lengths are required")

        audio_features = self.encoder(input_values, input_lengths)
        audio_embeds = self.audio_projector(audio_features)

        batch_size, audio_seq_len, hidden_dim = audio_embeds.shape
        device = audio_embeds.device

        embed_layer = self.decoder.model.get_input_embeddings()

        audio_start_tokens = torch.full(
            (batch_size, 1), self.audio_start_id, device=device, dtype=torch.long
        )
        audio_end_tokens = torch.full(
            (batch_size, 1), self.audio_end_id, device=device, dtype=torch.long
        )

        audio_start_embeds = embed_layer(audio_start_tokens)
        audio_end_embeds = embed_layer(audio_end_tokens)

        if labels is not None:
            text_embeds = embed_layer(labels)
            audio_embeds = audio_embeds.to(text_embeds.dtype)

            inputs_embeds = torch.cat(
                [audio_start_embeds, audio_embeds, audio_end_embeds, text_embeds], dim=1
            )

            audio_len = audio_seq_len + 2
            if attention_mask is not None:
                audio_mask = torch.ones(
                    batch_size, audio_len, dtype=attention_mask.dtype, device=device
                )
                attention_mask = torch.cat([audio_mask, attention_mask], dim=1)
            else:
                attention_mask = torch.ones(
                    inputs_embeds.shape[:2], dtype=torch.long, device=device
                )

            audio_labels = torch.full(
                (batch_size, audio_len), -100, dtype=labels.dtype, device=device
            )
            labels = torch.cat([audio_labels, labels], dim=1)
        else:
            inputs_embeds = torch.cat([audio_start_embeds, audio_embeds, audio_end_embeds], dim=1)
            attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=device)
            labels = None

        return self.decoder.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )


    @torch.no_grad()
    def generate(
        self,
        input_values: torch.Tensor,
        input_lengths: torch.Tensor,
        max_new_tokens: int = 100,
        **kwargs,
    ) -> torch.Tensor:
        audio_features = self.encoder(input_values, input_lengths)
        audio_embeds = self.audio_projector(audio_features)

        batch_size = audio_embeds.shape[0]
        device = audio_embeds.device
        embed_layer = self.decoder.model.get_input_embeddings()

        audio_start_tokens = torch.full(
            (batch_size, 1), self.audio_start_id, device=device, dtype=torch.long
        )
        audio_end_tokens = torch.full(
            (batch_size, 1), self.audio_end_id, device=device, dtype=torch.long
        )

        audio_start_embeds = embed_layer(audio_start_tokens)
        audio_end_embeds = embed_layer(audio_end_tokens)

        inputs_embeds = torch.cat([audio_start_embeds, audio_embeds, audio_end_embeds], dim=1)

        return self.decoder.model.generate(  # type: ignore[no-any-return]
            inputs_embeds=inputs_embeds,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.decoder.tokenizer.pad_token_id,
            eos_token_id=self.decoder.tokenizer.eos_token_id,
            **kwargs,
        )

class DataCollator:
    """Data collator that performs preprocessing on-the-fly."""

    def __init__(
        self,
        tokenizer: Any,
        feature_extractor: WhisperFeatureExtractor,
        config: DictConfig,
    ):
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.sample_rate = config.data.sample_rate
        self.max_audio_seconds = config.data.max_audio_seconds
        self.max_text_words = config.data.max_text_words
        self.n_mels = 80  # Whisper uses 80 mel bins

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        # Filter samples that are too long
        valid_features = []
        for f in features:
            try:
                audio_array = f["audio"]["array"]
                audio_len_sec = len(audio_array) / self.sample_rate
                text_len_words = len(f["text"].split())
                if (
                    audio_len_sec <= self.max_audio_seconds
                    and text_len_words <= self.max_text_words
                ):
                    valid_features.append(f)
            except Exception:
                continue

        if not valid_features:
            pad_token_id = (
                self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            )
            dummy_seq_len = 10
            return {
                "input_values": torch.zeros((1, self.n_mels, 100)),
                "input_lengths": torch.tensor([100]),
                "labels": torch.full((1, dummy_seq_len), pad_token_id, dtype=torch.long),
                "attention_mask": torch.ones((1, dummy_seq_len), dtype=torch.long),
            }

        audio_arrays = []
        texts = []
        for f in valid_features:
            audio_arrays.append(f["audio"]["array"])
            texts.append(f["text"])

        audio_features = self.feature_extractor(
            audio_arrays,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding="longest",
        )

        padded_specs = audio_features.input_features

        input_lengths = torch.tensor(
            [min(len(arr) // 160, 3000) for arr in audio_arrays],
            dtype=torch.long,
        )

        labels = self.tokenizer(texts, padding="longest", truncation=True, return_tensors="pt")

        return {
            "input_values": padded_specs,
            "input_lengths": input_lengths,
            "labels": labels["input_ids"],
            "attention_mask": labels["attention_mask"],
        }



def initialize_model(config: DictConfig) -> Tuple[ASRModel, Any, WhisperFeatureExtractor]:
    """Initialize the ASR model, tokenizer, and feature extractor."""
    model = ASRModel(config)
    tokenizer = model.decoder.tokenizer
    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
    return model, tokenizer, feature_extractor


def load_datasets(config: DictConfig) -> Tuple[Dataset, Dataset]:
    """Load training and validation datasets."""
    import platform

    from datasets import Audio, concatenate_datasets

    safe_num_proc = 1 if platform.system() == "Darwin" else config.data.num_proc
    dataset_dicts = []

    for i, dataset_config in enumerate(config.data.dataset_configs):
        train_split = config.data.train_splits[i]
        eval_split = config.data.eval_splits[i]

        # Load dataset
        ds_dict = load_dataset(
            config.data.dataset_name,
            dataset_config,
            split={"train": train_split, "validation": eval_split},
            cache_dir=config.data.dataset_cache_dir,
            num_proc=safe_num_proc,
        )
        dataset_dicts.append(ds_dict)

    if len(dataset_dicts) > 1:
        train_dataset = concatenate_datasets([d["train"] for d in dataset_dicts])
        val_dataset = concatenate_datasets([d["validation"] for d in dataset_dicts])
    else:
        train_dataset = dataset_dicts[0]["train"]
        val_dataset = dataset_dicts[0]["validation"]

    # Ensure audio is decoded and resampled correctly
    train_dataset = train_dataset.cast_column("audio", Audio(sampling_rate=config.data.sample_rate))
    val_dataset = val_dataset.cast_column("audio", Audio(sampling_rate=config.data.sample_rate))

    if config.data.max_train_samples:
        train_dataset = train_dataset.select(
            range(min(config.data.max_train_samples, len(train_dataset)))
        )
    if config.data.max_eval_samples:
        val_dataset = val_dataset.select(range(min(config.data.max_eval_samples, len(val_dataset))))

    return train_dataset, val_dataset


class PredictionLoggingCallback(TrainerCallback):
    """Callback to log sample predictions to TensorBoard during evaluation."""

    def __init__(self, tokenizer, num_samples=5):
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.writer = None

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Log sample predictions after evaluation."""
        if metrics and hasattr(state, 'log_history') and len(state.log_history) > 0:
            # Get TensorBoard writer from trainer
            if self.writer is None and args.logging_dir:
                self.writer = SummaryWriter(log_dir=args.logging_dir)

            # Get sample predictions if they were stored
            if hasattr(state, 'sample_predictions'):
                samples = state.sample_predictions
                step = state.global_step

                # Create formatted text table for TensorBoard
                text_table = "| Ground Truth | Prediction |\n|---|---|\n"
                for _i, (truth, pred) in enumerate(samples[:self.num_samples]):
                    # Escape markdown special characters
                    truth = truth.replace('|', '\\|').replace('\n', ' ')
                    pred = pred.replace('|', '\\|').replace('\n', ' ')
                    text_table += f"| {truth} | {pred} |\n"

                if self.writer:
                    self.writer.add_text('predictions/samples', text_table, step)
                    self.writer.flush()


def compute_metrics(eval_pred: EvalPrediction, tokenizer: Any) -> Dict[str, float]:
    """Compute WER metric and store sample predictions."""
    predictions, label_ids = eval_pred.predictions, eval_pred.label_ids
    # Handle tuple of arrays (logits, hidden_states)
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    if predictions.ndim == 3:
        predictions = np.argmax(predictions, axis=-1)
    # Ensure label_ids is an array and create a copy
    if isinstance(label_ids, tuple):
        label_ids = label_ids[0]
    label_ids = label_ids.copy()
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # Store some sample predictions for logging (will be accessed by callback)
    sample_predictions = list(zip(decoded_labels[:10], decoded_preds[:10]))

    wer_metric = evaluate.load("wer", cache_dir="./metrics_cache")
    wer = wer_metric.compute(predictions=decoded_preds, references=decoded_labels)

    # Return metrics with samples attached
    metrics = {"wer": wer}
    metrics['_sample_predictions'] = sample_predictions  # Prefix with _ to avoid logging as metric
    return metrics


@hydra.main(version_base=None, config_path="../configs/hydra", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function using Hydra configuration."""

    print(OmegaConf.to_yaml(cfg))

    if cfg.eval_checkpoint:
        print(f"Evaluation mode: Loading checkpoint from {cfg.eval_checkpoint}")
        # TODO: Implement evaluation logic
        return

    model, tokenizer, feature_extractor = initialize_model(cfg)
    train_dataset, val_dataset = load_datasets(cfg)

    training_args_dict = OmegaConf.to_container(cfg.training, resolve=True)
    assert isinstance(training_args_dict, dict), "Training args must be a dict"
    training_args = TrainingArguments(**training_args_dict)  # type: ignore[arg-type]

    prediction_callback = PredictionLoggingCallback(tokenizer, num_samples=5)

    # Custom compute metrics that stores samples
    def compute_metrics_with_samples(eval_pred):
        if not cfg.training.compute_metrics:
            return None
        metrics = compute_metrics(eval_pred, tokenizer)
        if '_sample_predictions' in metrics:
            trainer.state.sample_predictions = metrics.pop('_sample_predictions')
        return metrics

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollator(
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
            config=cfg,
        ),
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_with_samples,
        callbacks=[prediction_callback] if cfg.training.compute_metrics else [],
    )

    trainer.train(resume_from_checkpoint=cfg.resume_from_checkpoint)
    trainer.save_model()


if __name__ == "__main__":
    main()
