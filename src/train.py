#!/usr/bin/env python3
"""
🎙️ ASR Training
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import evaluate
import hydra
import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset, load_dataset
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, PeftMixedModel, PeftModel, TaskType, get_peft_model
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    EvalPrediction,
    PretrainedConfig,
    PreTrainedModel,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    WhisperFeatureExtractor,
    WhisperModel,
)
from transformers.models.llama.modeling_llama import LlamaRMSNorm as RMSNorm


class WhisperEncoder(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.whisper = WhisperModel.from_pretrained(
            "openai/whisper-small", dtype=torch.bfloat16, token=False
        )

        for param in self.whisper.parameters():
            param.requires_grad = False

        self.d_model = self.whisper.config.d_model
        self.whisper.eval()

    def forward(self, input_features: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        input_features and attention_mask come directly from the WhisperFeatureExtractor.
        """
        with torch.no_grad():
            input_features = input_features.to(self.whisper.dtype)
            # The model internally handles the attention mask to ignore padding
            outputs = self.whisper.encoder(input_features, attention_mask=attention_mask)
            return outputs.last_hidden_state


class AudioProjector(nn.Module):
    def __init__(self, audio_dim: int, text_dim: int, config: DictConfig):
        super().__init__()
        self.norm = RMSNorm(audio_dim, eps=1e-6)

        self.linear_1 = nn.Linear(audio_dim, text_dim, bias=True)
        self.act = nn.GELU()
        self.linear_2 = nn.Linear(text_dim, text_dim, bias=True)

        nn.init.normal_(self.linear_1.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.linear_1.bias)
        nn.init.normal_(self.linear_2.weight)
        nn.init.zeros_(self.linear_2.bias)

    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        hidden_states = self.norm(audio_features)
        hidden_states = self.linear_1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)

        return hidden_states * 0.01


class LLMDecoder(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.model: Union[PreTrainedModel, PeftModel, PeftMixedModel] = AutoModelForCausalLM.from_pretrained(
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
                    existing_embeds = embeddings.weight[:-num_added]
                    mean_embedding = existing_embeds.mean(dim=0)
                    std_embedding = existing_embeds.std()

                    for i in range(num_added):
                        embeddings.weight[-num_added + i] = mean_embedding + torch.randn_like(
                            embeddings.weight[0]
                        ) * (std_embedding * 0.02)

        self.audio_start_id = self.decoder.tokenizer.convert_tokens_to_ids("<|audio_start|>")
        self.audio_end_id = self.decoder.tokenizer.convert_tokens_to_ids("<|audio_end|>")
        self.audio_pad_id = self.decoder.tokenizer.convert_tokens_to_ids("<|audio_pad|>")

    def forward(
        self,
        input_values: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if input_values is None:
            input_values = kwargs.get("input_values")
        if encoder_attention_mask is None:
            encoder_attention_mask = kwargs.get("encoder_attention_mask")

        if input_values is None:
            raise ValueError("input_values are required")

        audio_features = self.encoder(input_values, attention_mask=encoder_attention_mask)
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
        encoder_attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 100,
        **kwargs,
    ) -> torch.Tensor:
        audio_features = self.encoder(input_values, attention_mask=encoder_attention_mask)
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


def preprocess_audio_text(
    examples: Dict[str, Any],
    feature_extractor: WhisperFeatureExtractor,
    tokenizer: Any,
    sample_rate: int,
) -> Dict[str, Any]:
    """Preprocess audio and text examples for the dataset."""
    audio_arrays = [x["array"] for x in examples["audio"]]

    # Process audio with feature extractor
    audio_features = feature_extractor(
        audio_arrays,
        sampling_rate=sample_rate,
        return_attention_mask=True,
        return_tensors="pt",
        padding="longest",
    )

    # Process text with tokenizer
    text_outputs = tokenizer(
        examples["text"],
        padding="longest",
        truncation=True,
        return_tensors="pt",
    )

    return {
        "input_values": audio_features.input_features,
        "encoder_attention_mask": audio_features.attention_mask if hasattr(audio_features, "attention_mask") else None,
        "labels": text_outputs["input_ids"],
        "attention_mask": text_outputs["attention_mask"],
    }


class DataCollator:
    """Simple data collator that only handles batching and padding."""

    def __init__(
        self,
        tokenizer: Any,
        feature_extractor: WhisperFeatureExtractor,
    ):
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        # Stack tensors that are already preprocessed
        batch = {}

        # Get all keys from the first feature
        keys = features[0].keys()

        for key in keys:
            if key == "encoder_attention_mask" and features[0][key] is None:
                continue

            if isinstance(features[0][key], torch.Tensor):
                # Pad sequences to the same length within the batch
                if key == "input_values":
                    # Pad mel spectrograms along the time dimension
                    max_len = max(f[key].shape[-1] for f in features)
                    padded = []
                    for f in features:
                        spec = f[key]
                        if spec.shape[-1] < max_len:
                            pad_amount = max_len - spec.shape[-1]
                            spec = torch.nn.functional.pad(spec, (0, pad_amount), value=0)
                        padded.append(spec)
                    batch[key] = torch.stack(padded)
                elif key in ["labels", "attention_mask"]:
                    # Pad text tokens
                    max_len = max(f[key].shape[-1] for f in features)
                    padded = []
                    for f in features:
                        seq = f[key]
                        if seq.shape[-1] < max_len:
                            pad_amount = max_len - seq.shape[-1]
                            pad_value = self.pad_token_id if key == "labels" else 0
                            seq = torch.nn.functional.pad(seq, (0, pad_amount), value=pad_value)
                        padded.append(seq)
                    batch[key] = torch.stack(padded)
                else:
                    # Stack other tensors directly
                    batch[key] = torch.stack([f[key] for f in features])
            else:
                # Handle non-tensor values
                batch[key] = [f[key] for f in features]

        return batch


def initialize_model(config: DictConfig) -> Tuple[ASRModel, Any, WhisperFeatureExtractor]:
    """Initialize the ASR model, tokenizer, and feature extractor."""
    model = ASRModel(config)
    tokenizer = model.decoder.tokenizer
    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
    return model, tokenizer, feature_extractor


def load_datasets(
    config: DictConfig,
    feature_extractor: WhisperFeatureExtractor,
    tokenizer: Any
) -> Tuple[Dataset, Dataset]:
    """Load training and validation datasets with preprocessing."""
    import platform

    from datasets import Audio, concatenate_datasets

    # Use multiprocessing for both filter and map operations
    filter_num_proc = 1 if platform.system() == "Darwin" else config.data.num_proc
    map_num_proc = 1  # Use single process for map operations to avoid tokenizer issues
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
            num_proc=filter_num_proc,
        )
        dataset_dicts.append(ds_dict)

    if len(dataset_dicts) > 1:
        train_dataset = concatenate_datasets([d["train"] for d in dataset_dicts])
        val_dataset = concatenate_datasets([d["validation"] for d in dataset_dicts])
    else:
        train_dataset = dataset_dicts[0]["train"]
        val_dataset = dataset_dicts[0]["validation"]

    train_dataset = train_dataset.cast_column("audio", Audio(sampling_rate=config.data.sample_rate))
    val_dataset = val_dataset.cast_column("audio", Audio(sampling_rate=config.data.sample_rate))

    if config.data.max_train_samples:
        train_dataset = train_dataset.select(
            range(min(config.data.max_train_samples, len(train_dataset)))
        )
    if config.data.max_eval_samples:
        val_dataset = val_dataset.select(range(min(config.data.max_eval_samples, len(val_dataset))))

    # Filter samples based on length constraints
    def filter_by_length(example):
        audio_len_sec = len(example["audio"]["array"]) / config.data.sample_rate
        text_len_words = len(example["text"].split())
        return (
            audio_len_sec <= config.data.max_audio_seconds
            and text_len_words <= config.data.max_text_words
        )

    train_dataset = train_dataset.filter(
        filter_by_length,
        num_proc=filter_num_proc,
        load_from_cache_file=True,
        desc="Filtering training data by length"
    )
    val_dataset = val_dataset.filter(
        filter_by_length,
        num_proc=filter_num_proc,
        load_from_cache_file=True,
        desc="Filtering validation data by length"
    )

    # Apply preprocessing
    def preprocess_batch(examples):
        return preprocess_audio_text(examples, feature_extractor, tokenizer, config.data.sample_rate)

    # Remove original columns and keep only preprocessed features
    columns_to_remove = train_dataset.column_names

    train_dataset = train_dataset.map(
        preprocess_batch,
        batched=True,
        remove_columns=columns_to_remove,
        num_proc=map_num_proc,
        load_from_cache_file=True,
        desc="Preprocessing training data",
    )

    val_dataset = val_dataset.map(
        preprocess_batch,
        batched=True,
        remove_columns=columns_to_remove,
        num_proc=map_num_proc,
        load_from_cache_file=True,
        desc="Preprocessing validation data",
    )

    # Set format to PyTorch tensors
    train_dataset.set_format(type="torch")
    val_dataset.set_format(type="torch")

    return train_dataset, val_dataset


class PredictionLoggingCallback(TrainerCallback):
    """Callback to log sample predictions to TensorBoard during evaluation."""

    def __init__(self, tokenizer, num_samples=5):
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.writer = None

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Log sample predictions after evaluation."""
        if metrics and hasattr(state, "log_history") and len(state.log_history) > 0:
            if self.writer is None and args.logging_dir:
                self.writer = SummaryWriter(log_dir=args.logging_dir)

            if hasattr(state, "sample_predictions"):
                samples = state.sample_predictions
                step = state.global_step

                text_table = "| Ground Truth | Prediction |\n|---|---|\n"
                for _i, (truth, pred) in enumerate(samples[: self.num_samples]):
                    truth = truth.replace("|", "\\|").replace("\n", " ")
                    pred = pred.replace("|", "\\|").replace("\n", " ")
                    text_table += f"| {truth} | {pred} |\n"

                if self.writer:
                    self.writer.add_text("predictions/samples", text_table, step)
                    self.writer.flush()


def compute_metrics(eval_pred: EvalPrediction, tokenizer: Any) -> Dict[str, float]:
    """Compute WER metric and store sample predictions."""
    predictions, label_ids = eval_pred.predictions, eval_pred.label_ids
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    if predictions.ndim == 3:
        predictions = np.argmax(predictions, axis=-1)
    if isinstance(label_ids, tuple):
        label_ids = label_ids[0]
    label_ids = label_ids.copy()
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    sample_predictions = list(zip(decoded_labels[:10], decoded_preds[:10]))

    wer_metric = evaluate.load("wer", cache_dir="./metrics_cache")
    wer = wer_metric.compute(predictions=decoded_preds, references=decoded_labels)

    metrics = {"wer": wer}
    metrics["_sample_predictions"] = sample_predictions  # Prefix with _ to avoid logging as metric
    return metrics


@hydra.main(version_base=None, config_path="../configs/hydra", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function using Hydra configuration."""

    print(OmegaConf.to_yaml(cfg))

    model, tokenizer, feature_extractor = initialize_model(cfg)
    train_dataset, val_dataset = load_datasets(cfg, feature_extractor, tokenizer)

    if cfg.eval_checkpoint:
        print(f"Evaluation mode: Loading checkpoint from {cfg.eval_checkpoint}")
        model = ASRModel.from_pretrained(cfg.eval_checkpoint)

    training_args_dict = OmegaConf.to_container(cfg.training, resolve=True)
    assert isinstance(training_args_dict, dict), "Training args must be a dict"
    compute_metrics_enabled = training_args_dict.pop("compute_metrics", True)
    training_args = TrainingArguments(**training_args_dict)  # type: ignore[arg-type]

    prediction_callback = PredictionLoggingCallback(tokenizer, num_samples=10)

    def compute_metrics_with_samples(eval_pred):
        if not compute_metrics_enabled:
            return None
        metrics = compute_metrics(eval_pred, tokenizer)
        if "_sample_predictions" in metrics:
            trainer.state.sample_predictions = metrics.pop("_sample_predictions")
        return metrics

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollator(
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
        ),
        processing_class=tokenizer,
        compute_metrics=compute_metrics_with_samples,
        callbacks=[prediction_callback],
    )

    if cfg.eval_checkpoint:
        results = trainer.evaluate()
        print(f"Evaluation results: {results}")
    else:
        trainer.train(resume_from_checkpoint=cfg.resume_from_checkpoint)
        trainer.save_model()


if __name__ == "__main__":
    main()
