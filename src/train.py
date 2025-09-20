#!/usr/bin/env python3
"""
🎙️ ASR Training
"""

from typing import Any, Dict, List, Optional, Tuple, Union, cast

import hydra
import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset, load_dataset
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, PeftMixedModel, PeftModel, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedModel,
    Trainer,
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

    def forward(
        self, input_features: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        input_features and attention_mask come directly from the WhisperFeatureExtractor.
        """
        with torch.no_grad():
            input_features = input_features.to(self.whisper.dtype)
            # The model internally handles the attention mask to ignore padding
            outputs = self.whisper.encoder(input_features, attention_mask=attention_mask)
            return cast(torch.Tensor, outputs.last_hidden_state)


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

        result: torch.Tensor = hidden_states * 0.01
        return result


class LLMDecoder(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.model: Union[PreTrainedModel, PeftModel, PeftMixedModel] = (
            AutoModelForCausalLM.from_pretrained(
                config.model.decoder_model_name,
                dtype=torch.bfloat16,
                token=False,
                use_cache=False,  # Disable KV cache for training (incompatible with gradient checkpointing)
            )
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
                "<|audio_chunk|>",  # Placeholder for audio embeddings in instruction
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
        self.audio_chunk_id = self.decoder.tokenizer.convert_tokens_to_ids("<|audio_chunk|>")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        input_values: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        # 1. Get audio embeddings
        audio_features = self.encoder(input_values, attention_mask=encoder_attention_mask)
        audio_embeds = self.audio_projector(audio_features)
        audio_embeds = audio_embeds.to(self.decoder.model.dtype)

        # 2. Get text embeddings
        embed_layer = self.decoder.model.get_input_embeddings()
        text_embeds = embed_layer(input_ids)

        # 3. Find placeholder and insert audio
        final_inputs_embeds = []
        final_labels = []
        for i in range(input_ids.shape[0]):  # Iterate over batch
            chunk_idx = (input_ids[i] == self.audio_chunk_id).nonzero()
            if chunk_idx.shape[0] == 0:
                raise ValueError("'<|audio_chunk|>' token not found in input_ids.")
            chunk_idx = chunk_idx[0].item()

            # Concatenate: [text_before_chunk, audio_embeds, text_after_chunk]
            combined_embeds = torch.cat(
                [
                    text_embeds[i, :chunk_idx],
                    audio_embeds[i],
                    text_embeds[i, chunk_idx + 1 :],
                ],
                dim=0,
            )
            final_inputs_embeds.append(combined_embeds)

            # Adjust labels to match the new sequence length
            # Insert -100 labels for the audio embeddings
            audio_len = audio_embeds[i].shape[0]
            label_before = labels[i, :chunk_idx]
            label_after = labels[i, chunk_idx + 1 :]
            audio_labels = torch.full((audio_len,), -100, dtype=labels.dtype, device=labels.device)
            combined_labels = torch.cat([label_before, audio_labels, label_after], dim=0)
            final_labels.append(combined_labels)

        # Pad the combined embeddings to the same length
        inputs_embeds = torch.nn.utils.rnn.pad_sequence(
            final_inputs_embeds, batch_first=True, padding_value=0
        )
        labels = torch.nn.utils.rnn.pad_sequence(final_labels, batch_first=True, padding_value=-100)

        # Recreate the attention mask for the new embedding sequence length
        new_attention_mask = torch.ones(
            inputs_embeds.shape[:2], device=inputs_embeds.device, dtype=attention_mask.dtype
        )
        # Mask padding positions
        for i, embed in enumerate(final_inputs_embeds):
            new_attention_mask[i, embed.shape[0] :] = 0

        return self.decoder.model(
            inputs_embeds=inputs_embeds,
            attention_mask=new_attention_mask,
            labels=labels,
            return_dict=True,
        )


class DataCollator:
    """Data collator that performs instruction formatting and masking."""

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
                text_len_words = len(text.split())

                if (
                    audio_len_sec <= self.max_audio_seconds
                    and text_len_words <= self.max_text_words
                ):
                    # Normalize to use "text" field
                    if "text" not in f:
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
                "input_values": torch.zeros((1, 80, 3000)),
                "encoder_attention_mask": torch.ones((1, 1500)),
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
            "input_values": audio_features.input_features,
            "encoder_attention_mask": (
                audio_features.attention_mask if hasattr(audio_features, "attention_mask") else None
            ),
        }


def initialize_model(config: DictConfig) -> Tuple[ASRModel, Any, WhisperFeatureExtractor]:
    """Initialize the ASR model, tokenizer, and feature extractor."""
    model = ASRModel(config)
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

    print(OmegaConf.to_yaml(cfg))

    # Enable TensorFloat32 for better performance on Ampere and newer GPUs
    torch.set_float32_matmul_precision("high")

    model, tokenizer, feature_extractor = initialize_model(cfg)
    train_dataset, val_dataset = load_datasets(cfg)

    # Handle checkpoint resuming
    if cfg.resume_from_checkpoint:
        print(f"Resuming training from checkpoint: {cfg.resume_from_checkpoint}")
        # The Trainer will handle loading optimizer and scheduler states

    training_args_dict = OmegaConf.to_container(cfg.training, resolve=True)
    assert isinstance(training_args_dict, dict), "Training args must be a dict"

    training_args = TrainingArguments(**training_args_dict)

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
        processing_class=tokenizer,
        # No compute_metrics - only eval_loss will be calculated
    )

    # Train the model
    trainer.train(resume_from_checkpoint=cfg.resume_from_checkpoint)
    trainer.save_model()


if __name__ == "__main__":
    main()
