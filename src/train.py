#!/usr/bin/env python3
"""
🎙️ Conformer-SmolLM2 ASR - Accelerate Version
Simplified training script using Accelerate for hardware management.
All GPU detection, optimization flags, and distributed training is handled by Accelerate.
"""

import os
import re
import sys
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import evaluate
import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset, load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
    WhisperFeatureExtractor,
    WhisperModel,
)
from transformers.models.llama.modeling_llama import LlamaRMSNorm

# Minimal environment setup - Accelerate handles the rest
# Use local directories if /workspace doesn't exist (e.g., on Mac)
workspace_dir = "/workspace" if os.path.exists("/workspace") else os.path.expanduser("~/.cache")
os.environ["HF_HOME"] = os.environ.get("HF_HOME", workspace_dir)
os.environ["HF_DATASETS_CACHE"] = os.environ.get(
    "HF_DATASETS_CACHE", os.path.join(workspace_dir, "datasets")
)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

warnings.filterwarnings("ignore")


@dataclass
class CustomTrainingArguments(TrainingArguments):
    """Extended TrainingArguments with custom fields for ASR."""

    resume_from_checkpoint: Optional[str] = field(
        default=None, metadata={"help": "Path to checkpoint to resume training from"}
    )


@dataclass
class ModelArguments:
    """
    Unified configuration for all model components.
    """

    # SmolLM2 Config
    decoder_model_name: str = field(
        default="HuggingFaceTB/SmolLM2-360M-Instruct",
        metadata={"help": "The decoder model name or path."},
    )
    use_lora: bool = field(default=True, metadata={"help": "Whether to use LoRA."})
    lora_r: int = field(default=8, metadata={"help": "LoRA attention dimension."})
    lora_alpha: int = field(default=16, metadata={"help": "LoRA scaling factor."})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout."})
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"],
        metadata={"help": "Modules to apply LoRA to."},
    )


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: str = field(
        default="librispeech_asr", metadata={"help": "The name of the dataset to use."}
    )
    dataset_configs: List[str] = field(
        default_factory=lambda: ["clean"],
        metadata={"help": "List of dataset configurations to concatenate."},
    )
    train_splits: List[str] = field(
        default_factory=lambda: ["train.100"],
        metadata={"help": "List of training splits corresponding to dataset_configs."},
    )
    eval_splits: List[str] = field(
        default_factory=lambda: ["validation"],
        metadata={"help": "List of evaluation splits corresponding to dataset_configs."},
    )
    max_audio_seconds: float = field(
        default=20.0, metadata={"help": "Filter out audio samples longer than this."}
    )
    max_text_words: int = field(
        default=150, metadata={"help": "Filter out text samples longer than this."}
    )
    sample_rate: int = field(default=16000, metadata={"help": "Audio sample rate."})
    dataset_cache_dir: str = field(
        default="/workspace/datasets", metadata={"help": "Directory to cache datasets."}
    )
    num_proc: int = field(default=8, metadata={"help": "Number of processes for dataset loading."})
    max_train_samples: Optional[int] = field(
        default=None, metadata={"help": "Maximum number of training samples to use."}
    )
    max_eval_samples: Optional[int] = field(
        default=None, metadata={"help": "Maximum number of evaluation samples to use."}
    )


class ASRConfig(PretrainedConfig):
    """Configuration for ASRModel."""

    model_type = "asr_whisper_smollm2"

    def __init__(self, model_args: Optional[ModelArguments] = None, **kwargs):
        super().__init__(**kwargs)
        if model_args:
            for key, value in model_args.__dict__.items():
                setattr(self, key, value)


class WhisperEncoder(nn.Module):
    """Frozen Whisper encoder wrapper."""

    def __init__(self, config: Union[ModelArguments, ASRConfig]):
        super().__init__()
        # Load Whisper-small model
        self.whisper = WhisperModel.from_pretrained(
            "openai/whisper-small",
            dtype=torch.bfloat16,
            token=False
        )

        for param in self.whisper.parameters():
            param.requires_grad = False

        self.d_model = self.whisper.config.d_model

        self.whisper.eval()

    def forward(self, x: torch.Tensor, input_lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input mel-spectrogram of shape (batch, n_mels, time)
            input_lengths: Lengths of input sequences
        Returns:
            Encoded features of shape (batch, time, d_model)
        """
        batch_size, n_mels, time_frames = x.shape
        original_time_frames = time_frames

        # Whisper expects exactly 3000 frames (30 seconds of audio)
        # This is a requirement of the model architecture
        expected_frames = 3000

        if time_frames < expected_frames:
            # Pad to 3000 frames
            pad_amount = expected_frames - time_frames
            x = torch.nn.functional.pad(x, (0, pad_amount), mode="constant", value=0)
        elif time_frames > expected_frames:
            # Truncate to 3000 frames
            x = x[:, :, :expected_frames]
            original_time_frames = expected_frames

        with torch.no_grad():
            # Ensure input dtype matches the model's dtype
            x = x.to(self.whisper.dtype)
            outputs = self.whisper.encoder(x)
            encoder_outputs = outputs.last_hidden_state

        # Whisper encoder downsamples by factor of 2 (conv stride)
        # So 3000 input frames -> 1500 output frames
        # We need to mask out the padding in the output
        if original_time_frames < expected_frames:
            # Calculate actual output length (Whisper uses stride 2)
            actual_output_frames = (original_time_frames + 1) // 2
            # Only keep the non-padded portion
            encoder_outputs = encoder_outputs[:, :actual_output_frames, :]

        return encoder_outputs


class AudioProjector(nn.Module):
    """Simple 2-layer MLP projector using Pre-Normalization with small initialization."""

    def __init__(self, audio_dim: int, text_dim: int, config: Union[ModelArguments, ASRConfig]):
        super().__init__()
        self.norm = LlamaRMSNorm(audio_dim, eps=1e-6)

        self.linear_1 = nn.Linear(audio_dim, text_dim, bias=True)
        self.act = nn.GELU()
        self.linear_2 = nn.Linear(text_dim, text_dim, bias=True)

        nn.init.normal_(self.linear_1.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.linear_1.bias)
        nn.init.normal_(self.linear_2.weight, mean=0.0, std=0.002) 
        nn.init.zeros_(self.linear_2.bias)

    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        # 1. Normalize the INPUT audio features first
        hidden_states = self.norm(audio_features)

        # 2. Then apply the transformations
        hidden_states = self.linear_1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)

        return hidden_states


class SmolLM2Decoder(nn.Module):
    def __init__(self, config: Union[ModelArguments, ASRConfig]):
        super().__init__()
        self.model: nn.Module = AutoModelForCausalLM.from_pretrained(
            config.decoder_model_name,
            dtype=torch.bfloat16,  # Use bf16 for speed and stability
            token=False
        )
        self.tokenizer = AutoTokenizer.from_pretrained(config.decoder_model_name, token=False)

        # Standard approach: use EOS token as padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Sync model config with tokenizer
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.bos_token_id = self.tokenizer.bos_token_id
        self.model.config.eos_token_id = self.tokenizer.eos_token_id

        # Update generation config if it exists
        if hasattr(self.model, "generation_config"):
            self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
            if self.tokenizer.bos_token_id is not None:
                self.model.generation_config.bos_token_id = self.tokenizer.bos_token_id
            self.model.generation_config.eos_token_id = self.tokenizer.eos_token_id

        # Continue with LoRA setup if needed
        if config.use_lora:
            lora_config = LoraConfig(
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                target_modules=(
                    list(config.lora_target_modules) if config.lora_target_modules else None
                ),
                lora_dropout=config.lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            self.model = get_peft_model(self.model, lora_config)

    def forward(self, **kwargs):
        """Simple forward pass through the decoder model."""
        return self.model(**kwargs)


class ASRModel(PreTrainedModel):
    """ASR model using standard HuggingFace PreTrainedModel."""

    config_class = ASRConfig
    base_model_prefix = "asr"
    supports_gradient_checkpointing = True
    _no_split_modules = ["WhisperEncoder", "SmolLM2Decoder", "AudioProjector"]

    def __init__(self, config: Union[ASRConfig, ModelArguments]) -> None:
        # Convert ModelArguments to ASRConfig if needed
        if isinstance(config, ModelArguments):
            config = ASRConfig(model_args=config)
        super().__init__(config)
        self.model_args = config  # Store for compatibility
        self.encoder = WhisperEncoder(config)
        self.decoder = SmolLM2Decoder(config)
        self.config = self.decoder.model.config
        text_dim = getattr(self.decoder.model.config, "hidden_size", 768)
        audio_dim = self.encoder.d_model  # 768 for whisper-small
        self.audio_projector = AudioProjector(audio_dim, text_dim, config)
        self.add_audio_special_tokens()

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Enable gradient checkpointing for the decoder model only."""
        if hasattr(self.decoder.model, "gradient_checkpointing_enable"):
            self.decoder.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing for the decoder model."""
        if hasattr(self.decoder.model, "gradient_checkpointing_disable"):
            self.decoder.model.gradient_checkpointing_disable()

    def add_audio_special_tokens(self):
        """Add audio-specific special tokens for better audio-text alignment."""
        special_tokens = {
            "additional_special_tokens": [
                "<|audio_start|>",
                "<|audio_end|>",
                "<|audio_pad|>",
                "<|audio_sep|>",  # For potential future multi-audio support
            ]
        }

        num_added = self.decoder.tokenizer.add_special_tokens(special_tokens)

        if num_added > 0:
            self.decoder.model.resize_token_embeddings(len(self.decoder.tokenizer))

            with torch.no_grad():
                embeddings = self.decoder.model.get_input_embeddings()
                if embeddings is not None and hasattr(embeddings, "weight"):
                    # mean_embedding = embeddings.weight[:-num_added].mean(dim=0)  # unused
                    std_embedding = embeddings.weight[:-num_added].std()

                    for i in range(num_added):
                        # Initialize near zero with very small variance
                        embeddings.weight[-num_added + i] = torch.randn_like(
                            embeddings.weight[0]
                        ) * (
                            std_embedding * 0.02
                        )  # Much smaller scale

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
        """Forward pass with standard HuggingFace patterns."""
        # Extract inputs from kwargs if needed (standard HF pattern)
        if input_values is None:
            input_values = kwargs.get("input_values")
        if input_lengths is None:
            input_lengths = kwargs.get("input_lengths")

        if input_values is None or input_lengths is None:
            raise ValueError("input_values and input_lengths are required")

        # Debug first forward pass
        if not hasattr(self, '_first_forward_logged'):
            print("\n" + "="*80)
            print("🧠 MODEL FORWARD PASS - FIRST BATCH")
            print("="*80)
            print(f"Input values shape: {input_values.shape}")
            print(f"Input lengths: {input_lengths.tolist()}")
            if labels is not None:
                print(f"Labels shape: {labels.shape}")
                print(f"Labels sample (first 30 tokens): {labels[0][:30].tolist()}")
                # Decode to check text
                if hasattr(self.decoder, 'tokenizer'):
                    decoded = self.decoder.tokenizer.decode(labels[0], skip_special_tokens=False)
                    print(f"Decoded labels: '{decoded[:100]}...'")
            if attention_mask is not None:
                print(f"Attention mask shape: {attention_mask.shape}")
                print(f"Attention mask has zeros (padding): {(attention_mask == 0).any().item()}")
            print("="*80 + "\n")
            self._first_forward_logged = True

        # Encode audio features
        audio_features = self.encoder(input_values, input_lengths)
        audio_embeds = self.audio_projector(audio_features)

        # Debug encoder output on first pass
        if not hasattr(self, '_encoder_output_logged'):
            print("\n" + "="*80)
            print("🎤 ENCODER OUTPUT")
            print("="*80)
            print(f"Audio features shape: {audio_features.shape}")
            print(f"Audio embeds shape (after projection): {audio_embeds.shape}")
            print(f"Audio embeds min/max: {audio_embeds.min().item():.4f} / {audio_embeds.max().item():.4f}")
            print("="*80 + "\n")
            self._encoder_output_logged = True

        # Prepare embeddings with special tokens
        batch_size, audio_seq_len, hidden_dim = audio_embeds.shape
        device = audio_embeds.device

        # Get embedding layer for special tokens
        embed_layer = self.decoder.model.get_input_embeddings()

        # Create special token embeddings
        audio_start_tokens = torch.full(
            (batch_size, 1), self.audio_start_id, device=device, dtype=torch.long
        )
        audio_end_tokens = torch.full(
            (batch_size, 1), self.audio_end_id, device=device, dtype=torch.long
        )

        audio_start_embeds = embed_layer(audio_start_tokens)
        audio_end_embeds = embed_layer(audio_end_tokens)

        if labels is not None:
            # Training mode: combine audio and text
            text_embeds = embed_layer(labels)

            # Ensure dtype consistency
            audio_embeds = audio_embeds.to(text_embeds.dtype)

            # Combine embeddings: [audio_start, audio_features, audio_end, text]
            inputs_embeds = torch.cat(
                [audio_start_embeds, audio_embeds, audio_end_embeds, text_embeds], dim=1
            )

            # Create attention mask
            audio_len = audio_seq_len + 2  # +2 for special tokens
            if attention_mask is not None:
                # Extend attention mask for audio tokens
                audio_mask = torch.ones(
                    batch_size, audio_len, dtype=attention_mask.dtype, device=device
                )
                attention_mask = torch.cat([audio_mask, attention_mask], dim=1)
            else:
                attention_mask = torch.ones(
                    inputs_embeds.shape[:2], dtype=torch.long, device=device
                )

            # Prepare labels: mask audio tokens with -100 (ignore index)
            audio_labels = torch.full(
                (batch_size, audio_len), -100, dtype=labels.dtype, device=device
            )
            labels = torch.cat([audio_labels, labels], dim=1)
        else:
            # Inference mode: audio only
            inputs_embeds = torch.cat([audio_start_embeds, audio_embeds, audio_end_embeds], dim=1)
            attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=device)
            labels = None

        # Debug what's being sent to decoder on first pass
        if not hasattr(self, '_decoder_input_logged'):
            print("\n" + "="*80)
            print("📤 DECODER INPUT")
            print("="*80)
            print(f"Inputs embeds shape: {inputs_embeds.shape}")
            print(f"Attention mask shape: {attention_mask.shape}")
            if labels is not None:
                print(f"Labels shape: {labels.shape}")
                # Show how labels align with inputs
                print(f"\nAlignment check:")
                print(f"  Audio tokens: {audio_len if 'audio_len' in locals() else 'N/A'}")
                print(f"  Text tokens: {labels.shape[1] - audio_len if 'audio_len' in locals() and labels is not None else 'N/A'}")
                print(f"  Total sequence: {inputs_embeds.shape[1]}")
                # Check label masking
                masked_positions = (labels == -100).sum(dim=1)
                print(f"  Masked positions per sample: {masked_positions.tolist()}")
            print("="*80 + "\n")
            self._decoder_input_logged = True

        # Forward through decoder
        outputs = self.decoder.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )

        return outputs

    @torch.no_grad()
    def generate(
        self,
        input_values: torch.Tensor,
        input_lengths: torch.Tensor,
        max_new_tokens: int = 100,
        **kwargs,
    ) -> torch.Tensor:
        """Generate text from audio input using standard HF generation."""
        # Get audio embeddings
        audio_features = self.encoder(input_values, input_lengths)
        audio_embeds = self.audio_projector(audio_features)

        batch_size = audio_embeds.shape[0]
        device = audio_embeds.device
        embed_layer = self.decoder.model.get_input_embeddings()

        # Create special token embeddings
        audio_start_tokens = torch.full(
            (batch_size, 1), self.audio_start_id, device=device, dtype=torch.long
        )
        audio_end_tokens = torch.full(
            (batch_size, 1), self.audio_end_id, device=device, dtype=torch.long
        )

        audio_start_embeds = embed_layer(audio_start_tokens)
        audio_end_embeds = embed_layer(audio_end_tokens)

        # Combine embeddings for generation
        inputs_embeds = torch.cat([audio_start_embeds, audio_embeds, audio_end_embeds], dim=1)

        return self.decoder.model.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.decoder.tokenizer.pad_token_id,
            eos_token_id=self.decoder.tokenizer.eos_token_id,
            **kwargs,
        )

    def transcribe(
        self, input_values: torch.Tensor, input_lengths: torch.Tensor, **kwargs
    ) -> List[str]:
        """
        Convenience method to generate and decode transcriptions.

        Args:
            input_values: Mel-spectrogram input
            input_lengths: Input lengths
            **kwargs: Generation parameters

        Returns:
            List of decoded transcription strings
        """
        # Generate tokens
        generated_tokens = self.generate(input_values, input_lengths, **kwargs)

        # Decode to text, skipping special tokens
        transcriptions = self.decoder.tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

        return transcriptions

    def save_pretrained(self, save_directory: str, **kwargs):
        """Save model using HuggingFace standard format."""
        # Save config
        if hasattr(self, "model_args") and isinstance(self.model_args, ModelArguments):
            # Convert ModelArguments to config for saving
            config = ASRConfig(model_args=self.model_args)
            config.save_pretrained(save_directory)
        else:
            super().save_pretrained(save_directory, **kwargs)

        # Save tokenizer
        if hasattr(self.decoder, "tokenizer"):
            self.decoder.tokenizer.save_pretrained(save_directory)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        """Load model using HuggingFace standard format."""
        # Load config
        config = ASRConfig.from_pretrained(pretrained_model_name_or_path)

        # Create model
        model = cls(config)

        # Load state dict
        import os

        from safetensors.torch import load_file

        model_file = os.path.join(pretrained_model_name_or_path, "model.safetensors")
        if os.path.exists(model_file):
            state_dict = load_file(model_file)
        else:
            model_file = os.path.join(pretrained_model_name_or_path, "pytorch_model.bin")
            if os.path.exists(model_file):
                state_dict = torch.load(model_file, map_location="cpu")
            else:
                raise FileNotFoundError(
                    f"No model weights found in {pretrained_model_name_or_path}"
                )

        model.load_state_dict(state_dict, strict=False)
        return model


@dataclass
class DataCollator:
    """Data collator that performs preprocessing on-the-fly."""

    tokenizer: AutoTokenizer
    feature_extractor: WhisperFeatureExtractor
    sample_rate: int = 16000
    max_audio_seconds: float = 20.0
    max_text_words: int = 150

    def __call__(self, features: List[Dict[str, Union[str, Dict]]]) -> Dict[str, torch.Tensor]:
        # Filter samples that are too long
        valid_features = []
        for i, f in enumerate(features):
            try:
                # Audio is decoded lazily when accessed
                audio_array = f["audio"]["array"]

                # Debug first batch (after successful audio access)
                if i == 0 and not hasattr(self, '_first_batch_logged'):
                    print("\n" + "="*80)
                    print("🔍 DEBUGGING FIRST BATCH")
                    print("="*80)
                    print(f"Number of samples in batch: {len(features)}")
                    print(f"Sample keys: {f.keys()}")
                    print(f"Audio array shape: {np.array(audio_array).shape}")
                    print(f"Audio array dtype: {np.array(audio_array).dtype}")
                    print(f"Audio array min/max: {np.min(audio_array):.4f} / {np.max(audio_array):.4f}")
                    print(f"Audio sample rate: {f['audio']['sampling_rate']}")
                    if 'text' in f:
                        print(f"Text sample: '{f['text'][:100]}...'")
                    print("="*80 + "\n")
                    self._first_batch_logged = True

                audio_len_sec = len(audio_array) / self.sample_rate
                text_len_words = len(f["text"].split())
                if (
                    audio_len_sec <= self.max_audio_seconds
                    and text_len_words <= self.max_text_words
                ):
                    valid_features.append(f)
            except Exception as e:
                print(f"⚠️ Error processing sample {i}: {e}")
                import traceback
                traceback.print_exc()
                continue

        if not valid_features:
            # Return dummy batch if all samples filtered
            pad_token_id = (
                self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            )
            dummy_seq_len = 10  # Reasonable sequence length for dummy batch
            return {
                "input_values": torch.zeros((1, self.n_mels, 100)),
                "input_lengths": torch.tensor([100]),
                "labels": torch.full((1, dummy_seq_len), pad_token_id, dtype=torch.long),
                "attention_mask": torch.ones((1, dummy_seq_len), dtype=torch.long),
            }

        # Extract audio arrays and texts
        audio_arrays = []
        texts = []
        for f in valid_features:
            audio_arrays.append(f["audio"]["array"])
            texts.append(f["text"])  # Use original text without normalization

        # Use WhisperFeatureExtractor to process audio
        # It automatically handles mel spectrogram generation, normalization, and padding to 3000 frames
        audio_features = self.feature_extractor(
            audio_arrays,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True,  # Pad to longest in batch (will be 3000 for Whisper)
        )

        # Debug first batch processing
        if not hasattr(self, '_first_audio_processed'):
            print("\n" + "="*80)
            print("🎵 PROCESSING AUDIO WITH WHISPER FEATURE EXTRACTOR")
            print("="*80)
            print(f"Number of audio samples: {len(audio_arrays)}")
            print(f"First audio shape: {np.array(audio_arrays[0]).shape}")
            print(f"Feature extractor output shape: {audio_features.input_features.shape}")
            print(f"Feature min/max: {audio_features.input_features.min().item():.4f} / {audio_features.input_features.max().item():.4f}")
            print("="*80 + "\n")
            self._first_audio_processed = True

        # Get the padded mel spectrograms
        padded_specs = audio_features.input_features

        # Calculate actual input lengths (before padding)
        # Since feature extractor pads to 3000, we need to track actual lengths
        input_lengths = torch.tensor(
            [min(len(arr) // 160, 3000) for arr in audio_arrays],  # 160 = hop_length
            dtype=torch.long
        )

        # Tokenize and pad text labels
        labels = self.tokenizer(texts, padding="longest", truncation=True, return_tensors="pt")

        output_batch = {
            "input_values": padded_specs,
            "input_lengths": input_lengths,
            "labels": labels["input_ids"],
            "attention_mask": labels["attention_mask"],
        }

        # Debug final batch output - comprehensive audio-text pairing check
        if not hasattr(self, '_first_batch_output_logged'):
            print("\n" + "="*80)
            print("📦 FINAL BATCH OUTPUT - AUDIO-TEXT PAIRS & TOKENS")
            print("="*80)
            print(f"Batch size: {len(texts)}")
            print(f"Batch keys: {output_batch.keys()}")
            print(f"\n🎵 AUDIO INPUT:")
            print(f"  Input values shape: {output_batch['input_values'].shape}")
            print(f"  Input values dtype: {output_batch['input_values'].dtype}")
            print(f"  Input values min/max: {output_batch['input_values'].min().item():.4f} / {output_batch['input_values'].max().item():.4f}")
            print(f"  Input lengths (frame counts): {output_batch['input_lengths'].tolist()}")

            print(f"\n📝 TEXT LABELS:")
            print(f"  Labels shape: {output_batch['labels'].shape}")
            print(f"  Attention mask shape: {output_batch['attention_mask'].shape}")

            # Show first few audio-text pairs
            print(f"\n🔗 AUDIO-TEXT PAIRS (first 3):")
            for i in range(min(3, len(texts))):
                print(f"\n  Sample {i}:")
                print(f"    Original text: '{texts[i][:100]}...'")
                print(f"    Token IDs: {output_batch['labels'][i][:30].tolist()}...")
                decoded_text = self.tokenizer.decode(output_batch['labels'][i], skip_special_tokens=False)
                print(f"    Decoded tokens: '{decoded_text[:100]}...'")
                print(f"    Audio frames: {output_batch['input_lengths'][i].item()}")
                print(f"    Audio duration: ~{output_batch['input_lengths'][i].item() * 0.01:.2f} seconds")  # 10ms per frame

            # Check for special tokens
            print(f"\n🔤 TOKENIZER INFO:")
            print(f"  Pad token ID: {self.tokenizer.pad_token_id}")
            print(f"  EOS token ID: {self.tokenizer.eos_token_id}")
            print(f"  BOS token ID: {self.tokenizer.bos_token_id}")

            # Check if padding is working correctly
            print(f"\n✅ VALIDATION CHECKS:")
            has_padding = (output_batch['labels'] == self.tokenizer.pad_token_id).any()
            print(f"  Labels contain padding: {has_padding}")
            print(f"  All audio inputs same shape: {len(set(output_batch['input_values'].shape[2:3])) == 1}")
            print(f"  All labels same length: {len(set([len(l) for l in output_batch['labels']])) == 1}")

            print("="*80 + "\n")
            self._first_batch_output_logged = True

        return output_batch


def parse_config(config_file: str) -> Tuple[ModelArguments, DataArguments, CustomTrainingArguments]:
    """Parse configuration from JSON file."""
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found: {config_file}")

    parser = HfArgumentParser((ModelArguments, DataArguments, CustomTrainingArguments))
    model_args, data_args, training_args = parser.parse_json_file(
        json_file=config_file, allow_extra_keys=True
    )

    # Essential settings for ASR
    training_args.remove_unused_columns = False
    training_args.label_names = ["labels"]

    return model_args, data_args, training_args


def initialize_model(model_args: ModelArguments) -> Tuple[ASRModel, AutoTokenizer, WhisperFeatureExtractor]:
    """Initialize the ASR model, tokenizer, and feature extractor."""
    model = ASRModel(model_args)
    tokenizer = model.decoder.tokenizer
    # Load WhisperFeatureExtractor for proper audio preprocessing
    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
    return model, tokenizer, feature_extractor


def load_datasets(data_args: DataArguments) -> Tuple[Dataset, Dataset]:
    """Load training and validation datasets using standard HuggingFace patterns."""
    import platform

    from datasets import Audio, concatenate_datasets

    safe_num_proc = 1 if platform.system() == "Darwin" else data_args.num_proc
    dataset_dicts = []
    for config, train_split, eval_split in zip(
        data_args.dataset_configs, data_args.train_splits, data_args.eval_splits
    ):
        # Load dataset
        ds_dict = load_dataset(
            data_args.dataset_name,
            config,
            split={"train": train_split, "validation": eval_split},
            cache_dir=data_args.dataset_cache_dir,
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
    train_dataset = train_dataset.cast_column(
        "audio", Audio(sampling_rate=data_args.sample_rate)
    )
    val_dataset = val_dataset.cast_column(
        "audio", Audio(sampling_rate=data_args.sample_rate)
    )

    if data_args.max_train_samples:
        train_dataset = train_dataset.select(
            range(min(data_args.max_train_samples, len(train_dataset)))
        )
    if data_args.max_eval_samples:
        val_dataset = val_dataset.select(range(min(data_args.max_eval_samples, len(val_dataset))))

    return train_dataset, val_dataset


def compute_metrics(eval_pred: EvalPrediction, tokenizer: AutoTokenizer) -> Dict[str, float]:
    """Compute WER metric using standard HuggingFace patterns."""
    predictions, label_ids = eval_pred.predictions, eval_pred.label_ids
    if predictions.ndim == 3:
        predictions = np.argmax(predictions, axis=-1)
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    wer_metric = evaluate.load("wer", cache_dir="./metrics_cache")
    wer = wer_metric.compute(predictions=decoded_preds, references=decoded_labels)

    return {"wer": wer}


def evaluate_checkpoint(checkpoint_dir: str, config_file: str) -> None:
    """Evaluate a checkpoint using standard HuggingFace patterns."""
    model_args, data_args, training_args = parse_config(config_file)
    training_args.do_train = False
    training_args.do_eval = True
    training_args.per_device_eval_batch_size = 8
    model = ASRModel.from_pretrained(checkpoint_dir)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")

    _, val_dataset = load_datasets(data_args)
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=val_dataset,
        data_collator=DataCollator(
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
            sample_rate=data_args.sample_rate,
            max_audio_seconds=data_args.max_audio_seconds,
            max_text_words=data_args.max_text_words,
        ),
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, tokenizer),
    )

    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)


def main(config_file: str) -> None:
    """Main training function using standard HuggingFace patterns."""
    model_args, data_args, training_args = parse_config(config_file)
    model, tokenizer, feature_extractor = initialize_model(model_args)

    train_dataset, val_dataset = load_datasets(data_args)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollator(
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
            sample_rate=data_args.sample_rate,
            max_audio_seconds=data_args.max_audio_seconds,
            max_text_words=data_args.max_text_words,
        ),
        tokenizer=tokenizer,
        compute_metrics=None,  # Disabled for faster evaluation
    )

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python train.py <config.json> [--eval <checkpoint_dir>]")
        print("Examples:")
        print("  Training: python train.py configs/experiment.json")
        print("  Evaluation: python train.py configs/experiment.json --eval ./checkpoint-100")
        sys.exit(1)

    config_file = sys.argv[1]
    if len(sys.argv) >= 4 and sys.argv[2] == "--eval":
        checkpoint_dir = sys.argv[3]
        evaluate_checkpoint(checkpoint_dir, config_file)
    else:
        main(config_file)
