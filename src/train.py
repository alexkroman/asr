#!/usr/bin/env python3
"""
🎙️ Conformer-SmolLM2 ASR - Accelerate Version
Simplified training script using Accelerate for hardware management.
All GPU detection, optimization flags, and distributed training is handled by Accelerate.
"""

import os
import random
import re
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import evaluate
import numpy as np
import torch
import torch.nn as nn
import torchaudio.transforms as T
from accelerate import Accelerator
from datasets import Dataset, load_dataset
from einops import rearrange
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)

# Minimal environment setup - Accelerate handles the rest
# Use local directories if /workspace doesn't exist (e.g., on Mac)
workspace_dir = "/workspace" if os.path.exists("/workspace") else os.path.expanduser("~/.cache")
os.environ["HF_HOME"] = os.environ.get("HF_HOME", workspace_dir)
os.environ["HF_DATASETS_CACHE"] = os.environ.get(
    "HF_DATASETS_CACHE", os.path.join(workspace_dir, "datasets")
)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

warnings.filterwarnings("ignore")

# Data path for outputs - use local directory on Mac
DATA_PATH = (
    "/workspace/ASR_Conformer_SmolLM2_Optimized" if os.path.exists("/workspace") else "./ASR_output"
)
os.makedirs(f"{DATA_PATH}/checkpoints", exist_ok=True)
os.makedirs(f"{DATA_PATH}/models", exist_ok=True)
os.makedirs(f"{DATA_PATH}/logs", exist_ok=True)

# Accelerator will be initialized in main function

# Handle Hugging Face authentication
hf_write_token = os.environ.get("HF_WRITE_TOKEN") or os.environ.get("HF_TOKEN")
hf_read_token = os.environ.get("HF_READ_TOKEN")

# Authentication will be handled in main function


@dataclass
class CustomTrainingArguments(TrainingArguments):
    """Extended TrainingArguments with custom fields for testing."""

    test_checkpoint_loading: bool = field(
        default=False, metadata={"help": "Whether to test checkpoint loading"}
    )
    resume_from_checkpoint: Optional[str] = field(
        default=None, metadata={"help": "Path to checkpoint to resume training from"}
    )


@dataclass
class ModelArguments:
    """
    Unified configuration for all model components.
    """

    # Conformer Config
    n_mels: int = field(default=80, metadata={"help": "Number of Mel bands."})
    d_model: int = field(default=512, metadata={"help": "Dimension of the model."})
    n_head: int = field(default=8, metadata={"help": "Number of attention heads."})
    num_layers: int = field(default=12, metadata={"help": "Number of encoder layers."})
    kernel_size: int = field(default=15, metadata={"help": "Kernel size for Conformer."})
    conformer_dropout: float = field(default=0.1, metadata={"help": "Dropout for Conformer."})

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

    # Projector Config
    num_queries: int = field(default=24, metadata={"help": "Number of queries for projector."})
    projector_num_heads: int = field(default=8, metadata={"help": "Number of heads for projector."})
    projector_dropout: float = field(default=0.1, metadata={"help": "Dropout for projector."})


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: str = field(
        default="librispeech_asr", metadata={"help": "The name of the dataset to use."}
    )
    dataset_config_name: str = field(
        default="clean", metadata={"help": "The configuration name of the dataset."}
    )
    train_split: str = field(default="train.100", metadata={"help": "The training split to use."})
    eval_split: str = field(default="validation", metadata={"help": "The evaluation split to use."})
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


class SpecAugment(nn.Module):
    def __init__(
        self,
        freq_mask_param: int = 27,
        time_mask_param: int = 100,
        n_freq_masks: int = 2,
        n_time_masks: int = 2,
    ) -> None:
        super().__init__()
        self.freq_masks = nn.ModuleList(
            [T.FrequencyMasking(freq_mask_param=freq_mask_param) for _ in range(n_freq_masks)]
        )
        self.time_masks = nn.ModuleList(
            [T.TimeMasking(time_mask_param=time_mask_param) for _ in range(n_time_masks)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for freq_mask in self.freq_masks:
            x = freq_mask(x)
        for time_mask in self.time_masks:
            x = time_mask(x)
        return x


class ConvolutionModule(nn.Module):
    """Convolution module for Conformer block."""

    def __init__(self, d_model: int, kernel_size: int = 31, dropout: float = 0.1):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd for 'same' padding"

        self.layer_norm = nn.LayerNorm(d_model)
        self.pointwise_conv1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            groups=d_model,
        )
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.activation = nn.SiLU()
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, time, d_model)
        Returns:
            Output tensor of shape (batch, time, d_model)
        """
        residual = x
        x = self.layer_norm(x)

        # Transpose for conv1d operations: (batch, time, d_model) -> (batch, d_model, time)
        x = x.transpose(1, 2)

        # Pointwise conv -> GLU
        x = self.pointwise_conv1(x)
        x = self.glu(x)

        # Depthwise conv
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)

        # Second pointwise conv
        x = self.pointwise_conv2(x)
        x = self.dropout(x)

        # Transpose back: (batch, d_model, time) -> (batch, time, d_model)
        x = x.transpose(1, 2)

        return residual + x


class FeedForwardModule(nn.Module):
    """Feed-forward module for Conformer block."""

    def __init__(self, d_model: int, d_ff: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        d_ff = d_ff or 4 * d_model

        self.layer_norm = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.SiLU()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, time, d_model)
        Returns:
            Output tensor of shape (batch, time, d_model)
        """
        residual = x
        x = self.layer_norm(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        return residual + 0.5 * x  # Half-step residual as in Conformer paper


class ConformerBlock(nn.Module):
    """Single Conformer block with the characteristic sandwich structure."""

    def __init__(
        self,
        d_model: int,
        n_head: int,
        d_ff: Optional[int] = None,
        kernel_size: int = 31,
        dropout: float = 0.1,
    ):
        super().__init__()

        # First feed-forward module
        self.ff1 = FeedForwardModule(d_model, d_ff, dropout)

        # Multi-head self-attention module
        self.self_attn_layer_norm = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.attn_dropout = nn.Dropout(dropout)

        # Convolution module
        self.conv_module = ConvolutionModule(d_model, kernel_size, dropout)

        # Second feed-forward module
        self.ff2 = FeedForwardModule(d_model, d_ff, dropout)

        # Final layer norm
        self.final_layer_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, time, d_model)
            key_padding_mask: Padding mask of shape (batch, time)
        Returns:
            Output tensor of shape (batch, time, d_model)
        """
        # First feed-forward
        x = self.ff1(x)

        # Multi-head self-attention with residual
        residual = x
        x = self.self_attn_layer_norm(x)
        x, _ = self.self_attn(x, x, x, key_padding_mask=key_padding_mask)
        x = self.attn_dropout(x)
        x = residual + x

        # Convolution module
        x = self.conv_module(x)

        # Second feed-forward
        x = self.ff2(x)

        # Final layer norm
        x = self.final_layer_norm(x)

        return x


class ConformerEncoder(nn.Module):
    """True Conformer encoder with proper architecture."""

    def __init__(self, config: ModelArguments):
        super().__init__()
        self.config = config
        self.gradient_checkpointing = False  # Track gradient checkpointing state

        # Convolutional subsampling with stride 2 for each layer
        # Total subsampling factor = 2^num_subsample_layers
        self.subsample_layers = 2  # Number of conv layers with stride 2
        self.subsample_factor = 2**self.subsample_layers  # Total subsampling: 4x

        # Convolutional subsampling
        self.subsample = nn.Sequential(
            nn.Conv2d(1, config.d_model, 3, 2, 1),
            nn.SiLU(),
            nn.Conv2d(config.d_model, config.d_model, 3, 2, 1),
            nn.SiLU(),
        )

        # Linear projection after subsampling
        # Calculate feature dimension after frequency subsampling
        freq_subsample_factor = (
            self.subsample_factor
        )  # Frequency dimension also reduced by same factor
        self.input_proj = nn.Linear(
            config.d_model * (config.n_mels // freq_subsample_factor), config.d_model
        )
        self.dropout = nn.Dropout(config.conformer_dropout)

        # Stack of Conformer blocks
        self.conformer_blocks = nn.ModuleList(
            [
                ConformerBlock(
                    d_model=config.d_model,
                    n_head=config.n_head,
                    d_ff=config.d_model * 4,
                    kernel_size=config.kernel_size,
                    dropout=config.conformer_dropout,
                )
                for _ in range(config.num_layers)
            ]
        )

    def forward(self, x: torch.Tensor, input_lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input mel-spectrogram of shape (batch, n_mels, time)
            input_lengths: Lengths of input sequences
        Returns:
            Encoded features of shape (batch, time, d_model)
        """
        # Ensure tensor is contiguous for MPS compatibility
        x = x.contiguous()

        # Convolutional subsampling
        x = self.subsample(x.unsqueeze(1))
        x = rearrange(x, "b c f t -> b t (c f)")
        x = self.input_proj(x)
        x = self.dropout(x)

        # Calculate output lengths after subsampling
        output_lengths = input_lengths // self.subsample_factor
        max_len = x.size(1)

        # Create padding mask
        key_padding_mask = (
            torch.arange(max_len, device=x.device)[None, :] >= output_lengths[:, None]
        )

        # Apply Conformer blocks with optional gradient checkpointing
        for conformer_block in self.conformer_blocks:
            if self.gradient_checkpointing and self.training:
                # Use gradient checkpointing to save memory during training
                from torch.utils.checkpoint import checkpoint

                x = checkpoint(conformer_block, x, key_padding_mask, use_reentrant=False)
            else:
                x = conformer_block(x, key_padding_mask)

        return x


class LightweightAudioProjector(nn.Module):
    def __init__(self, audio_dim: int, text_dim: int, config: ModelArguments):
        super().__init__()
        self.audio_proj = nn.Linear(audio_dim, text_dim)
        self.queries = nn.Parameter(torch.randn(config.num_queries, text_dim))
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=text_dim,
            num_heads=config.projector_num_heads,
            dropout=config.projector_dropout,
            batch_first=True,
        )
        self.mlp = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, text_dim * 2),
            nn.GELU(),
            nn.Dropout(config.projector_dropout),
            nn.Linear(text_dim * 2, text_dim),
            nn.LayerNorm(text_dim),
        )

    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        B = audio_features.shape[0]
        audio_proj = self.audio_proj(audio_features)
        queries = self.queries.unsqueeze(0).expand(B, -1, -1)
        attn_out, _ = self.cross_attn(queries, audio_proj, audio_proj)
        result: torch.Tensor = self.mlp(attn_out + queries)
        return result


class SmolLM2Decoder(nn.Module):
    def __init__(self, config: ModelArguments):
        super().__init__()
        self.model: nn.Module = AutoModelForCausalLM.from_pretrained(
            config.decoder_model_name, dtype=torch.bfloat16, token=False
        )
        self.tokenizer = AutoTokenizer.from_pretrained(config.decoder_model_name, token=False)
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "<pad>"})
            self.model.resize_token_embeddings(len(self.tokenizer), mean_resizing=False)
            with torch.no_grad():
                embeddings = self.model.get_input_embeddings()
                if embeddings is not None and hasattr(embeddings, "weight"):
                    embedding_weight: torch.nn.Parameter = embeddings.weight  # type: ignore
                    embedding_weight.data[-1] = embedding_weight.data[:-1].mean(dim=0)
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
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
            if hasattr(self.model, "print_trainable_parameters"):
                self.model.print_trainable_parameters()


class CustomTrainer(Trainer):
    """Custom trainer that handles tied embeddings properly during saving."""

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        """Override save to handle tied embeddings in the decoder."""
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)

        # Save the model using save_model which handles tied embeddings
        if self.model is not None:
            self._save_model(output_dir, state_dict)

        # Save tokenizer
        if self.tokenizer is not None and output_dir is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Save training args
        if output_dir is not None:
            torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

    def _save_model(self, output_dir, state_dict=None):
        """Save model handling tied embeddings properly."""
        # Use the model's save_pretrained if available (for HF models)
        if hasattr(self.model, "decoder") and hasattr(self.model.decoder, "model") and hasattr(self.model.decoder.model, "save_pretrained"):
            # Save decoder with tied embeddings handled properly
            if output_dir is not None:
                decoder_path = os.path.join(output_dir, "decoder")
                self.model.decoder.model.save_pretrained(decoder_path)

            # Save encoder and projector separately
            if output_dir is not None and hasattr(self.model, "encoder"):
                torch.save(self.model.encoder.state_dict(), os.path.join(output_dir, "encoder.bin"))
            if output_dir is not None and hasattr(self.model, "audio_projector"):
                torch.save(
                    self.model.audio_projector.state_dict(), os.path.join(output_dir, "projector.bin")
                )
        else:
            # Fallback to standard save
            super()._save(output_dir, state_dict)


class ASRModel(nn.Module):
    def __init__(self, config: ModelArguments) -> None:
        super().__init__()
        self.encoder = ConformerEncoder(config)
        self.decoder = SmolLM2Decoder(config)
        text_dim = getattr(self.decoder.model.config, "hidden_size", 768)
        self.audio_projector = LightweightAudioProjector(config.d_model, text_dim, config)
        self.spec_augment = SpecAugment()

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Enable gradient checkpointing for both encoder and decoder models."""
        # Enable for encoder
        self.encoder.gradient_checkpointing = True

        # Enable for decoder
        if hasattr(self.decoder.model, "gradient_checkpointing_enable"):
            self.decoder.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing for both encoder and decoder models."""
        # Disable for encoder
        self.encoder.gradient_checkpointing = False

        # Disable for decoder
        if hasattr(self.decoder.model, "gradient_checkpointing_disable"):
            self.decoder.model.gradient_checkpointing_disable()

    def forward(
        self,
        input_values: Optional[torch.Tensor] = None,
        input_lengths: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        # Handle dict input from Trainer
        if input_values is None and "input_values" in kwargs:
            input_values = kwargs["input_values"]
        if input_lengths is None and "input_lengths" in kwargs:
            input_lengths = kwargs["input_lengths"]

        if input_values is None or input_lengths is None:
            raise ValueError("input_values and input_lengths are required")

        if self.training:
            input_values = self.spec_augment(input_values)

        # Use SDPA optimization only on CUDA devices
        if input_values.is_cuda:
            with torch.backends.cuda.sdp_kernel(
                enable_flash=True, enable_math=False, enable_mem_efficient=True
            ):
                audio_features = self.encoder(input_values, input_lengths)
        else:
            # For CPU, MPS, or other devices, use default attention implementation
            audio_features = self.encoder(input_values, input_lengths)

        audio_prefix = self.audio_projector(audio_features)

        embeddings = self.decoder.model.get_input_embeddings()
        if labels is not None and callable(embeddings):
            text_embeds = embeddings(labels)
            combined_embeds = torch.cat([audio_prefix, text_embeds], dim=1)
        else:
            combined_embeds = audio_prefix

        audio_mask = torch.ones(
            audio_prefix.shape[:2], dtype=torch.long, device=input_values.device
        )
        if attention_mask is not None and labels is not None:
            combined_attention_mask = torch.cat([audio_mask, attention_mask], dim=1)
        else:
            combined_attention_mask = audio_mask

        if labels is not None:
            prefix_labels = torch.full(
                audio_prefix.shape[:2],
                fill_value=-100,
                dtype=labels.dtype,
                device=labels.device,
            )
            combined_labels = torch.cat([prefix_labels, labels], dim=1)
        else:
            combined_labels = None

        outputs = self.decoder.model.forward(
            inputs_embeds=combined_embeds,
            attention_mask=combined_attention_mask,
            labels=combined_labels,
        )

        # Only return loss and logits to avoid DynamicCache padding issues
        if hasattr(outputs, "loss") and hasattr(outputs, "logits"):
            from transformers.modeling_outputs import CausalLMOutputWithPast

            return CausalLMOutputWithPast(
                loss=outputs.loss,
                logits=outputs.logits,
                past_key_values=None,  # Don't return cache
                hidden_states=None,
                attentions=None,
            )
        return outputs

    @torch.inference_mode()
    def generate(
        self,
        input_values: torch.Tensor,
        input_lengths: torch.Tensor,
        **kwargs: Dict[str, Union[int, float, torch.Tensor]],
    ) -> torch.Tensor:
        # Process through encoder (encoder handles its own dtype internally)
        audio_features = self.encoder(input_values, input_lengths)

        # Project audio features to text embedding dimension
        audio_prefix = self.audio_projector(audio_features)

        # Ensure audio_prefix matches decoder's dtype (explicitly set to bfloat16 in __init__)
        decoder_dtype = next(self.decoder.model.parameters()).dtype
        audio_prefix = audio_prefix.to(dtype=decoder_dtype)

        return self.decoder.model.generate(inputs_embeds=audio_prefix, **kwargs)


@dataclass
class DataCollator:
    """Data collator that performs preprocessing on-the-fly."""

    tokenizer: AutoTokenizer
    sample_rate: int = 16000
    n_mels: int = 80
    max_audio_seconds: float = 20.0
    max_text_words: int = 150

    def __post_init__(self) -> None:
        self.mel_transform = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=512,
            win_length=400,
            hop_length=160,
        )
        self.amp_to_db = T.AmplitudeToDB(stype="magnitude", top_db=80)

    def _normalize_text(self, text: str) -> str:
        return re.sub(r"[^\w\s'\-]", "", text.lower().strip())

    def __call__(self, features: List[Dict[str, Union[str, Dict]]]) -> Dict[str, torch.Tensor]:
        # Filter samples that are too long
        valid_features = []
        for f in features:
            try:
                audio_len_sec = len(f["audio"]["array"]) / self.sample_rate
                text_len_words = len(self._normalize_text(f["text"]).split())
                if (
                    audio_len_sec <= self.max_audio_seconds
                    and text_len_words <= self.max_text_words
                ):
                    valid_features.append(f)
            except Exception:
                continue

        if not valid_features:
            # Return dummy batch if all samples filtered
            # Use a more realistic dummy sequence length
            pad_token_id = (
                self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            )
            dummy_seq_len = 10  # Reasonable sequence length for dummy batch
            return {
                "input_values": torch.zeros((1, self.n_mels, 100)),
                "input_lengths": torch.tensor([100]),
                "labels": torch.full((1, dummy_seq_len), pad_token_id, dtype=torch.long),
                "attention_mask": torch.ones(
                    (1, dummy_seq_len), dtype=torch.long
                ),  # Fixed: was zeros
            }

        # Process audio to spectrograms and normalize texts
        specs = []
        texts = []
        for f in valid_features:
            audio_array = torch.from_numpy(np.array(f["audio"]["array"], dtype=np.float32))
            spec = self.mel_transform(audio_array)
            spec_db = self.amp_to_db(spec)
            spec_norm = (spec_db - spec_db.mean()) / (spec_db.std() + 1e-8)
            specs.append(spec_norm)
            texts.append(self._normalize_text(f["text"]))

        # Pad spectrograms to the same length within the batch
        input_lengths = torch.tensor([s.shape[1] for s in specs], dtype=torch.long)
        specs_transposed = [s.transpose(0, 1) for s in specs]
        padded_specs = (
            torch.nn.utils.rnn.pad_sequence(specs_transposed, batch_first=True)
            .permute(0, 2, 1)
            .contiguous()
        )

        # Tokenize and pad text labels
        labels = self.tokenizer(texts, padding="longest", truncation=True, return_tensors="pt")

        return {
            "input_values": padded_specs,
            "input_lengths": input_lengths,
            "labels": labels["input_ids"],
            "attention_mask": labels["attention_mask"],
        }


def compute_metrics(
    eval_pred: EvalPrediction,
    tokenizer: AutoTokenizer,
    wer_metric: evaluate.EvaluationModule,
) -> Dict[str, float]:
    """Placeholder for compatibility - actual WER is computed in CustomASRTrainer."""
    # This function is not used when CustomASRTrainer computes WER via generation
    # Return empty dict as the real metrics are computed in evaluation_loop
    return {}


def parse_config(
    config_file: str, accelerator: Accelerator
) -> Tuple[ModelArguments, DataArguments, CustomTrainingArguments]:
    """Parse configuration from JSON file."""
    import sys

    if not os.path.exists(config_file):
        print(f"❌ Error: Config file '{config_file}' not found!")
        sys.exit(1)

    parser = HfArgumentParser((ModelArguments, DataArguments, CustomTrainingArguments))

    try:
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=config_file, allow_extra_keys=True
        )
    except Exception as e:
        print(f"❌ Error parsing config file: {e}")
        sys.exit(1)

    if accelerator.is_main_process:
        print(f"✅ Loaded configuration from: {config_file}")

    # Apply essential runtime settings
    training_args.remove_unused_columns = False
    training_args.label_names = ["labels"]

    # Set report_to based on environment
    if os.environ.get("WANDB_API_KEY"):
        if "wandb" not in training_args.report_to:
            training_args.report_to.append("wandb")

    # Hub settings validation
    if training_args.push_to_hub and not hf_write_token:
        if accelerator.is_main_process:
            print(
                "⚠️  Warning: push_to_hub is True but no HF_WRITE_TOKEN found. "
                "Disabling hub upload."
            )
        training_args.push_to_hub = False

    return model_args, data_args, training_args


def initialize_model(
    model_args: ModelArguments, accelerator: Accelerator
) -> Tuple[ASRModel, AutoTokenizer]:
    """Initialize the ASR model and tokenizer."""
    if accelerator.is_main_process:
        print("🚀 Initializing model and tokenizer...")

    model = ASRModel(model_args)
    tokenizer = model.decoder.tokenizer

    # Note: Gradient checkpointing is handled by Trainer based on
    # training_args.gradient_checkpointing. The model has
    # gradient_checkpointing_enable/disable methods that Trainer will call

    return model, tokenizer


def load_datasets(data_args: DataArguments, accelerator: Accelerator) -> Tuple[Dataset, Dataset]:
    """Load training and validation datasets."""
    if accelerator.is_main_process:
        print("📦 Loading datasets...")

    # Load datasets
    train_dataset = load_dataset(
        data_args.dataset_name,
        data_args.dataset_config_name,
        split=data_args.train_split,
        cache_dir=data_args.dataset_cache_dir,
        num_proc=data_args.num_proc if not accelerator.is_main_process else None,
    )
    val_dataset = load_dataset(
        data_args.dataset_name,
        data_args.dataset_config_name,
        split=data_args.eval_split,
        cache_dir=data_args.dataset_cache_dir,
        num_proc=data_args.num_proc if not accelerator.is_main_process else None,
    )

    if accelerator.is_main_process:
        print(f"✅ Datasets loaded. Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    return train_dataset, val_dataset


def setup_trainer(
    model: ASRModel,
    tokenizer: AutoTokenizer,
    training_args: CustomTrainingArguments,
    train_dataset: Dataset,
    val_dataset: Dataset,
    model_args: ModelArguments,
    data_args: DataArguments,
) -> Trainer:
    """Setup the trainer with data collator and compute metrics."""
    # Create data collator
    data_collator = DataCollator(
        tokenizer=tokenizer,
        sample_rate=data_args.sample_rate,
        n_mels=model_args.n_mels,
        max_audio_seconds=data_args.max_audio_seconds,
        max_text_words=data_args.max_text_words,
    )

    # Initialize CustomTrainer to handle tied embeddings properly
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=None,  # Using eval_loss for model selection
    )

    return trainer


def show_sample_predictions(
    model: ASRModel,
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    data_collator: DataCollator,
    device: torch.device,
    num_samples: int = 10,
) -> None:
    """Show sample predictions from the validation set using generation."""
    model.eval()

    # Get random samples from the dataset
    wer_metric = evaluate.load("wer")
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))

    with torch.no_grad():
        for i, idx in enumerate(indices, 1):
            # Get sample and collate it
            sample = dataset[idx]
            batch = data_collator([sample])

            # Move to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)

            # Use actual generation
            generated_ids = model.generate(
                input_values=batch["input_values"],
                input_lengths=batch["input_lengths"],
                max_length=256,
                num_beams=1,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            pred_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

            # Get ground truth, handling -100 padding
            labels = batch["labels"][0]
            labels = torch.where(labels == -100, tokenizer.pad_token_id, labels)
            true_text = tokenizer.decode(labels, skip_special_tokens=True)

            # Clean up and truncate texts
            pred_text = pred_text.strip()[:100] if pred_text.strip() else "[EMPTY]"
            true_text = true_text.strip()[:100] if true_text.strip() else "[EMPTY]"

            # Calculate sample WER for display
            if true_text != "[EMPTY]":
                sample_wer = wer_metric.compute(predictions=[pred_text], references=[true_text])
                print(f"\nSample {i} (WER: {sample_wer:.2f}):")
            else:
                print(f"\nSample {i}:")
            print(f"  Truth: {true_text}")
            print(f"  Pred:  {pred_text}")


def load_checkpoint(checkpoint_dir: str, model: ASRModel, accelerator: Accelerator) -> bool:
    """Load a checkpoint into the model. Returns True if successful."""
    try:
        # Check if it's a LoRA model or full model
        if os.path.exists(os.path.join(checkpoint_dir, "adapter_config.json")):
            print("   Loading LoRA adapters...")
            from peft import PeftModel

            model.decoder.model = PeftModel.from_pretrained(
                model.decoder.model.base_model.model,
                checkpoint_dir,
                device_map=accelerator.device,
            )
            # Load encoder and projector if they exist
            if os.path.exists(os.path.join(checkpoint_dir, "audio_components.bin")):
                audio_components = torch.load(
                    os.path.join(checkpoint_dir, "audio_components.bin"),
                    map_location=accelerator.device,
                )
                model.encoder.load_state_dict(audio_components["encoder"])
                model.audio_projector.load_state_dict(audio_components["audio_projector"])
        elif os.path.exists(os.path.join(checkpoint_dir, "decoder")):
            # Load from CustomTrainer saved format
            print("   Loading from checkpoint format...")
            decoder_path = os.path.join(checkpoint_dir, "decoder")
            if os.path.exists(decoder_path):
                model.decoder.model = AutoModelForCausalLM.from_pretrained(
                    decoder_path, device_map=accelerator.device, dtype=torch.bfloat16
                )
            if os.path.exists(os.path.join(checkpoint_dir, "encoder.bin")):
                model.encoder.load_state_dict(
                    torch.load(
                        os.path.join(checkpoint_dir, "encoder.bin"), map_location=accelerator.device
                    )
                )
            if os.path.exists(os.path.join(checkpoint_dir, "projector.bin")):
                model.audio_projector.load_state_dict(
                    torch.load(
                        os.path.join(checkpoint_dir, "projector.bin"),
                        map_location=accelerator.device,
                    )
                )
        else:
            print(f"⚠️  Unknown checkpoint format in {checkpoint_dir}")
            return False

        print("✅ Model loaded successfully")
        return True
    except Exception as e:
        print(f"⚠️  Failed to load checkpoint: {e}")
        return False


def find_latest_checkpoint(output_dir: str, model_args: ModelArguments) -> Optional[str]:
    """Find the latest checkpoint, either final model or training checkpoint."""
    # First, check for final model
    model_size = f"d{model_args.d_model}_l{model_args.num_layers}_r{model_args.lora_r}"
    save_path = f"{DATA_PATH}/models/final_model_{model_size}"

    if os.path.exists(f"{save_path}/adapter_config.json") or os.path.exists(f"{save_path}/decoder"):
        print(f"📂 Found final model at {save_path}")
        return save_path

    # Look for latest checkpoint in output_dir
    if os.path.exists(output_dir):
        checkpoint_dirs = [
            d
            for d in os.listdir(output_dir)
            if d.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, d))
        ]

        if checkpoint_dirs:
            # Sort by step number and get the latest
            checkpoint_dirs.sort(key=lambda x: int(x.split("-")[-1]))
            checkpoint_dir = os.path.join(output_dir, checkpoint_dirs[-1])
            print(f"📂 Found checkpoint at {checkpoint_dir}")
            return checkpoint_dir
        else:
            print(f"⚠️  No checkpoints found in {output_dir}")
    else:
        print(f"⚠️  Output directory {output_dir} does not exist")

    return None


def run_training(
    trainer: Trainer,
    tokenizer: AutoTokenizer,
    training_args: CustomTrainingArguments,
    accelerator: Accelerator,
    data_args: DataArguments = None,
    model_args: ModelArguments = None,
) -> None:
    """Run the training and save the model."""
    # Start training
    if accelerator.is_main_process:
        print("🚀 Starting training...")
        print(f"   Device: {accelerator.device}")
        print(f"   Distributed: {accelerator.state.distributed_type}")
        print(f"   Mixed precision: {accelerator.mixed_precision}")

    # Resume from checkpoint if specified
    if training_args.resume_from_checkpoint and os.path.exists(
        training_args.resume_from_checkpoint
    ):
        print(f"📂 Resuming training from checkpoint: {training_args.resume_from_checkpoint}")
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    else:
        trainer.train()

    # Push to hub if requested
    if training_args.push_to_hub and hf_write_token:
        print(f"📤 Pushing model to hub: {training_args.hub_model_id}")
        trainer.push_to_hub()
        print(f"✅ Model pushed to https://huggingface.co/{training_args.hub_model_id}")


def main() -> None:
    """Main training function - simplified with Accelerate."""
    import sys

    # Check for eval-only mode
    eval_only = False
    if "--eval-only" in sys.argv:
        eval_only = True
        sys.argv.remove("--eval-only")
        print("🔍 Running in evaluation-only mode")

    # Initialize Accelerator
    accelerator = Accelerator()

    # Handle Hugging Face authentication
    if hf_read_token:
        from huggingface_hub import login

        login(token=hf_read_token)
        if accelerator.is_main_process:
            print("✅ Logged in to Hugging Face Hub with read token")
    elif hf_write_token:
        from huggingface_hub import login

        login(token=hf_write_token)
        if accelerator.is_main_process:
            print("✅ Logged in to Hugging Face Hub with write token")
    else:
        if accelerator.is_main_process:
            print("⚠️  No HF_WRITE_TOKEN or HF_READ_TOKEN found. Model upload will be skipped.")

    # Optional: Setup WandB if available
    if os.environ.get("WANDB_API_KEY") and accelerator.is_main_process:
        import wandb

        wandb.login(key=os.environ.get("WANDB_API_KEY"))
        print("✅ Logged in to Weights & Biases")

    # Require a config file to be provided
    if len(sys.argv) < 2 or sys.argv[1] != "--config":
        print("❌ Error: Config file is required!")
        print("Usage: accelerate launch train.py --config <config_file.json>")
        print("\nExample:")
        print("  accelerate launch train.py --config experiment_config.json")
        sys.exit(1)

    if len(sys.argv) < 3:
        print("❌ Error: Config file path is missing!")
        print("Usage: accelerate launch train.py --config <config_file.json>")
        sys.exit(1)

    config_file = sys.argv[2]

    # Parse configuration
    model_args, data_args, training_args = parse_config(config_file, accelerator)

    # Initialize model
    model, tokenizer = initialize_model(model_args, accelerator)

    # Load datasets
    train_dataset, val_dataset = load_datasets(data_args, accelerator)

    # Setup trainer
    trainer = setup_trainer(
        model=model,
        tokenizer=tokenizer,
        training_args=training_args,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        model_args=model_args,
        data_args=data_args,
    )

    if eval_only:
        # Find and load the latest checkpoint
        checkpoint_dir = find_latest_checkpoint(training_args.output_dir, model_args)

        if checkpoint_dir:
            load_checkpoint(checkpoint_dir, model, accelerator)
        else:
            print("⚠️  No saved model or checkpoint found, using initialized model")

        # Run evaluation
        print("\n📊 Running evaluation...")
        metrics = trainer.evaluate()

        # Print metrics
        print("\n📈 Evaluation Results:")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.4f}")
            else:
                print(f"   {key}: {value}")

        # Show sample predictions
        print("\n📝 Sample Predictions (10 samples):")
        print("-" * 80)
        show_sample_predictions(
            model=model,
            dataset=val_dataset,
            tokenizer=tokenizer,
            data_collator=trainer.data_collator,
            device=accelerator.device,
            num_samples=10,
        )

    else:
        # Run training
        run_training(trainer, tokenizer, training_args, accelerator, data_args, model_args)


if __name__ == "__main__":
    main()
