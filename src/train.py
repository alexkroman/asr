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




@dataclass
class CustomTrainingArguments(TrainingArguments):
    """Extended TrainingArguments with custom fields."""

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
    projector_num_layers: int = field(default=2, metadata={"help": "Number of layers in the deep projector."})


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: str = field(
        default="librispeech_asr", metadata={"help": "The name of the dataset to use."}
    )
    dataset_configs: List[str] = field(
        default_factory=lambda: ["clean"], metadata={"help": "List of dataset configurations to concatenate."}
    )
    train_splits: List[str] = field(
        default_factory=lambda: ["train.100"], metadata={"help": "List of training splits corresponding to dataset_configs."}
    )
    eval_splits: List[str] = field(
        default_factory=lambda: ["validation"], metadata={"help": "List of evaluation splits corresponding to dataset_configs."}
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
        self.pointwise_conv1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1, bias=False)
        self.glu = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            groups=d_model,
            bias=False,
        )
        self.activation = nn.SiLU()
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1, bias=False)
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
        self.linear1 = nn.Linear(d_model, d_ff, bias=False)
        self.activation = nn.SiLU()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model, bias=False)
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
        self.self_attn = nn.MultiheadAttention(
            d_model, n_head, dropout=dropout, batch_first=True, bias=False
        )
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
        self._gradient_checkpointing = False  # Use private attribute to follow HF convention

        # Convolutional subsampling with stride 2 for each layer
        # Total subsampling factor = 2^num_subsample_layers
        self.subsample_layers = 2  # Number of conv layers with stride 2
        self.subsample_factor = 2**self.subsample_layers  # Total subsampling: 4x

        # Convolutional subsampling - simplified without normalization layers
        # The normalization will happen after reshaping
        self.subsample = nn.Sequential(
            nn.Conv2d(1, config.d_model, 3, 2, 1, bias=False),
            nn.SiLU(),
            nn.Conv2d(config.d_model, config.d_model, 3, 2, 1, bias=False),
            nn.SiLU(),
        )

        # Linear projection after subsampling with normalization
        # Calculate feature dimension after frequency subsampling
        freq_subsample_factor = (
            self.subsample_factor
        )  # Frequency dimension also reduced by same factor
        input_dim = config.d_model * (config.n_mels // freq_subsample_factor)

        # Add layer norm before projection to stabilize gradients
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Linear(input_dim, config.d_model, bias=False)
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
        x = x.contiguous()

        # Convolutional subsampling
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.subsample(x)

        # Clamp values to prevent explosion
        x = torch.clamp(x, min=-10, max=10)

        x = rearrange(x, "b c f t -> b t (c f)")
        x = self.input_norm(x)  # Normalize before projection
        x = self.input_proj(x)
        x = self.dropout(x)

        # Calculate output lengths after subsampling
        # Use integer arithmetic to avoid floating point errors
        # For Conv2d with kernel=3, stride=2, padding=1:
        # output = floor((input + 2*padding - kernel) / stride) + 1
        output_lengths = input_lengths.clone()

        # Apply convolution length calculation for each subsampling layer
        # Using explicit parameters to match the actual Conv2d layers
        for _ in range(self.subsample_layers):
            # Matches nn.Conv2d(_, _, kernel_size=3, stride=2, padding=1)
            output_lengths = torch.div(output_lengths + 2 * 1 - 3, 2, rounding_mode='floor') + 1

        # Ensure output lengths don't exceed actual tensor size
        max_len = x.size(1)
        output_lengths = torch.minimum(output_lengths, torch.tensor(max_len, device=output_lengths.device))

        # Create padding mask
        key_padding_mask = (
            torch.arange(max_len, device=x.device)[None, :] >= output_lengths[:, None]
        )

        # Apply Conformer blocks with optional gradient checkpointing
        for conformer_block in self.conformer_blocks:
            if self._gradient_checkpointing and self.training:
                # Use gradient checkpointing to save memory during training
                x = self._checkpoint_forward(conformer_block, x, key_padding_mask)
            else:
                x = conformer_block(x, key_padding_mask)

        return x

    def _checkpoint_forward(self, module, *args):
        """Helper method for gradient checkpointing with proper handling."""
        from torch.utils.checkpoint import checkpoint
        return checkpoint(module, *args, use_reentrant=False)

    @property
    def gradient_checkpointing(self):
        return self._gradient_checkpointing

    @gradient_checkpointing.setter
    def gradient_checkpointing(self, value):
        self._gradient_checkpointing = value


class DeepAudioProjector(nn.Module):
    def __init__(self, audio_dim: int, text_dim: int, config: ModelArguments):
        super().__init__()
        self.num_layers = config.projector_num_layers

        self.audio_proj = nn.Linear(audio_dim, text_dim, bias=False)
        self.queries = nn.Parameter(torch.zeros(config.num_queries, text_dim))
        nn.init.normal_(self.queries, mean=0.0, std=0.01)

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'cross_attn': nn.MultiheadAttention(
                    embed_dim=text_dim,
                    num_heads=config.projector_num_heads,
                    dropout=config.projector_dropout,
                    batch_first=True,
                ),
                'self_attn': nn.MultiheadAttention(
                    embed_dim=text_dim,
                    num_heads=config.projector_num_heads,
                    dropout=config.projector_dropout,
                    batch_first=True,
                ),
                'mlp': nn.Sequential(
                    nn.LayerNorm(text_dim),
                    nn.Linear(text_dim, text_dim * 4),
                    nn.GELU(),
                    nn.Linear(text_dim * 4, text_dim),
                ),
                'norm1': nn.LayerNorm(text_dim),
                'norm2': nn.LayerNorm(text_dim),
            }) for _ in range(self.num_layers)
        ])

    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        B = audio_features.shape[0]
        audio_proj = self.audio_proj(audio_features)

        q = self.queries.unsqueeze(0).expand(B, -1, -1)

        for layer in self.layers:
            attn_out, _ = layer['cross_attn'](q, audio_proj, audio_proj)
            q = layer['norm1'](q + attn_out)

            self_attn_out, _ = layer['self_attn'](q, q, q)
            q = layer['norm2'](q + self_attn_out)

            q = q + layer['mlp'](q)

        return q


class SmolLM2Decoder(nn.Module):
    def __init__(self, config: ModelArguments):
        super().__init__()
        self.model: nn.Module = AutoModelForCausalLM.from_pretrained(
            config.decoder_model_name, dtype=torch.bfloat16, token=False
        )
        self.tokenizer = AutoTokenizer.from_pretrained(config.decoder_model_name, token=False)

        # Handle padding token configuration
        if self.tokenizer.pad_token is None:
            # Check if model already has a pad token configured
            if hasattr(self.model.config, 'pad_token_id') and self.model.config.pad_token_id is not None:
                # Use the model's existing pad token
                vocab = self.tokenizer.get_vocab()
                # Try to find the token corresponding to the model's pad_token_id
                for token, token_id in vocab.items():
                    if token_id == self.model.config.pad_token_id:
                        self.tokenizer.pad_token = token
                        break
                else:
                    # If no corresponding token found, add a new one
                    self._add_padding_token()
            else:
                # Neither tokenizer nor model has padding configured
                self._add_padding_token()
        else:
            # Tokenizer has pad token, ensure model config is synchronized
            if hasattr(self.model.config, 'pad_token_id'):
                self.model.config.pad_token_id = self.tokenizer.pad_token_id

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

    def _add_padding_token(self):
        """Add a padding token to the tokenizer and resize model embeddings."""
        # Add the padding token
        self.tokenizer.add_special_tokens({"pad_token": "<pad>"})

        # Resize model embeddings to accommodate the new token
        self.model.resize_token_embeddings(len(self.tokenizer), mean_resizing=False)

        # Initialize the new embedding with the mean of existing embeddings
        with torch.no_grad():
            embeddings = self.model.get_input_embeddings()
            if embeddings is not None and hasattr(embeddings, "weight"):
                embedding_weight: torch.nn.Parameter = embeddings.weight  # type: ignore
                # Initialize new token embedding as mean of existing embeddings
                embedding_weight.data[-1] = embedding_weight.data[:-1].mean(dim=0)

        # Update model config
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

    def forward(self, **kwargs):
        """Simple forward pass through the decoder model."""
        return self.model(**kwargs)





class ASRModel(nn.Module):
    def __init__(self, config: ModelArguments) -> None:
        super().__init__()
        self.encoder = ConformerEncoder(config)
        self.decoder = SmolLM2Decoder(config)
        text_dim = getattr(self.decoder.model.config, "hidden_size", 768)
        self.audio_projector = DeepAudioProjector(config.d_model, text_dim, config)
        self.spec_augment = SpecAugment()

        # Initialize weights with smaller values for stability
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with more conservative values to prevent gradient explosion."""
        for name, module in self.named_modules():
            # Skip decoder weights - they're already pretrained
            if 'decoder' in name:
                continue

            if isinstance(module, nn.Linear):
                # Very conservative initialization to prevent gradient explosion
                nn.init.xavier_uniform_(module.weight, gain=0.02)  # Even smaller gain
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d):
                # Very conservative initialization for conv layers
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                with torch.no_grad():
                    module.weight.data *= 0.1  # Much smaller scale
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.GroupNorm):
                if hasattr(module, 'weight') and module.weight is not None:
                    nn.init.ones_(module.weight)
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                if hasattr(module, 'weight') and module.weight is not None:
                    nn.init.ones_(module.weight)
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.MultiheadAttention):
                # More conservative initialization for attention
                for param in module.parameters():
                    if param.dim() > 1:
                        nn.init.xavier_uniform_(param, gain=0.1)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """
        Enable gradient checkpointing for both encoder and decoder models.
        This method is called by the Trainer when gradient_checkpointing is enabled.
        """
        # Enable for encoder using property
        self.encoder.gradient_checkpointing = True

        # Enable for decoder using its native method
        if hasattr(self.decoder.model, "gradient_checkpointing_enable"):
            self.decoder.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
        elif hasattr(self.decoder.model, "enable_gradient_checkpointing"):
            self.decoder.model.enable_gradient_checkpointing()

    def gradient_checkpointing_disable(self):
        """
        Disable gradient checkpointing for both encoder and decoder models.
        This method is called by the Trainer when gradient_checkpointing is disabled.
        """
        # Disable for encoder using property
        self.encoder.gradient_checkpointing = False

        # Disable for decoder using its native method
        if hasattr(self.decoder.model, "gradient_checkpointing_disable"):
            self.decoder.model.gradient_checkpointing_disable()
        elif hasattr(self.decoder.model, "disable_gradient_checkpointing"):
            self.decoder.model.disable_gradient_checkpointing()

    @property
    def is_gradient_checkpointing(self):
        """Check if gradient checkpointing is enabled."""
        return self.encoder.gradient_checkpointing

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

            # Convert audio_prefix to same dtype as text embeddings
            if audio_prefix.dtype != text_embeds.dtype:
                audio_prefix = audio_prefix.to(text_embeds.dtype)

            combined_embeds = torch.cat([audio_prefix, text_embeds], dim=1)
        else:
            # Convert to decoder dtype when no labels
            decoder_dtype = next(self.decoder.model.parameters()).dtype
            if audio_prefix.dtype != decoder_dtype:
                audio_prefix = audio_prefix.to(decoder_dtype)
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

        # During training, only return loss and logits to avoid DynamicCache padding issues
        # During inference/generation, return full outputs including KV cache
        if self.training and hasattr(outputs, "loss") and hasattr(outputs, "logits"):
            from transformers.modeling_outputs import CausalLMOutputWithPast

            return CausalLMOutputWithPast(
                loss=outputs.loss,
                logits=outputs.logits,
                past_key_values=None,  # Don't return cache during training
                hidden_states=None,
                attentions=None,
            )

        # Return full outputs (including KV cache) for eval/generation
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

        # Process audio to spectrograms and normalize texts
        specs = []
        texts = []
        for f in valid_features:
            audio_array = torch.from_numpy(np.array(f["audio"]["array"], dtype=np.float32))

            # Check for NaN/Inf in raw audio
            if not torch.all(torch.isfinite(audio_array)):
                # Handle corrupted audio by replacing with silence
                audio_array = torch.zeros_like(audio_array)

            # Clip audio to prevent extreme values
            audio_array = torch.clamp(audio_array, min=-1.0, max=1.0)

            # Apply mel transform
            spec = self.mel_transform(audio_array)

            # Add small epsilon to prevent log(0)
            spec = spec + 1e-10

            # Convert to dB scale
            spec_db = self.amp_to_db(spec)

            # Check for NaN/Inf after dB conversion
            if not torch.all(torch.isfinite(spec_db)):
                # Fall back to zero spectrogram
                spec_db = torch.zeros_like(spec_db)

            # Robust normalization
            mean_val = spec_db.mean()
            std_val = spec_db.std()

            # Handle edge cases for normalization
            if not torch.isfinite(mean_val):
                mean_val = torch.tensor(0.0)
            if not torch.isfinite(std_val) or std_val < 1e-6:
                std_val = torch.tensor(1.0)

            # Apply normalization with safe std
            spec_norm = (spec_db - mean_val) / std_val

            # Final clamping to ensure bounded values
            spec_norm = torch.clamp(spec_norm, min=-5.0, max=5.0)

            # Final check
            if not torch.all(torch.isfinite(spec_norm)):
                spec_norm = torch.zeros_like(spec_norm)

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

        output_batch = {
            "input_values": padded_specs,
            "input_lengths": input_lengths,
            "labels": labels["input_ids"],
            "attention_mask": labels["attention_mask"],
        }

        return output_batch




def parse_config(
    config_file: str, accelerator: Accelerator
) -> Tuple[ModelArguments, DataArguments, CustomTrainingArguments]:
    """Parse configuration from JSON file."""
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file '{config_file}' not found!")

    parser = HfArgumentParser((ModelArguments, DataArguments, CustomTrainingArguments))

    try:
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=config_file, allow_extra_keys=True
        )
    except Exception as e:
        raise ValueError(f"Error parsing config file: {e}") from e

    # Configuration loaded successfully

    # Apply essential runtime settings
    training_args.remove_unused_columns = False
    training_args.label_names = ["labels"]

    # Enable TensorBoard if specified, otherwise disable reporting
    if "tensorboard" not in training_args.report_to:
        training_args.report_to = []

    # Disable hub pushing
    training_args.push_to_hub = False

    return model_args, data_args, training_args


def initialize_model(
    model_args: ModelArguments, accelerator: Accelerator
) -> Tuple[ASRModel, AutoTokenizer]:
    """Initialize the ASR model and tokenizer."""
    # Initialize model and tokenizer

    model = ASRModel(model_args)
    tokenizer = model.decoder.tokenizer

    # Note: Gradient checkpointing is handled by Trainer based on
    # training_args.gradient_checkpointing. The model has
    # gradient_checkpointing_enable/disable methods that Trainer will call

    return model, tokenizer


def load_datasets(data_args: DataArguments, accelerator: Accelerator) -> Tuple[Dataset, Dataset]:
    """Load training and validation datasets."""
    from datasets import concatenate_datasets

    # Load datasets

    # Adjust num_proc based on platform
    import platform
    if platform.system() == "Darwin":  # Mac
        safe_num_proc = min(data_args.num_proc, 1) if data_args.num_proc else None
    else:  # Linux/CUDA
        safe_num_proc = data_args.num_proc if data_args.num_proc else None

    train_datasets = []
    val_datasets = []

    for config, train_split, eval_split in zip(
        data_args.dataset_configs, data_args.train_splits, data_args.eval_splits
    ):
        # Load dataset configuration

        train_ds = load_dataset(
            data_args.dataset_name,
            config,
            split=train_split,
            cache_dir=data_args.dataset_cache_dir,
            num_proc=safe_num_proc,
        )
        train_datasets.append(train_ds)

        val_ds = load_dataset(
            data_args.dataset_name,
            config,
            split=eval_split,
            cache_dir=data_args.dataset_cache_dir,
            num_proc=safe_num_proc,
        )
        val_datasets.append(val_ds)

    # Concatenate all datasets
    train_dataset = concatenate_datasets(train_datasets)
    val_dataset = concatenate_datasets(val_datasets)

    # Dataset sizes recorded

    # Limit samples if specified (for overfitting experiments)
    if data_args.max_train_samples is not None:
        train_dataset = train_dataset.select(range(min(data_args.max_train_samples, len(train_dataset))))

    if data_args.max_eval_samples is not None:
        val_dataset = val_dataset.select(range(min(data_args.max_eval_samples, len(val_dataset))))

    # Datasets loaded successfully

    return train_dataset, val_dataset



def compute_metrics(eval_pred: EvalPrediction, tokenizer: AutoTokenizer) -> Dict[str, float]:
    """Compute WER metric for evaluation.

    Args:
        eval_pred: EvalPrediction object from Trainer
        tokenizer: Tokenizer for decoding
    """
    predictions = eval_pred.predictions
    label_ids = eval_pred.label_ids

    # Handle small evaluation sets
    total_samples = len(predictions)
    # Sample up to 10% of data or max 100 samples for WER calculation to save time
    max_samples = min(max(10, int(total_samples * 0.1)), 100)

    if total_samples > max_samples:
        indices = random.sample(range(total_samples), max_samples)
        predictions = predictions[indices]
        label_ids = label_ids[indices]
    else:
        max_samples = total_samples

    # Decode predictions and labels
    if len(predictions.shape) == 3:
        # If predictions are logits, get argmax
        predictions = np.argmax(predictions, axis=-1)

    # Replace -100 with pad_token_id for labels
    label_ids = np.where(label_ids == -100, tokenizer.pad_token_id, label_ids)

    # Decode texts
    pred_texts = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    label_texts = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # Normalize texts
    pred_texts = [text.lower().strip() for text in pred_texts]
    label_texts = [text.lower().strip() for text in label_texts]

    # Calculate WER
    wer_metric = evaluate.load("wer")
    wer = wer_metric.compute(predictions=pred_texts, references=label_texts)

    return {
        "eval_wer": wer,
        "eval_samples": max_samples,
    }


def setup_trainer(
    model: ASRModel,
    tokenizer: AutoTokenizer,
    training_args: CustomTrainingArguments,
    train_dataset: Dataset,
    val_dataset: Dataset,
    model_args: ModelArguments,
    data_args: DataArguments,
    accelerator: Accelerator,
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

    # Create compute_metrics function with tokenizer bound
    def compute_metrics_fn(eval_pred):
        return compute_metrics(eval_pred, tokenizer)

    # Initialize standard Trainer
    # Only compute metrics if explicitly requested in config
    use_metrics = getattr(training_args, 'compute_metrics', False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_fn if use_metrics else None,
    )

    return trainer


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

    # Resume from checkpoint if specified
    if training_args.resume_from_checkpoint and os.path.exists(
        training_args.resume_from_checkpoint
    ):
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    else:
        trainer.train()

    # Hub pushing disabled


def main() -> None:
    """Main training function - simplified with Accelerate."""
    import sys

    # Initialize Accelerator
    accelerator = Accelerator()

    # HuggingFace authentication removed - not needed for local training

    # WandB removed - not needed for local training

    # Require a config file to be provided
    if len(sys.argv) < 2 or sys.argv[1] != "--config":
        raise ValueError(
            "Config file is required!\n"
            "Usage: accelerate launch train.py --config <config_file.json>\n"
            "Example: accelerate launch train.py --config experiment_config.json"
        )

    if len(sys.argv) < 3:
        raise ValueError(
            "Config file path is missing!\n"
            "Usage: accelerate launch train.py --config <config_file.json>"
        )

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
        accelerator=accelerator,
    )

    # Run training
    run_training(trainer, tokenizer, training_args, accelerator, data_args, model_args)


if __name__ == "__main__":
    main()
