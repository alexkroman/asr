#!/usr/bin/env python3
"""
🎙️ Conformer-SmolLM2 ASR - Accelerate Version
Simplified training script using Accelerate for hardware management.
All GPU detection, optimization flags, and distributed training is handled by Accelerate.
"""

import os
import re
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Tuple, cast

# Minimal environment setup - Accelerate handles the rest
os.environ["HF_HOME"] = "/workspace"
os.environ["HF_DATASETS_CACHE"] = "/workspace/datasets"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import torch
import torch.nn as nn
import evaluate
import numpy as np
import torchaudio.transforms as T
from datasets import load_dataset, Dataset
from einops import rearrange
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
    logging,
    EvalPrediction,
)
from accelerate import Accelerator

warnings.filterwarnings("ignore")
logging.set_verbosity_error()

# Data path for outputs
DATA_PATH = "/workspace/ASR_Conformer_SmolLM2_Optimized"
os.makedirs(f"{DATA_PATH}/checkpoints", exist_ok=True)
os.makedirs(f"{DATA_PATH}/models", exist_ok=True)
os.makedirs(f"{DATA_PATH}/logs", exist_ok=True)

# Initialize Accelerator - it will handle all hardware optimization
accelerator = Accelerator()

# Handle Hugging Face authentication
hf_write_token = os.environ.get("HF_WRITE_TOKEN") or os.environ.get("HF_TOKEN")
hf_read_token = os.environ.get("HF_READ_TOKEN")

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
    train_split: str = field(
        default="train.100", metadata={"help": "The training split to use."}
    )
    eval_split: str = field(
        default="validation", metadata={"help": "The evaluation split to use."}
    )
    max_audio_seconds: float = field(
        default=20.0, metadata={"help": "Filter out audio samples longer than this."}
    )
    max_text_words: int = field(
        default=150, metadata={"help": "Filter out text samples longer than this."}
    )
    sample_rate: int = field(
        default=16000, metadata={"help": "Audio sample rate."}
    )
    dataset_cache_dir: str = field(
        default="/workspace/datasets", metadata={"help": "Directory to cache datasets."}
    )
    num_proc: int = field(
        default=8, metadata={"help": "Number of processes for dataset loading."}
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
            [
                T.FrequencyMasking(freq_mask_param=freq_mask_param)
                for _ in range(n_freq_masks)
            ]
        )
        self.time_masks = nn.ModuleList(
            [
                T.TimeMasking(time_mask_param=time_mask_param)
                for _ in range(n_time_masks)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for freq_mask in self.freq_masks:
            x = freq_mask(x)
        for time_mask in self.time_masks:
            x = time_mask(x)
        return x


class ConformerEncoder(nn.Module):
    def __init__(self, config: ModelArguments):
        super().__init__()
        self.config = config
        self.subsample = nn.Sequential(
            nn.Conv2d(1, config.d_model, 3, 2, 1),
            nn.SiLU(),
            nn.Conv2d(config.d_model, config.d_model, 3, 2, 1),
            nn.SiLU(),
        )
        self.input_proj = nn.Linear(
            config.d_model * (config.n_mels // 4), config.d_model
        )
        self.dropout = nn.Dropout(config.conformer_dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_head,
            dim_feedforward=config.d_model * 4,
            dropout=config.conformer_dropout,
            activation="gelu",
            batch_first=False,  # We will feed it Time-first data
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=config.num_layers,
        )

    def forward(self, x: torch.Tensor, input_lengths: torch.Tensor) -> torch.Tensor:
        x = self.subsample(x.unsqueeze(1))
        x = rearrange(x, "b c f t -> b t (c f)")
        x = self.input_proj(x)
        x = self.dropout(x)
        x = x.permute(1, 0, 2)  # Switch to (Time, Batch, Dim) format

        output_lengths = input_lengths // 4
        max_len = x.size(0)  # Get the max sequence length from the time dimension

        # Create a mask of shape (Batch, Time)
        mask = (
            torch.arange(max_len, device=x.device)[None, :] >= output_lengths[:, None]
        )

        # Pass the tensor and the mask to the transformer encoder
        x = self.transformer_encoder(x, src_key_padding_mask=mask)

        x = x.permute(1, 0, 2)  # Switch back to (Batch, Time, Dim)
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
            config.decoder_model_name, torch_dtype=torch.bfloat16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(config.decoder_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "<pad>"})
            self.model.resize_token_embeddings(len(self.tokenizer))
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
                    list(config.lora_target_modules)
                    if config.lora_target_modules
                    else None
                ),
                lora_dropout=config.lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            self.model = get_peft_model(self.model, lora_config)
            if hasattr(self.model, "print_trainable_parameters") and accelerator.is_main_process:
                self.model.print_trainable_parameters()


class ASRModel(nn.Module):
    def __init__(self, config: ModelArguments) -> None:
        super().__init__()
        self.encoder = ConformerEncoder(config)
        self.decoder = SmolLM2Decoder(config)
        text_dim = getattr(self.decoder.model.config, "hidden_size", 768)
        self.audio_projector = LightweightAudioProjector(
            config.d_model, text_dim, config
        )
        self.spec_augment = SpecAugment()

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

        # Use SDPA for attention (Accelerate will optimize this for the hardware)
        with torch.backends.cuda.sdp_kernel(
            enable_flash=True, enable_math=False, enable_mem_efficient=True
        ):
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

        return self.decoder.model.forward(
            inputs_embeds=combined_embeds,
            attention_mask=combined_attention_mask,
            labels=combined_labels,
        )

    @torch.inference_mode()
    def generate(
        self, input_values: torch.Tensor, input_lengths: torch.Tensor, **kwargs: Dict[str, Union[int, float, torch.Tensor]]
    ) -> torch.Tensor:
        audio_features = self.encoder(input_values, input_lengths)
        audio_prefix = self.audio_projector(audio_features)
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

    def __call__(self, features: List[Dict[str, Union[str, Dict[str, Union[str, List[float]]]]]) -> Dict[str, torch.Tensor]:
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
            return {
                "input_values": torch.zeros((1, self.n_mels, 100)),
                "input_lengths": torch.tensor([100]),
                "labels": torch.tensor([[0]]),
                "attention_mask": torch.tensor([[1]]),
            }

        # Process audio to spectrograms and normalize texts
        specs = []
        texts = []
        for f in valid_features:
            audio_array = torch.from_numpy(
                np.array(f["audio"]["array"], dtype=np.float32)
            )
            spec = self.mel_transform(audio_array)
            spec_db = self.amp_to_db(spec)
            spec_norm = (spec_db - spec_db.mean()) / (spec_db.std() + 1e-8)
            specs.append(spec_norm)
            texts.append(self._normalize_text(f["text"]))

        # Pad spectrograms to the same length within the batch
        input_lengths = torch.tensor([s.shape[1] for s in specs], dtype=torch.long)
        specs_transposed = [s.transpose(0, 1) for s in specs]
        padded_specs = torch.nn.utils.rnn.pad_sequence(
            specs_transposed, batch_first=True
        ).permute(0, 2, 1)

        # Tokenize and pad text labels
        labels = self.tokenizer(
            texts, padding="longest", truncation=True, return_tensors="pt"
        )

        return {
            "input_values": padded_specs,
            "input_lengths": input_lengths,
            "labels": labels["input_ids"],
            "attention_mask": labels["attention_mask"],
        }


def compute_metrics(eval_pred: EvalPrediction, tokenizer: AutoTokenizer, wer_metric: evaluate.EvaluationModule) -> Dict[str, float]:
    """Compute metrics for ASR evaluation."""
    predictions = eval_pred.predictions
    labels = eval_pred.label_ids

    # Handle the case where predictions might be a tuple
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    # Replace -100 with pad token id for proper decoding
    labels = np.where(labels == -100, tokenizer.pad_token_id, labels)

    # Decode predictions and labels
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Normalize text (remove extra spaces, convert to lowercase)
    decoded_preds = [pred.strip().lower() for pred in decoded_preds]
    decoded_labels = [label.strip().lower() for label in decoded_labels]

    # Compute WER
    wer = wer_metric.compute(predictions=decoded_preds, references=decoded_labels)

    return {"wer": wer if wer is not None else 0.0}


def parse_config(config_file: str) -> Tuple[ModelArguments, DataArguments, TrainingArguments]:
    """Parse configuration from JSON file."""
    import os
    import sys

    if not os.path.exists(config_file):
        print(f"❌ Error: Config file '{config_file}' not found!")
        sys.exit(1)

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))

    try:
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=config_file,
            allow_extra_keys=True
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
            print("⚠️  Warning: push_to_hub is True but no HF_WRITE_TOKEN found. Disabling hub upload.")
        training_args.push_to_hub = False

    return model_args, data_args, training_args


def initialize_model(model_args: ModelArguments) -> Tuple[ASRModel, AutoTokenizer]:
    """Initialize the ASR model and tokenizer."""
    if accelerator.is_main_process:
        print("🚀 Initializing model and tokenizer...")

    model = ASRModel(model_args)
    tokenizer = model.decoder.tokenizer

    # Enable gradient checkpointing
    if hasattr(model.decoder.model, "gradient_checkpointing_enable"):
        if callable(model.decoder.model.gradient_checkpointing_enable):
            model.decoder.model.gradient_checkpointing_enable()
            if accelerator.is_main_process:
                print("✅ Gradient checkpointing enabled")

    # Model compilation with torch.compile (if supported)
    if torch.__version__ >= "2.0.0" and accelerator.state.distributed_type == "NO":
        if accelerator.is_main_process:
            print("🚀 Compiling model with torch.compile...")
        model.encoder = torch.compile(model.encoder, mode="reduce-overhead")
        model.audio_projector = torch.compile(model.audio_projector, mode="reduce-overhead")
        if accelerator.is_main_process:
            print("✅ Model compilation complete")

    return model, tokenizer

def load_datasets(data_args: DataArguments) -> Tuple[Dataset, Dataset]:
    """Load training and validation datasets."""
    if accelerator.is_main_process:
        print("📦 Loading datasets...")

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
    training_args: TrainingArguments,
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

    # Setup metrics computation
    wer_metric = evaluate.load("wer")
    compute_metrics_fn = lambda eval_pred: compute_metrics(eval_pred, tokenizer, wer_metric)

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_fn,
    )

    return trainer

def run_training(trainer: Trainer, tokenizer: AutoTokenizer, training_args: TrainingArguments) -> None:
    """Run the training and save the model."""
    # Start training
    if accelerator.is_main_process:
        print("🚀 Starting training...")
        print(f"   Device: {accelerator.device}")
        print(f"   Distributed: {accelerator.state.distributed_type}")
        print(f"   Mixed precision: {accelerator.mixed_precision}")

    trainer.train()

    # Save final model (only on main process)
    if accelerator.is_main_process:
        print("💾 Saving final model...")
        save_path = f"{DATA_PATH}/models/final_model"
        trainer.save_model(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"✅ Model saved to {save_path}")

        # Push to hub if requested
        if training_args.push_to_hub and hf_write_token:
            print(f"📤 Pushing model to hub: {training_args.hub_model_id}")
            trainer.push_to_hub()
            print(f"✅ Model pushed to https://huggingface.co/{training_args.hub_model_id}")


def main() -> None:
    """Main training function - simplified with Accelerate."""
    import sys

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
    model_args, data_args, training_args = parse_config(config_file)

    # Initialize model
    model, tokenizer = initialize_model(model_args)

    # Load datasets
    train_dataset, val_dataset = load_datasets(data_args)

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

    # Run training
    run_training(trainer, tokenizer, training_args)


if __name__ == "__main__":
    main()