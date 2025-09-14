"""
🎙️ Conformer-SmolLM2 ASR - Hugging Face Trainer Version
- Uses Hugging Face Trainer for training loop
- Optimized for A40 GPUs with 9 vCPUs
- Configured for the LibriSpeech train.clean.100 dataset
"""

import argparse
import os
import re
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Tuple

# Set Hugging Face cache directory to /workspace BEFORE importing any HF libraries
os.environ["HF_HOME"] = "/workspace"
os.environ["HF_DATASETS_CACHE"] = "/workspace/datasets"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/workspace/hub"
os.environ["XDG_CACHE_HOME"] = "/workspace"  # Some libraries use this as fallback

# Optimize data downloading
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"  # Use faster hf_transfer for downloads
os.environ["HF_DATASETS_DOWNLOAD_MANAGER_MAX_WORKERS"] = "8"  # Parallel downloads

# Optimize for A40 with 9 vCPUs
os.environ["OMP_NUM_THREADS"] = "9"
os.environ["MKL_NUM_THREADS"] = "9"
os.environ["NUMEXPR_NUM_THREADS"] = "9"

import torch
import torch.nn as nn
import evaluate
import numpy as np
import torchaudio.transforms as T
from datasets import load_dataset
from einops import rearrange
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    logging,
    EvalPrediction,
)

warnings.filterwarnings("ignore")
logging.set_verbosity_error()
DATA_PATH = "/workspace/ASR_Conformer_SmolLM2_Optimized"
os.makedirs(f"{DATA_PATH}/checkpoints", exist_ok=True)
os.makedirs(f"{DATA_PATH}/models", exist_ok=True)
os.makedirs(f"{DATA_PATH}/logs", exist_ok=True)

# A40 Optimizations (NVIDIA Ampere architecture)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.set_float32_matmul_precision("high")

# Detect GPU type
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    if "A40" in gpu_name:
        print("🚀 A40 detected - enabling Ampere-specific optimizations")
        # Set specific optimizations for A40
        torch.cuda.set_per_process_memory_fraction(0.95)  # Use most of available memory
    elif "H100" in gpu_name or "H200" in gpu_name:
        print("🚀 H100/H200 detected - enabling Hopper-specific optimizations")
    elif "A100" in gpu_name:
        print("🚀 A100 detected - enabling Ampere-specific optimizations")

# Handle Hugging Face authentication
hf_read_token = os.environ.get("HF_READ_TOKEN")
hf_write_token = os.environ.get("HF_WRITE_TOKEN") or os.environ.get("HF_TOKEN")

if hf_read_token:
    from huggingface_hub import login
    login(token=hf_read_token)
    print("✅ Logged in to Hugging Face Hub with read token")
elif hf_write_token:
    from huggingface_hub import login
    login(token=hf_write_token)
    print("✅ Logged in to Hugging Face Hub with write token")
else:
    print("⚠️  No HF_WRITE_TOKEN or HF_READ_TOKEN found. Model upload will be skipped.")

# Optional: Setup WandB if available
if os.environ.get("WANDB_API_KEY"):
    import wandb
    wandb.login(key=os.environ.get("WANDB_API_KEY"))
    print("✅ Logged in to Weights & Biases")


@dataclass
class ConformerConfig:
    n_mels: int = 80
    d_model: int = 512
    n_head: int = 8
    num_layers: int = 12
    kernel_size: int = 15
    dropout: float = 0.1


@dataclass
class SmolLM2Config:
    model_name: str = "HuggingFaceTB/SmolLM2-360M-Instruct"
    use_lora: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )


@dataclass
class ProjectorConfig:
    num_queries: int = 24
    num_heads: int = 8
    dropout: float = 0.1


class SpecAugment(nn.Module):
    def __init__(
        self, freq_mask_param: int = 27, time_mask_param: int = 100,
        n_freq_masks: int = 2, n_time_masks: int = 2
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
    def __init__(self, config: ConformerConfig):
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
        self.dropout = nn.Dropout(config.dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_head,
            dim_feedforward=config.d_model * 4,
            dropout=config.dropout,
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
        mask = torch.arange(max_len, device=x.device)[None, :] >= output_lengths[:, None]

        # Pass the tensor and the mask to the transformer encoder
        x = self.transformer_encoder(x, src_key_padding_mask=mask)

        x = x.permute(1, 0, 2)  # Switch back to (Batch, Time, Dim)
        return x


class LightweightAudioProjector(nn.Module):
    def __init__(self, audio_dim: int, text_dim: int, config: ProjectorConfig):
        super().__init__()
        self.audio_proj = nn.Linear(audio_dim, text_dim)
        self.queries = nn.Parameter(torch.randn(config.num_queries, text_dim))
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=text_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.mlp = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, text_dim * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
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
    def __init__(self, config: SmolLM2Config):
        super().__init__()
        self.model: nn.Module = AutoModelForCausalLM.from_pretrained(
            config.model_name, torch_dtype=torch.bfloat16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "<pad>"})
            self.model.resize_token_embeddings(len(self.tokenizer))
            with torch.no_grad():
                embeddings = self.model.get_input_embeddings()
                if embeddings is not None and hasattr(embeddings, 'weight'):
                    embedding_weight: torch.nn.Parameter = embeddings.weight  # type: ignore
                    embedding_weight.data[-1] = embedding_weight.data[:-1].mean(dim=0)
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        if config.use_lora:
            lora_config = LoraConfig(
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                target_modules=list(config.lora_target_modules) if config.lora_target_modules else None,
                lora_dropout=config.lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            self.model = get_peft_model(self.model, lora_config)
            if hasattr(self.model, 'print_trainable_parameters'):
                self.model.print_trainable_parameters()


class ASRModel(nn.Module):
    def __init__(self, conformer_cfg: ConformerConfig, smollm2_cfg: SmolLM2Config,
                 proj_cfg: ProjectorConfig) -> None:
        super().__init__()
        self.encoder = ConformerEncoder(conformer_cfg)
        self.decoder = SmolLM2Decoder(smollm2_cfg)
        text_dim = getattr(self.decoder.model.config, 'hidden_size', 768)
        self.audio_projector = LightweightAudioProjector(
            conformer_cfg.d_model, text_dim, proj_cfg
        )
        self.spec_augment = SpecAugment()

    def forward(
        self,
        input_values: Optional[torch.Tensor] = None,
        input_lengths: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any
    ) -> Any:
        # Handle dict input from Trainer
        if input_values is None and "input_values" in kwargs:
            input_values = kwargs["input_values"]
        if input_lengths is None and "input_lengths" in kwargs:
            input_lengths = kwargs["input_lengths"]

        if input_values is None or input_lengths is None:
            raise ValueError("input_values and input_lengths are required")

        if self.training:
            input_values = self.spec_augment(input_values)

        # Enable FlashAttention 2 via SDPA (works on A40 too)
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True):
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
                device=labels.device
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
    def generate(self, input_values: torch.Tensor, input_lengths: torch.Tensor,
                 **kwargs: Any) -> Any:
        audio_features = self.encoder(input_values, input_lengths)
        audio_prefix = self.audio_projector(audio_features)
        return self.decoder.model.generate(inputs_embeds=audio_prefix, **kwargs)


@dataclass
class DataCollator:
    """Data collator that performs preprocessing on-the-fly."""
    tokenizer: Any
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

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Filter samples that are too long
        valid_features = []
        for f in features:
            try:
                audio_len_sec = len(f["audio"]["array"]) / self.sample_rate
                text_len_words = len(self._normalize_text(f["text"]).split())
                if (audio_len_sec <= self.max_audio_seconds
                        and text_len_words <= self.max_text_words):
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
            audio_array = torch.from_numpy(np.array(f["audio"]["array"], dtype=np.float32))
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


def compute_metrics(eval_pred: EvalPrediction, tokenizer: Any, wer_metric: Any) -> Dict[str, float]:
    """Compute WER metric for evaluation."""
    predictions = eval_pred.predictions
    label_ids = eval_pred.label_ids

    if predictions is None or label_ids is None:
        return {"wer": 0.0}

    # Decode predictions and labels
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # Compute WER
    wer = wer_metric.compute(predictions=decoded_preds, references=decoded_labels)

    return {"wer": wer if wer is not None else 0.0}


class ASRTrainer(Trainer):
    """Custom trainer with ASR-specific generate method for evaluation."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.wer_metric = evaluate.load("wer")

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        """Compute loss with proper input handling."""
        outputs = model(**inputs)
        loss = outputs.loss if hasattr(outputs, 'loss') else outputs['loss']
        return (loss, outputs) if return_outputs else loss

    def evaluation_loop(
        self,
        dataloader: Any,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval"
    ) -> Any:
        """Custom evaluation loop with WER computation."""
        model = self._wrap_model(self.model, training=False)
        model.eval()

        all_preds: List[np.ndarray] = []
        all_labels: List[np.ndarray] = []
        total_loss = 0.0

        for step, inputs in enumerate(dataloader):
            inputs = self._prepare_inputs(inputs)

            with torch.no_grad():
                # Compute loss
                outputs = model(**inputs)
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs['loss']
                total_loss += loss.item()

                # Generate predictions for WER
                if hasattr(model, 'module'):
                    generate_fn = model.module.generate
                else:
                    generate_fn = model.generate

                if self.tokenizer is not None:
                    predictions = generate_fn(
                        input_values=inputs["input_values"],
                        input_lengths=inputs["input_lengths"],
                        max_new_tokens=150,
                        num_beams=5,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )

                    all_preds.extend(predictions.cpu().numpy())
                    all_labels.extend(inputs["labels"].cpu().numpy())

        # Compute metrics
        avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0

        # Decode and compute WER
        if self.tokenizer is not None and all_preds:
            decoded_preds = self.tokenizer.batch_decode(all_preds, skip_special_tokens=True)
            decoded_labels = self.tokenizer.batch_decode(all_labels, skip_special_tokens=True)
            wer = self.wer_metric.compute(predictions=decoded_preds, references=decoded_labels)
        else:
            wer = 0.0

        metrics = {
            f"{metric_key_prefix}_loss": avg_loss,
            f"{metric_key_prefix}_wer": wer if wer is not None else 0.0,
        }

        self.log(metrics)

        return metrics


def get_training_args(output_dir: str = f"{DATA_PATH}/checkpoints") -> TrainingArguments:
    """Get optimized training arguments for A40 GPU with 9 vCPUs."""
    # Detect GPU and adjust batch size
    batch_size = 16  # default
    num_workers = 4  # default for data loading

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        gpu_mem = props.total_memory / 1e9

        # A40 specific optimizations
        if "A40" in gpu_name:
            batch_size = 96  # A40 has 48GB VRAM, optimal batch size
            num_workers = 8  # Use most of the 9 vCPUs for data loading
            print(f"📊 A40 detected ({gpu_mem:.1f}GB VRAM) with 9 vCPUs")
            print(f"   Optimized batch size: {batch_size}, Data workers: {num_workers}")
        elif "H100" in gpu_name or "H200" in gpu_name:
            batch_size = 512
            num_workers = 8
        elif gpu_mem >= 80:  # A100 80GB
            batch_size = 256
            num_workers = 8
        elif gpu_mem >= 40:  # A100 40GB, RTX 6000 Ada
            batch_size = 128
            num_workers = 6
        elif gpu_mem >= 24:
            batch_size = 64
            num_workers = 4
        elif gpu_mem >= 16:
            batch_size = 32
            num_workers = 4
        else:
            print(f"📊 Auto-adjusted batch size for {gpu_name} ({gpu_mem:.1f}GB VRAM): {batch_size}")

    return TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=1,
        learning_rate=8e-4,
        weight_decay=0.05,
        warmup_steps=1000,
        max_steps=50000,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=250,
        save_strategy="steps",
        save_steps=250,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_wer",
        greater_is_better=False,
        bf16=True,
        gradient_checkpointing=True,
        dataloader_num_workers=num_workers,
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=2,  # Optimize for A40's memory bandwidth
        dataloader_persistent_workers=True,  # Keep workers alive between epochs
        remove_unused_columns=False,
        label_names=["labels"],
        report_to=["tensorboard", "wandb"] if os.environ.get("WANDB_API_KEY") else ["tensorboard"],
        push_to_hub=bool(hf_write_token),
        hub_model_id="mazesmazes/asr" if hf_write_token else None,
        hub_strategy="checkpoint",
        seed=42,
    )


def main() -> None:
    """Main training function."""
    parser = argparse.ArgumentParser(description="ASR Training with Conformer-SmolLM2")
    parser.add_argument("--push-to-hub", action="store_true", help="Push model to HF Hub")
    parser.add_argument("--hub-model-id", type=str, default="mazesmazes/asr", help="Hub model ID")
    parser.add_argument("--max-steps", type=int, default=50000, help="Maximum training steps")
    parser.add_argument("--eval-steps", type=int, default=250, help="Evaluation frequency")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    args = parser.parse_args()

    # Initialize configs
    conformer_cfg = ConformerConfig()
    smollm2_cfg = SmolLM2Config()
    proj_cfg = ProjectorConfig()

    # Initialize model and tokenizer
    print("🚀 Initializing model and tokenizer...")
    model = ASRModel(conformer_cfg, smollm2_cfg, proj_cfg)
    tokenizer = model.decoder.tokenizer

    # Enable gradient checkpointing
    if hasattr(model.decoder.model, "gradient_checkpointing_enable"):
        if callable(model.decoder.model.gradient_checkpointing_enable):
            model.decoder.model.gradient_checkpointing_enable()
            print("✅ Gradient checkpointing enabled")

    # Compile model components for better performance
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        if "A40" in gpu_name:
            # A40 optimizations
            compile_mode = "reduce-overhead"  # Better for A40
            print(f"🚀 Compiling model for A40 with torch.compile (mode='{compile_mode}')...")
            model.encoder = torch.compile(model.encoder, mode=compile_mode)  # type: ignore
            model.audio_projector = torch.compile(model.audio_projector, mode=compile_mode)  # type: ignore
            # Don't compile decoder for A40 to avoid OOM
            print("✅ Model compilation complete (encoder and projector only for A40)")
        elif "H100" in gpu_name or "H200" in gpu_name or "A100" in gpu_name:
            compile_mode = "max-autotune"
            print(f"🚀 Compiling model with torch.compile (mode='{compile_mode}')...")
            model.encoder = torch.compile(model.encoder, mode=compile_mode)  # type: ignore
            model.audio_projector = torch.compile(model.audio_projector, mode=compile_mode)  # type: ignore
            model.decoder.model = torch.compile(model.decoder.model, mode="reduce-overhead")  # type: ignore
            print("✅ Model compilation complete")

    # Load datasets with optimized settings for 9 vCPUs
    print("📦 Loading datasets...")
    train_dataset = load_dataset(
        "librispeech_asr", "clean", split="train.100",
        cache_dir="/workspace/datasets", num_proc=8  # Use 8 of 9 CPUs for loading
    )
    val_dataset = load_dataset(
        "librispeech_asr", "clean", split="validation",
        cache_dir="/workspace/datasets", num_proc=8
    )
    print(f"✅ Datasets loaded. Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Create data collator
    data_collator = DataCollator(tokenizer)

    # Get training arguments
    training_args = get_training_args()
    if args.push_to_hub:
        training_args.push_to_hub = True
        training_args.hub_model_id = args.hub_model_id
    if args.max_steps:
        training_args.max_steps = args.max_steps
    if args.eval_steps:
        training_args.eval_steps = args.eval_steps
    if args.batch_size:
        training_args.per_device_train_batch_size = args.batch_size
        training_args.per_device_eval_batch_size = args.batch_size

    # Initialize trainer
    trainer = ASRTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # Start training
    print("🚀 Starting training...")
    trainer.train()

    # Save final model
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


if __name__ == "__main__":
    main()
