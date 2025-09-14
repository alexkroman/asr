"""
🎙️ Conformer-SmolLM2 ASR - Final Optimized Version for H100/H200
- Uses Hugging Face Trainer for a simple, powerful training loop.
- Replaced custom data classes with the standard `datasets.map()` method.
- Uses torch.compile, fused optimizers, and bfloat16 for max performance.
- Configured for the LibriSpeech train.clean.100 dataset.
- Added H200-specific optimizations like FlashAttention and larger batch sizes.
"""

import argparse
import math
import os

# Set Hugging Face cache directory to /workspace BEFORE importing any HF libraries
os.environ["HF_HOME"] = "/workspace"
os.environ["HF_DATASETS_CACHE"] = "/workspace/datasets"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/workspace/hub"
os.environ["XDG_CACHE_HOME"] = "/workspace"  # Some libraries use this as fallback

# Optimize data downloading
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"  # Use faster hf_transfer for downloads
os.environ["HF_DATASETS_DOWNLOAD_MANAGER_MAX_WORKERS"] = "8"  # Parallel downloads

import re
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, cast, Tuple

# Import torch early to check for GPU
import torch
import evaluate
import numpy as np
import torch.nn as nn
import torchaudio.models as T_models
import torchaudio.transforms as T
from accelerate import Accelerator
from accelerate.utils import set_seed
from datasets import load_dataset
from einops import rearrange
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    logging,
)

warnings.filterwarnings("ignore")
logging.set_verbosity_error()
DATA_PATH = "/workspace/ASR_Conformer_SmolLM2_Optimized"
os.makedirs(f"{DATA_PATH}/checkpoints", exist_ok=True)
os.makedirs(f"{DATA_PATH}/models", exist_ok=True)
os.makedirs(f"{DATA_PATH}/logs", exist_ok=True)


# H200/A100 General Optimizations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
# 'high' precision is recommended for both TF32 on Ampere and for FlashAttention on Hopper
torch.set_float32_matmul_precision("high")

# H100/H200-specific optimizations
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    if "H100" in gpu_name or "H200" in gpu_name:
        print("🚀 H100/H200 detected - enabling Hopper-specific optimizations")
        # torch.compile mode='max-autotune' is ideal for Hopper
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
    print(
        "⚠️  No HF_WRITE_TOKEN or HF_READ_TOKEN found. " "Model upload will be skipped."
    )

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
        max_len = x.size(0) # Get the max sequence length from the time dimension
        
        # Create a mask of shape (Batch, Time)
        mask = torch.arange(max_len, device=x.device)[None, :] >= output_lengths[:, None]
        
        # Pass the tensor and the mask to the transformer encoder
        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        
        x = x.permute(1, 0, 2) # Switch back to (Batch, Time, Dim)
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
        self.model: Any = AutoModelForCausalLM.from_pretrained(
            config.model_name, torch_dtype=torch.bfloat16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "<pad>"})
            self.model.resize_token_embeddings(len(self.tokenizer))
            with torch.no_grad():
                embeddings = self.model.get_input_embeddings()
                if embeddings is not None and hasattr(embeddings, 'weight'):
                    # Type assertion for mypy
                    embedding_weight: torch.nn.Parameter = embeddings.weight  # type: ignore
                    # embedding_weight is a Parameter, we need to modify its data
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

    def forward(self, input_values: torch.Tensor, input_lengths: torch.Tensor, 
                    labels: Optional[torch.Tensor] = None, 
                    attention_mask: Optional[torch.Tensor] = None) -> Any:
            if self.training:
                input_values = self.spec_augment(input_values)

            # ✅ H200 OPTIMIZATION: Enable FlashAttention 2 via SDPA
            # This context manager tells PyTorch to use the most efficient attention backend
            # available (FlashAttention on Hopper GPUs) for the operations inside.
            with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
                audio_features = self.encoder(input_values, input_lengths)

            audio_prefix = self.audio_projector(audio_features)

            embeddings = self.decoder.model.get_input_embeddings()
            text_embeds = embeddings(labels) if callable(embeddings) else embeddings.forward(labels)
            combined_embeds = torch.cat([audio_prefix, text_embeds], dim=1)

            audio_mask = torch.ones(
                audio_prefix.shape[:2], dtype=torch.long, device=input_values.device
            )
            if attention_mask is not None:
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
class UnifiedDataCollator:
    """
    A unified data collator that performs all preprocessing steps on-the-fly.
    This includes filtering, feature extraction (spectrograms), text normalization,
    and batch padding for both audio and text.
    """
    tokenizer: Any
    sample_rate: int = 16000
    n_mels: int = 80
    max_audio_seconds: float = 20.0
    max_text_words: int = 150

    def __post_init__(self):
        # Initialize audio transforms once
        self.mel_transform = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=512,
            win_length=400,
            hop_length=160,
        )
        self.amp_to_db = T.AmplitudeToDB(stype="magnitude", top_db=80)

    def _normalize_text(self, text: str) -> str:
        # Simple text normalization
        return re.sub(r"[^\w\s'\-]", "", text.lower().strip())

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Step 1: Filter samples that are too long
        valid_features = []
        for f in features:
            try:
                audio_len_sec = len(f["audio"]["array"]) / self.sample_rate
                text_len_words = len(self._normalize_text(f["text"]).split())
                if (audio_len_sec <= self.max_audio_seconds and
                    text_len_words <= self.max_text_words):
                    valid_features.append(f)
            except Exception:
                # Skip samples with processing errors
                continue
        
        # If the entire batch is filtered out, return an empty dictionary
        if not valid_features:
            return {}

        # Step 2: Process audio to spectrograms and normalize texts
        specs, texts = [], []
        for f in valid_features:
            audio_array = torch.from_numpy(np.array(f["audio"]["array"], dtype=np.float32))
            spec = self.mel_transform(audio_array)
            spec_db = self.amp_to_db(spec)
            spec_norm = (spec_db - spec_db.mean()) / (spec_db.std() + 1e-8)
            specs.append(spec_norm)
            texts.append(self._normalize_text(f["text"]))

        # Step 3: Pad spectrograms to the same length within the batch
        input_lengths = torch.tensor([s.shape[1] for s in specs], dtype=torch.long)
        specs_transposed = [s.transpose(0, 1) for s in specs]
        padded_specs = torch.nn.utils.rnn.pad_sequence(
            specs_transposed, batch_first=True
        ).permute(0, 2, 1)

        # Step 4: Tokenize and pad text labels
        labels = self.tokenizer(
            texts, padding="longest", truncation=True, return_tensors="pt"
        )
        
        return {
            "input_values": padded_specs,
            "input_lengths": input_lengths,
            "labels": labels["input_ids"],
            "attention_mask": labels["attention_mask"],
        }


@dataclass
class TrainingConfig:
    output_dir: str = f"{DATA_PATH}/checkpoints"
    # Set a much larger default for H200
    per_device_train_batch_size: int = 512  
    per_device_eval_batch_size: int = 512
    gradient_accumulation_steps: int = 1  
    learning_rate: float = 8e-4
    weight_decay: float = 0.05
    warmup_steps: int = 1000
    max_steps: int = 50000
    logging_steps: int = 10
    eval_steps: int = 250
    save_steps: int = 250
    save_total_limit: int = 3
    mixed_precision: str = "bf16"
    gradient_checkpointing: bool = True
    seed: int = 42
    push_to_hub: bool = True
    hub_model_id: str = "mazesmazes/asr"
    hub_private: bool = False

    def __post_init__(self) -> None:
        # Adjust batch sizes based on available GPUs and VRAM
        if os.environ.get("RUNPOD_POD_ID") or torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            if gpu_count > 0:
                props = torch.cuda.get_device_properties(0)
                gpu_mem = props.total_memory / 1e9
                gpu_name = props.name

                # ✅ H200 OPTIMIZATION: Set a much larger batch size
                if "H100" in gpu_name or "H200" in gpu_name: # 80GB+ Hopper
                    self.per_device_train_batch_size = 512
                    self.per_device_eval_batch_size = 512
                    self.gradient_accumulation_steps = 1
                    self.gradient_checkpointing = True
                elif gpu_mem >= 80:  # 80GB Ampere (A100 80GB)
                    self.per_device_train_batch_size = 256
                    self.per_device_eval_batch_size = 256
                elif gpu_mem >= 40:  # 40-80GB VRAM (A100 40GB, A6000)
                    self.per_device_train_batch_size = 128
                    self.per_device_eval_batch_size = 128
                elif gpu_mem >= 24:  # 24-40GB VRAM
                    self.per_device_train_batch_size = 64
                    self.per_device_eval_batch_size = 64
                elif gpu_mem >= 16:  # 16-24GB VRAM
                    self.per_device_train_batch_size = 32
                    self.per_device_eval_batch_size = 48
                else:  # <16GB VRAM
                    self.per_device_train_batch_size = 16
                    self.per_device_eval_batch_size = 32

                print(f"📊 Auto-adjusted batch sizes for {gpu_name} ({gpu_mem:.1f}GB VRAM):")
                print(
                    f"   Train: {self.per_device_train_batch_size}, "
                    f"Eval: {self.per_device_eval_batch_size}"
                )


def train_with_accelerate() -> None:
    # Initialize configs
    conformer_cfg = ConformerConfig()
    smollm2_cfg = SmolLM2Config()
    proj_cfg = ProjectorConfig()
    training_cfg = TrainingConfig()

    # Initialize accelerator with appropriate logging
    from accelerate.utils import LoggerType

    log_with = [LoggerType.TENSORBOARD]
    if os.environ.get("WANDB_API_KEY"):
        log_with.append(LoggerType.WANDB)

    accelerator = Accelerator(
        mixed_precision=training_cfg.mixed_precision,
        gradient_accumulation_steps=training_cfg.gradient_accumulation_steps,
        log_with=log_with,
        project_dir=f"{DATA_PATH}/logs",
    )

    # Set seed for reproducibility
    set_seed(training_cfg.seed)

    if accelerator.is_main_process:
        print("🚀 Initializing model and tokenizer...")

    # Initialize model and tokenizer
    model = ASRModel(conformer_cfg, smollm2_cfg, proj_cfg)
    tokenizer = model.decoder.tokenizer

    if training_cfg.gradient_checkpointing:
        if hasattr(model.decoder.model, "gradient_checkpointing_enable"):
            model.decoder.model.gradient_checkpointing_enable()
            
        if accelerator.is_main_process:
            print("✅ Gradient checkpointing enabled for memory efficiency")
            
    # Compile model components
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        if "H100" in gpu_name or "H200" in gpu_name or "A100" in gpu_name:
            compile_mode = "max-autotune"
            if accelerator.is_main_process:
                print(
                    f"🚀 Compiling model with torch.compile "
                    f"(mode='{compile_mode}' for {gpu_name})..."
                )
            model.encoder = torch.compile(model.encoder, mode=compile_mode) # type: ignore
            model.audio_projector = torch.compile(model.audio_projector, mode=compile_mode) # type: ignore
            model.decoder.model = torch.compile(model.decoder.model, mode="reduce-overhead") # type: ignore
            if accelerator.is_main_process:
                print("✅ Model compilation complete")

    # Load raw datasets directly
    train_dataset = load_dataset("librispeech_asr", "clean", split="train.100",
                                 cache_dir="/workspace/datasets",
                                 num_proc=12)
    val_dataset = load_dataset("librispeech_asr", "clean", split="validation",
                               cache_dir="/workspace/datasets",
                               num_proc=12)

    # Reset default device after loading
    if torch.cuda.is_available():
        torch.set_default_device(None)
        
    if accelerator.is_main_process:
        print(f"✅ Raw datasets loaded. Train: {len(train_dataset)}, Val: {len(val_dataset)}")
        print("Preprocessing will be done on-the-fly by the data collator.")

    collator = UnifiedDataCollator(tokenizer)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=training_cfg.per_device_train_batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=1,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=training_cfg.per_device_eval_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=1,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True,
    )

    # Optimizer with fused kernels for H100/H200/A100
    use_fused = torch.cuda.is_available()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_cfg.learning_rate,
        weight_decay=training_cfg.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
        fused=use_fused,
    )
    if use_fused and accelerator.is_main_process:
        print(f"✅ Using fused AdamW optimizer for {torch.cuda.get_device_name(0)}")

    num_training_steps = training_cfg.max_steps
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=training_cfg.warmup_steps,
        num_training_steps=num_training_steps,
    )

    # Prepare with accelerator
    model, optimizer, train_dataloader, val_dataloader, lr_scheduler = (
        accelerator.prepare(
            model, optimizer, train_dataloader, val_dataloader, lr_scheduler
        )
    )

    # Initialize WER metric
    wer_metric = evaluate.load("wer")

    # Initialize tracking variables
    global_step = 0
    best_wer = float("inf")
    patience_counter = 0
    early_stopping_patience = 7

    # Training loop
    if accelerator.is_main_process:
        print("🚀 Starting training...")
        tracker_kwargs = {}
        if LoggerType.WANDB in log_with:
            tracker_kwargs["wandb"] = {
                "name": f"asr-training-{training_cfg.seed}",
                "tags": ["asr", "conformer", "smollm2"],
                "config": {
                    **training_cfg.__dict__,
                    **conformer_cfg.__dict__,
                    **smollm2_cfg.__dict__,
                    "gpu_count": torch.cuda.device_count(),
                    "gpu_name": (
                        torch.cuda.get_device_name(0)
                        if torch.cuda.is_available()
                        else "CPU"
                    ),
                },
            }
        accelerator.init_trackers(
            "asr_training", config=training_cfg.__dict__, init_kwargs=tracker_kwargs
        )

    model.train()
    progress_bar = tqdm(
        range(num_training_steps),
        disable=not accelerator.is_local_main_process,
        desc="Training",
    )

    for epoch in range(math.ceil(num_training_steps / len(train_dataloader))):
        for batch in train_dataloader:
            # If the collator returned an empty batch, skip it
            if not batch:
                continue

            with accelerator.accumulate(model):
                outputs = model(
                    input_values=batch["input_values"],
                    input_lengths=batch["input_lengths"],
                    labels=batch["labels"],
                    attention_mask=batch["attention_mask"],
                )
                loss = outputs.loss

                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if global_step % training_cfg.logging_steps == 0:
                accelerator.log(
                    {
                        "train_loss": loss.detach().item(),
                        "learning_rate": lr_scheduler.get_last_lr()[0],
                        "epoch": epoch,
                    },
                    step=global_step,
                )

            if global_step % training_cfg.eval_steps == 0 and global_step > 0:
                model.eval()
                val_loss = 0
                all_predictions = []
                all_references = []

                with torch.no_grad():
                    for val_batch in tqdm(
                        val_dataloader,
                        desc="Evaluating",
                        disable=not accelerator.is_local_main_process,
                    ):
                        if not val_batch:
                            continue
                        
                        outputs = model(
                            input_values=val_batch["input_values"],
                            input_lengths=val_batch["input_lengths"],
                            labels=val_batch["labels"],
                            attention_mask=val_batch["attention_mask"],
                        )
                        val_loss += outputs.loss.item()

                        generated_ids = accelerator.unwrap_model(model).generate(
                            input_values=val_batch["input_values"],
                            input_lengths=val_batch["input_lengths"],
                            max_new_tokens=150,
                            num_beams=5,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                        )

                        predictions = tokenizer.batch_decode(
                            generated_ids, skip_special_tokens=True
                        )
                        references = tokenizer.batch_decode(
                            val_batch["labels"], skip_special_tokens=True
                        )

                        all_predictions.extend(predictions)
                        all_references.extend(references)

                avg_val_loss = val_loss / len(val_dataloader) if len(val_dataloader) > 0 else 0
                wer_score = wer_metric.compute(
                    predictions=all_predictions, references=all_references
                )

                if accelerator.is_main_process:
                    print(
                        f"\nStep {global_step}: Val Loss: {avg_val_loss:.4f}, "
                        f"WER: {wer_score:.4f}"
                    )
                    accelerator.log(
                        {"val_loss": avg_val_loss, "wer": wer_score},
                        step=global_step,
                    )

                    if wer_score is not None and float(wer_score) < best_wer:
                        best_wer = wer_score
                        patience_counter = 0
                        accelerator.save_state(f"{training_cfg.output_dir}/best_model")
                        print(f"✅ New best WER: {best_wer:.4f}. Model saved.")
                    else:
                        patience_counter += 1
                        if patience_counter >= early_stopping_patience:
                            print(
                                f"Early stopping triggered. Best WER: {best_wer:.4f}"
                            )
                            break
                model.train()

            if global_step % training_cfg.save_steps == 0 and global_step > 0:
                if accelerator.is_main_process:
                    accelerator.save_state(
                        f"{training_cfg.output_dir}/checkpoint-{global_step}"
                    )
                    print(f"✅ Checkpoint saved at step {global_step}")

            progress_bar.update(1)
            global_step += 1

            if global_step >= num_training_steps:
                break

        if (global_step >= num_training_steps or 
            patience_counter >= early_stopping_patience):
            break

    accelerator.end_training()

    if accelerator.is_main_process:
        print(f"\n✅ Training completed! Best WER: {best_wer:.4f}")
        accelerator.save_state(f"{training_cfg.output_dir}/final_model")
        unwrapped_model = accelerator.unwrap_model(model)
        save_path = f"{DATA_PATH}/models/final_model"
        unwrapped_model.save_pretrained(
            save_path,
            save_function=accelerator.save,
            state_dict=accelerator.get_state_dict(model),
        )
        tokenizer.save_pretrained(save_path)
        print(f"✅ Model saved to {save_path}")

        if training_cfg.push_to_hub and hf_write_token:
            try:
                from huggingface_hub import HfApi

                api = HfApi(token=hf_write_token)
                repo_id = training_cfg.hub_model_id
                print(f"📤 Uploading model to Hugging Face Hub: {repo_id}")
                api.upload_folder(
                    folder_path=save_path,
                    repo_id=repo_id,
                    repo_type="model",
                    token=hf_write_token,
                )
                model_card = f"""---
language: en
license: apache-2.0
tags:
- automatic-speech-recognition
- conformer
- smollm2
- librispeech
metrics:
- wer
model-index:
- name: {training_cfg.hub_model_id}
  results:
  - task:
      type: automatic-speech-recognition
    dataset:
      type: librispeech_asr
      name: LibriSpeech
      config: clean
      split: validation
    metrics:
    - type: wer
      value: {best_wer:.4f}
---
# Conformer-SmolLM2 ASR Model
This model combines a Conformer encoder with SmolLM2 decoder for automatic
speech recognition.
## Training Details
- **Dataset**: LibriSpeech train-clean-100
- **Best WER**: {best_wer:.4f}
- **Training Device**: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}
- **GPU Count**: {torch.cuda.device_count()}
"""
                api.upload_file(
                    path_or_fileobj=model_card.encode(),
                    path_in_repo="README.md",
                    repo_id=repo_id,
                    repo_type="model",
                    token=hf_write_token,
                )
                print(f"✅ Model uploaded to: https://huggingface.co/{repo_id}")
            except Exception as e:
                print(f"⚠️  Failed to upload to Hugging Face Hub: {e}")


def main() -> None:
    """Main entry point for the training script."""
    parser = argparse.ArgumentParser(description="ASR Training with Conformer-SmolLM2")
    parser.add_argument(
        "--push-to-hub", action="store_true", help="Push model to Hugging Face Hub"
    )
    parser.add_argument(
        "--hub-model-id",
        type=str,
        default="mazesmazes/asr",
        help="Model ID for Hugging Face Hub",
    )
    parser.add_argument(
        "--max-steps", type=int, default=50000, help="Maximum training steps"
    )
    parser.add_argument(
        "--eval-steps", type=int, default=250, help="Evaluation frequency"
    )
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Override automatic batch size"
    )
    args = parser.parse_args()

    # Override config with command line arguments
    training_cfg_instance = TrainingConfig()
    if args.push_to_hub:
        training_cfg_instance.push_to_hub = True
    if args.hub_model_id:
        training_cfg_instance.hub_model_id = args.hub_model_id
    if args.max_steps:
        training_cfg_instance.max_steps = args.max_steps
    if args.eval_steps:
        training_cfg_instance.eval_steps = args.eval_steps
    if args.batch_size:
        training_cfg_instance.per_device_train_batch_size = args.batch_size
        training_cfg_instance.per_device_eval_batch_size = args.batch_size

    train_with_accelerate()


if __name__ == "__main__":
    main()