"""
🎙️ Conformer-SmolLM2 ASR - Final Optimized Version
- Uses Hugging Face Trainer for a simple, powerful training loop.
- Replaced custom data classes with the standard `datasets.map()` method.
- Uses torch.compile, fused optimizers, and bfloat16 for max performance.
- Configured for the LibriSpeech train.clean.100 dataset.
"""

import subprocess
import sys
import os
import warnings
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import math
from tqdm.auto import tqdm
from pathlib import Path

warnings.filterwarnings('ignore')
from transformers import logging

logging.set_verbosity_error()

# Environment variables must be set on the host

# OPTIMIZATION: Added flash-attn for automatic memory-efficient attention dispatch.
# Skip package installation on RunPod (handled by setup script)
if not os.environ.get('RUNPOD_POD_ID'):
    packages = [
        "torch", "torchaudio", "torchvision", "datasets",
        "tokenizers", "jiwer", "huggingface_hub", "sentencepiece",
        "transformers>=4.41.0", "accelerate", "einops", "peft", "numpy",
        "tensorboard", "evaluate"
    ]
    # Only install flash-attn on Linux (not macOS)
    if sys.platform == "linux":
        packages.append("flash-attn")
    print("Installing/upgrading packages...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "--upgrade"] + packages)
    print("✅ Packages ready.")

# Detect environment and set paths accordingly
if os.environ.get('RUNPOD_POD_ID'):
    # RunPod environment
    DATA_PATH = "/workspace/ASR_Conformer_SmolLM2_Optimized"
    print(f"🚀 Running on RunPod (Pod ID: {os.environ.get('RUNPOD_POD_ID')})")
    print(f"📊 GPU Count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    DATA_PATH = "./ASR_Conformer_SmolLM2_Optimized"
    print("💻 Running on local machine")

os.makedirs(f"{DATA_PATH}/checkpoints", exist_ok=True)
os.makedirs(f"{DATA_PATH}/models", exist_ok=True)
os.makedirs(f"{DATA_PATH}/logs", exist_ok=True)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import torchaudio.models as T_models
from datasets import load_dataset
import jiwer
import re
import numpy as np
import evaluate
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    get_linear_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model, TaskType
from einops import rearrange
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader

# OPTIMIZATION: Ensure TF32 is enabled and set CudNN to benchmark mode for A100 performance.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.set_float32_matmul_precision('medium')

# Handle Hugging Face authentication
hf_read_token = os.environ.get('HF_READ_TOKEN')
hf_write_token = os.environ.get('HF_WRITE_TOKEN') or os.environ.get('HF_TOKEN')

if hf_read_token:
    from huggingface_hub import login
    login(token=hf_read_token)
    print("✅ Logged in to Hugging Face Hub with read token")
elif hf_write_token:
    from huggingface_hub import login
    login(token=hf_write_token)
    print("✅ Logged in to Hugging Face Hub with write token")
else:
    print("⚠️  No HF_TOKEN found. Model upload will be skipped.")

# Optional: Setup WandB if available
if os.environ.get('WANDB_API_KEY'):
    import wandb
    wandb.login(key=os.environ.get('WANDB_API_KEY'))
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
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "v_proj", "k_proj", "o_proj"
    ])

@dataclass
class ProjectorConfig:
    num_queries: int = 24
    num_heads: int = 8
    dropout: float = 0.1

class SpecAugment(nn.Module):
    def __init__(self, freq_mask_param=27, time_mask_param=100, n_freq_masks=2, n_time_masks=2):
        super().__init__()
        self.freq_masks = nn.ModuleList([T.FrequencyMasking(freq_mask_param=freq_mask_param) for _ in range(n_freq_masks)])
        self.time_masks = nn.ModuleList([T.TimeMasking(time_mask_param=time_mask_param) for _ in range(n_time_masks)])

    def forward(self, x):
        for freq_mask in self.freq_masks: x = freq_mask(x)
        for time_mask in self.time_masks: x = time_mask(x)
        return x

class ConformerEncoder(nn.Module):
    def __init__(self, config: ConformerConfig):
        super().__init__()
        self.config = config
        self.subsample = nn.Sequential(
            nn.Conv2d(1, config.d_model, 3, 2, 1), nn.SiLU(),
            nn.Conv2d(config.d_model, config.d_model, 3, 2, 1), nn.SiLU()
        )
        self.input_proj = nn.Linear(config.d_model * (config.n_mels // 4), config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.conformer = T_models.Conformer(
            input_dim=config.d_model, num_heads=config.n_head,
            ffn_dim=config.d_model * 4, num_layers=config.num_layers,
            depthwise_conv_kernel_size=config.kernel_size, dropout=config.dropout,
        )

    def forward(self, x, input_lengths):
        x = self.subsample(x.unsqueeze(1))
        x = rearrange(x, 'b c f t -> b t (c f)')
        x = self.input_proj(x)
        x = self.dropout(x)
        x = x.permute(1, 0, 2)
        output_lengths = input_lengths // 4
        x, _ = self.conformer(x, output_lengths)
        x = x.permute(1, 0, 2)
        return x

class LightweightAudioProjector(nn.Module):
    def __init__(self, audio_dim: int, text_dim: int, config: ProjectorConfig):
        super().__init__()
        self.audio_proj = nn.Linear(audio_dim, text_dim)
        self.queries = nn.Parameter(torch.randn(config.num_queries, text_dim))
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=text_dim, num_heads=config.num_heads,
            dropout=config.dropout, batch_first=True
        )
        self.mlp = nn.Sequential(
            nn.LayerNorm(text_dim), nn.Linear(text_dim, text_dim * 2),
            nn.GELU(), nn.Dropout(config.dropout),
            nn.Linear(text_dim * 2, text_dim), nn.LayerNorm(text_dim)
        )

    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        B = audio_features.shape[0]
        audio_proj = self.audio_proj(audio_features)
        queries = self.queries.unsqueeze(0).expand(B, -1, -1)
        attn_out, _ = self.cross_attn(queries, audio_proj, audio_proj)
        return self.mlp(attn_out + queries)

class SmolLM2Decoder(nn.Module):
    def __init__(self, config: SmolLM2Config):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(config.model_name, torch_dtype=torch.bfloat16)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '<pad>'})
            self.model.resize_token_embeddings(len(self.tokenizer))
            with torch.no_grad():
                self.model.get_input_embeddings().weight[-1] = self.model.get_input_embeddings().weight[:-1].mean(dim=0)
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        if config.use_lora:
            lora_config = LoraConfig(
                r=config.lora_r, lora_alpha=config.lora_alpha, target_modules=config.lora_target_modules,
                lora_dropout=config.lora_dropout, bias="none", task_type=TaskType.CAUSAL_LM,
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()

class ASRModel(nn.Module):
    def __init__(self, conformer_cfg, smollm2_cfg, proj_cfg):
        super().__init__()
        self.encoder = ConformerEncoder(conformer_cfg)
        self.decoder = SmolLM2Decoder(smollm2_cfg)
        self.audio_projector = LightweightAudioProjector(
            conformer_cfg.d_model, self.decoder.model.config.hidden_size, proj_cfg
        )
        self.spec_augment = SpecAugment()

    def forward(self, input_values, input_lengths, labels=None, attention_mask=None):
        if self.training:
            input_values = self.spec_augment(input_values)

        audio_features = self.encoder(input_values, input_lengths)
        audio_prefix = self.audio_projector(audio_features)

        text_embeds = self.decoder.model.get_input_embeddings()(labels)
        combined_embeds = torch.cat([audio_prefix, text_embeds], dim=1)
        
        audio_mask = torch.ones(audio_prefix.shape[:2], dtype=torch.long, device=input_values.device)
        combined_attention_mask = torch.cat([audio_mask, attention_mask], dim=1)
        
        return self.decoder.model(
            inputs_embeds=combined_embeds,
            attention_mask=combined_attention_mask,
            labels=labels
        )

    @torch.inference_mode()
    def generate(self, input_values, input_lengths, **kwargs):
        audio_features = self.encoder(input_values, input_lengths)
        audio_prefix = self.audio_projector(audio_features)
        return self.decoder.model.generate(inputs_embeds=audio_prefix, **kwargs)

class AudioDataProcessor:
    def __init__(self, tokenizer, sample_rate=16000, n_mels=80, max_audio_seconds=None, max_text_words=None):
        self.tokenizer = tokenizer
        self.sample_rate = sample_rate
        self.max_audio_seconds = max_audio_seconds
        self.max_text_words = max_text_words
        self.mel_transform = T.MelSpectrogram(sample_rate=sample_rate, n_mels=n_mels, n_fft=512, win_length=400, hop_length=160)
        self.amp_to_db = T.AmplitudeToDB(stype='magnitude', top_db=80)

    def _normalize_text(self, text):
        return re.sub(r"[^\w\s'\-]", '', text.lower().strip())

    def process_sample(self, sample):
        try:
            audio_array = np.array(sample["audio"]["array"], dtype=np.float32)
            if self.max_audio_seconds and (len(audio_array) / self.sample_rate > self.max_audio_seconds): return None
            clean_text = self._normalize_text(sample["text"])
            if self.max_text_words and (len(clean_text.split()) > self.max_text_words): return None
            spec = self.mel_transform(torch.from_numpy(audio_array))
            spec_db = self.amp_to_db(spec)
            spec_norm = (spec_db - spec_db.mean()) / (spec_db.std() + 1e-8)
            # The map function needs serializable outputs
            return {"spectrogram": spec_norm.numpy(), "text": clean_text}
        except Exception:
            return None

@dataclass
class DataCollator:
    tokenizer: Any
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Convert numpy arrays back to tensors
        specs = [torch.from_numpy(f["spectrogram"]) for f in features]
        texts = [f["text"] for f in features]

        input_lengths = torch.tensor([s.shape[1] for s in specs], dtype=torch.long)
        
        specs_transposed = [s.transpose(0, 1) for s in specs]
        padded_specs = torch.nn.utils.rnn.pad_sequence(specs_transposed, batch_first=True)
        padded_specs = padded_specs.permute(0, 2, 1)

        labels = self.tokenizer(texts, padding='longest', truncation=True, return_tensors='pt')
        
        return {
            "input_values": padded_specs,
            "input_lengths": input_lengths,
            "labels": labels["input_ids"],
            "attention_mask": labels["attention_mask"]
        }

@dataclass
class TrainingConfig:
    output_dir: str = f"{DATA_PATH}/checkpoints"
    per_device_train_batch_size: int = 96
    per_device_eval_batch_size: int = 64
    gradient_accumulation_steps: int = 2
    learning_rate: float = 6e-4
    weight_decay: float = 0.05
    warmup_steps: int = 2000
    max_steps: int = 50000
    logging_steps: int = 10
    eval_steps: int = 250
    save_steps: int = 250
    save_total_limit: int = 3
    mixed_precision: str = "bf16"
    gradient_checkpointing: bool = False
    seed: int = 42
    push_to_hub: bool = True  # Enable by default
    hub_model_id: str = "mazesmazes/asr"  # Your HF repository
    hub_private: bool = False  # Public repository
    
    def __post_init__(self):
        # Adjust batch sizes based on available GPUs and VRAM
        if os.environ.get('RUNPOD_POD_ID'):
            gpu_count = torch.cuda.device_count()
            if gpu_count > 0:
                # Get GPU memory in GB
                gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
                
                # Adjust batch size based on GPU memory
                if gpu_mem < 16:  # <16GB VRAM
                    self.per_device_train_batch_size = 16
                    self.per_device_eval_batch_size = 32
                elif gpu_mem < 24:  # 16-24GB VRAM
                    self.per_device_train_batch_size = 32
                    self.per_device_eval_batch_size = 48
                elif gpu_mem < 40:  # 24-40GB VRAM
                    self.per_device_train_batch_size = 64
                    self.per_device_eval_batch_size = 64
                else:  # 40GB+ VRAM (A100, H100)
                    self.per_device_train_batch_size = 96
                    self.per_device_eval_batch_size = 96
                
                print(f"📊 Auto-adjusted batch sizes for {gpu_mem:.1f}GB VRAM:")
                print(f"   Train: {self.per_device_train_batch_size}, Eval: {self.per_device_eval_batch_size}")

def train_with_accelerate():
    # Initialize configs
    conformer_cfg = ConformerConfig()
    smollm2_cfg = SmolLM2Config()
    proj_cfg = ProjectorConfig()
    training_cfg = TrainingConfig()
    
    # Initialize accelerator with appropriate logging
    log_with = ["tensorboard"]
    if os.environ.get('WANDB_API_KEY'):
        log_with.append("wandb")
    
    accelerator = Accelerator(
        mixed_precision=training_cfg.mixed_precision,
        gradient_accumulation_steps=training_cfg.gradient_accumulation_steps,
        log_with=log_with,
        project_dir=f"{DATA_PATH}/logs"
    )
    
    # Set seed for reproducibility
    set_seed(training_cfg.seed)
    
    if accelerator.is_main_process:
        print("🚀 Initializing model and tokenizer...")
    
    # Initialize model and tokenizer
    model = ASRModel(conformer_cfg, smollm2_cfg, proj_cfg)
    tokenizer = model.decoder.tokenizer
    
    # Compile model components if not using gradient checkpointing
    if not training_cfg.gradient_checkpointing and accelerator.is_main_process:
        print("🚀 Compiling model components with torch.compile...")
        model.encoder = torch.compile(model.encoder)
        model.audio_projector = torch.compile(model.audio_projector)
        model.decoder.model = torch.compile(model.decoder.model)
        print("✅ Model compilation complete.")

    # Data processing
    processor = AudioDataProcessor(
        tokenizer,
        max_audio_seconds=15.0,
        max_text_words=100
    )

    def preprocess_function(examples):
        return processor.process_sample(examples)

    if accelerator.is_main_process:
        print("Loading and preprocessing datasets...")
    
    train_dataset = load_dataset("librispeech_asr", "clean", split="train.100")
    val_dataset = load_dataset("librispeech_asr", "clean", split="validation")
    
    # Preprocess datasets
    with accelerator.main_process_first():
        train_dataset = train_dataset.map(
            preprocess_function, 
            num_proc=4 if accelerator.num_processes == 1 else 1,
            remove_columns=train_dataset.column_names
        ).filter(lambda x: x['spectrogram'] is not None)
        
        val_dataset = val_dataset.map(
            preprocess_function,
            num_proc=4 if accelerator.num_processes == 1 else 1,
            remove_columns=val_dataset.column_names
        ).filter(lambda x: x['spectrogram'] is not None)
    
    if accelerator.is_main_process:
        print(f"✅ Datasets ready. Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Create data collator and loaders
    collator = DataCollator(tokenizer)
    
    train_dataloader = DataLoader(
        train_dataset.with_format("torch"),
        batch_size=training_cfg.per_device_train_batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=4,
        pin_memory=True
    )
    
    val_dataloader = DataLoader(
        val_dataset.with_format("torch"),
        batch_size=training_cfg.per_device_eval_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=4,
        pin_memory=True
    )

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_cfg.learning_rate,
        weight_decay=training_cfg.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    num_training_steps = training_cfg.max_steps
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=training_cfg.warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # Prepare with accelerator
    model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, lr_scheduler
    )
    
    # Initialize WER metric
    wer_metric = evaluate.load("wer")
    
    # Initialize tracking variables
    global_step = 0
    best_wer = float('inf')
    patience_counter = 0
    early_stopping_patience = 7

    # Training loop
    if accelerator.is_main_process:
        print("🚀 Starting training...")
        tracker_kwargs = {}
        if "wandb" in log_with:
            tracker_kwargs["wandb"] = {
                "name": f"asr-training-{training_cfg.seed}",
                "tags": ["asr", "conformer", "smollm2"],
                "config": {
                    **training_cfg.__dict__,
                    **conformer_cfg.__dict__,
                    **smollm2_cfg.__dict__,
                    "gpu_count": torch.cuda.device_count(),
                    "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
                }
            }
        accelerator.init_trackers("asr_training", config=training_cfg.__dict__, init_kwargs=tracker_kwargs)
    
    model.train()
    progress_bar = tqdm(
        range(num_training_steps),
        disable=not accelerator.is_local_main_process,
        desc="Training"
    )
    
    for epoch in range(math.ceil(num_training_steps / len(train_dataloader))):
        for batch in train_dataloader:
            with accelerator.accumulate(model):
                # Forward pass
                outputs = model(
                    input_values=batch["input_values"],
                    input_lengths=batch["input_lengths"],
                    labels=batch["labels"],
                    attention_mask=batch["attention_mask"]
                )
                loss = outputs.loss
                
                # Backward pass
                accelerator.backward(loss)
                
                # Gradient clipping
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # Logging
            if global_step % training_cfg.logging_steps == 0:
                accelerator.log(
                    {
                        "train_loss": loss.detach().item(),
                        "learning_rate": lr_scheduler.get_last_lr()[0],
                        "epoch": epoch,
                    },
                    step=global_step
                )
            
            # Evaluation
            if global_step % training_cfg.eval_steps == 0 and global_step > 0:
                model.eval()
                val_loss = 0
                all_predictions = []
                all_references = []
                
                with torch.no_grad():
                    for val_batch in tqdm(
                        val_dataloader,
                        desc="Evaluating",
                        disable=not accelerator.is_local_main_process
                    ):
                        outputs = model(
                            input_values=val_batch["input_values"],
                            input_lengths=val_batch["input_lengths"],
                            labels=val_batch["labels"],
                            attention_mask=val_batch["attention_mask"]
                        )
                        val_loss += outputs.loss.item()
                        
                        # Generate predictions
                        generated_ids = accelerator.unwrap_model(model).generate(
                            input_values=val_batch["input_values"],
                            input_lengths=val_batch["input_lengths"],
                            max_new_tokens=150,
                            num_beams=5,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                        )
                        
                        # Decode predictions and references
                        predictions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                        references = tokenizer.batch_decode(val_batch["labels"], skip_special_tokens=True)
                        
                        all_predictions.extend(predictions)
                        all_references.extend(references)
                
                # Calculate WER
                avg_val_loss = val_loss / len(val_dataloader)
                wer_score = wer_metric.compute(predictions=all_predictions, references=all_references)
                
                if accelerator.is_main_process:
                    print(f"\nStep {global_step}: Val Loss: {avg_val_loss:.4f}, WER: {wer_score:.4f}")
                    accelerator.log(
                        {
                            "val_loss": avg_val_loss,
                            "wer": wer_score,
                        },
                        step=global_step
                    )
                    
                    # Early stopping check
                    if wer_score < best_wer:
                        best_wer = wer_score
                        patience_counter = 0
                        # Save best model
                        accelerator.save_state(f"{training_cfg.output_dir}/best_model")
                        print(f"✅ New best WER: {best_wer:.4f}. Model saved.")
                    else:
                        patience_counter += 1
                        if patience_counter >= early_stopping_patience:
                            print(f"Early stopping triggered. Best WER: {best_wer:.4f}")
                            break
                
                model.train()
            
            # Save checkpoint
            if global_step % training_cfg.save_steps == 0 and global_step > 0:
                if accelerator.is_main_process:
                    accelerator.save_state(f"{training_cfg.output_dir}/checkpoint-{global_step}")
                    print(f"✅ Checkpoint saved at step {global_step}")
            
            progress_bar.update(1)
            global_step += 1
            
            if global_step >= num_training_steps:
                break
        
        if global_step >= num_training_steps or patience_counter >= early_stopping_patience:
            break
    
    # End training
    accelerator.end_training()
    
    if accelerator.is_main_process:
        print(f"\n✅ Training completed! Best WER: {best_wer:.4f}")
        # Save final model
        accelerator.save_state(f"{training_cfg.output_dir}/final_model")
        
        # Save model in Hugging Face format
        unwrapped_model = accelerator.unwrap_model(model)
        save_path = f"{DATA_PATH}/models/final_model"
        unwrapped_model.save_pretrained(
            save_path,
            save_function=accelerator.save,
            state_dict=accelerator.get_state_dict(model)
        )
        tokenizer.save_pretrained(save_path)
        print(f"✅ Model saved to {save_path}")
        
        # Push to Hugging Face Hub if configured
        if training_cfg.push_to_hub and hf_write_token:
            try:
                from huggingface_hub import HfApi
                api = HfApi(token=hf_write_token)
                
                # Use the hub_model_id directly (already includes username)
                repo_id = training_cfg.hub_model_id
                
                # Upload model and tokenizer (assumes repo already exists)
                print(f"📤 Uploading model to Hugging Face Hub: {repo_id}")
                api.upload_folder(
                    folder_path=save_path,
                    repo_id=repo_id,
                    repo_type="model",
                    token=hf_write_token
                )
                
                # Create model card
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

This model combines a Conformer encoder with SmolLM2 decoder for automatic speech recognition.

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
                    token=hf_write_token
                )
                
                print(f"✅ Model uploaded to: https://huggingface.co/{repo_id}")
            except Exception as e:
                print(f"⚠️  Failed to upload to Hugging Face Hub: {e}")

if __name__ == '__main__':
    # Parse command line arguments for RunPod
    import argparse
    parser = argparse.ArgumentParser(description="ASR Training with Conformer-SmolLM2")
    parser.add_argument("--push-to-hub", action="store_true", help="Push model to Hugging Face Hub")
    parser.add_argument("--hub-model-id", type=str, default="mazesmazes/asr", help="Model ID for Hugging Face Hub")
    parser.add_argument("--max-steps", type=int, default=50000, help="Maximum training steps")
    parser.add_argument("--eval-steps", type=int, default=250, help="Evaluation frequency")
    parser.add_argument("--batch-size", type=int, default=None, help="Override automatic batch size")
    
    args = parser.parse_args()
    
    # Override config with command line arguments
    if args.push_to_hub:
        TrainingConfig.push_to_hub = True
    if args.hub_model_id:
        TrainingConfig.hub_model_id = args.hub_model_id
    if args.max_steps:
        TrainingConfig.max_steps = args.max_steps
    if args.eval_steps:
        TrainingConfig.eval_steps = args.eval_steps
    if args.batch_size:
        TrainingConfig.per_device_train_batch_size = args.batch_size
        TrainingConfig.per_device_eval_batch_size = args.batch_size
    
    train_with_accelerate()