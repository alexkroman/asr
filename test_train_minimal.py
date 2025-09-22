import torch
from omegaconf import OmegaConf
import sys
sys.path.insert(0, 'src')

from train import create_asr_model

# Create minimal config
cfg = OmegaConf.create({
    "model": {
        "decoder_model_name": "HuggingFaceTB/SmolLM2-360M-Instruct",
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "lora_target_modules": ["q_proj", "v_proj"],
    }
})

print("Creating model...")
model = create_asr_model(cfg)
print(f"Model created with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")

# Test forward pass
print("\nTesting forward pass...")
dummy_input_features = torch.randn(1, 80, 3000)
dummy_audio_mask = torch.ones(1, 1500)

# Create input_ids with the audio_chunk token
tokenizer = model.decoder.tokenizer
dummy_text = model.INSTRUCTION_TEMPLATE + "Test transcription"
dummy_tokens = tokenizer(dummy_text, return_tensors="pt")

result = model(
    input_ids=dummy_tokens["input_ids"],
    attention_mask=dummy_tokens["attention_mask"],
    labels=dummy_tokens["input_ids"],
    input_features=dummy_input_features,
    audio_attention_mask=dummy_audio_mask,
)

print(f"Loss: {result.loss.item():.4f}")
print("Success!")
