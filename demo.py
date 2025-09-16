#!/usr/bin/env python3
"""
Gradio demo for ASR model - record or upload audio for transcription.
"""

import gradio as gr
import torch
import torchaudio
import numpy as np
from pathlib import Path
import tempfile
import os
import sys

# Add src to path for local imports
sys.path.append(str(Path(__file__).parent / "src"))

# Import model components
from train import ASRModel, ModelArguments
from transformers import AutoTokenizer


class ASRDemo:
    def __init__(self, model_path="mazesmazes/asr", use_local=False):
        """Initialize the ASR demo with model and tokenizer."""
        # Determine best available device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        print(f"Using device: {self.device}")

        if use_local:
            # Load from local checkpoint
            print(f"Loading model from local path: {model_path}")
            checkpoint_path = Path(model_path)
            self.load_from_checkpoint(checkpoint_path)
        else:
            # Load from Hugging Face Hub
            print(f"Loading model from Hugging Face: {model_path}")
            from huggingface_hub import snapshot_download

            # Download model files
            model_dir = snapshot_download(repo_id=model_path)
            self.load_from_checkpoint(Path(model_dir))

        self.model = self.model.to(self.device)
        self.model.eval()

        # Audio parameters
        self.sample_rate = 16000
        print("Model loaded successfully!")

    def load_from_checkpoint(self, checkpoint_path: Path):
        """Load model from checkpoint directory."""
        import json

        # Load config
        config_path = checkpoint_path / "model_config.json"
        if not config_path.exists():
            config_path = checkpoint_path / "config.json"

        if config_path.exists():
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
        else:
            # Use default config - matching the actual production training config
            config_dict = {
                "encoder_dim": 512,
                "num_encoder_layers": 12,
                "num_attention_heads": 8,
                "feed_forward_expansion_factor": 4,
                "conv_expansion_factor": 2,
                "input_dropout_p": 0.1,
                "feed_forward_dropout_p": 0.1,
                "attention_dropout_p": 0.1,
                "conv_dropout_p": 0.1,
                "conv_kernel_size": 15,  # From production config
                "half_step_residual": True,
                "decoder_model_id": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
                "use_lora": True,
                "lora_rank": 32,
                "lora_alpha": 32,  # From production config
                "lora_dropout": 0.05,  # From production config
                "lora_target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
                "projector_type": "deep",
                "num_projector_layers": 2,  # From production config
                "projector_dropout": 0.1,
                "num_queries": 128,  # From production config
                "projector_num_heads": 8  # From production config
            }

        # Create ModelArguments from config
        config = ModelArguments(**{k: v for k, v in config_dict.items()
                                   if k in ModelArguments.__dataclass_fields__})

        # Initialize model
        self.model = ASRModel(config)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_path))

        # Load model weights
        model_path = checkpoint_path / "pytorch_model.bin"
        if model_path.exists():
            print(f"Loading model weights from {model_path}")
            state_dict = torch.load(model_path, map_location="cpu")
            self.model.load_state_dict(state_dict, strict=False)
        else:
            # Try loading individual components
            encoder_path = checkpoint_path / "encoder.bin"
            if encoder_path.exists():
                print(f"Loading encoder from {encoder_path}")
                encoder_state = torch.load(encoder_path, map_location="cpu")
                self.model.encoder.load_state_dict(encoder_state, strict=False)

            projector_path = checkpoint_path / "audio_projector.bin"
            if not projector_path.exists():
                projector_path = checkpoint_path / "projector.bin"
            if projector_path.exists():
                print(f"Loading projector from {projector_path}")
                projector_state = torch.load(projector_path, map_location="cpu")
                self.model.audio_projector.load_state_dict(projector_state, strict=False)

            # Load decoder/LoRA weights
            decoder_path = checkpoint_path / "decoder"
            if decoder_path.exists() and decoder_path.is_dir():
                print(f"Loading decoder from {decoder_path}")
                from peft import PeftModel
                self.model.decoder.model = PeftModel.from_pretrained(
                    self.model.decoder.model,
                    str(decoder_path)
                )

    def preprocess_audio(self, audio_path):
        """Load and preprocess audio file."""
        # Load audio
        waveform, sample_rate = torchaudio.load(audio_path)

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample if necessary
        if sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=self.sample_rate
            )
            waveform = resampler(waveform)

        return waveform

    def transcribe_audio(self, audio_input):
        """Transcribe audio input (file path or tuple from Gradio)."""
        if audio_input is None:
            return "Please record or upload an audio file."

        try:
            # Handle Gradio audio input (tuple: sample_rate, numpy array)
            if isinstance(audio_input, tuple):
                sample_rate, audio_data = audio_input

                # Convert to tensor
                if isinstance(audio_data, np.ndarray):
                    # Handle stereo to mono
                    if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
                        audio_data = np.mean(audio_data, axis=1)

                    waveform = torch.tensor(audio_data, dtype=torch.float32)

                    # Normalize if needed
                    if waveform.dtype == torch.int16:
                        waveform = waveform.float() / 32768.0
                    elif waveform.dtype == torch.int32:
                        waveform = waveform.float() / 2147483648.0

                    # Add batch dimension
                    if waveform.dim() == 1:
                        waveform = waveform.unsqueeze(0)

                    # Resample if necessary
                    if sample_rate != self.sample_rate:
                        resampler = torchaudio.transforms.Resample(
                            orig_freq=sample_rate,
                            new_freq=self.sample_rate
                        )
                        waveform = resampler(waveform)
                else:
                    waveform = audio_data

            # Handle file path input
            elif isinstance(audio_input, str):
                waveform = self.preprocess_audio(audio_input)

            else:
                return "Unsupported audio input format."

            # Move to device
            waveform = waveform.to(self.device)

            # Run inference
            with torch.no_grad():
                transcription = self.model.transcribe(waveform)

            return transcription

        except Exception as e:
            return f"Error during transcription: {str(e)}"


def create_demo(model_path="mazesmazes/asr", use_local=False):
    """Create and return Gradio demo interface."""

    # Initialize ASR model
    print("Initializing ASR model...")
    asr_demo = ASRDemo(model_path=model_path, use_local=use_local)

    # Create Gradio interface
    with gr.Blocks(title="ASR Demo - Conformer + SmolLM2") as demo:
        gr.Markdown(
            """
            # 🎤 ASR Demo: Conformer + SmolLM2

            Record or upload audio to get transcription using the Conformer-SmolLM2 ASR model.

            **Features:**
            - Real-time recording from microphone
            - File upload support (WAV, MP3, FLAC, etc.)
            - Automatic resampling to 16kHz
            - GPU acceleration (if available)
            """
        )

        with gr.Row():
            with gr.Column():
                audio_input = gr.Audio(
                    sources=["microphone", "upload"],
                    type="numpy",
                    label="Audio Input",
                    show_download_button=True
                )

                with gr.Row():
                    clear_btn = gr.Button("Clear", variant="secondary")
                    transcribe_btn = gr.Button("Transcribe", variant="primary")

                gr.Examples(
                    examples=[
                        # Add example audio files if you have them
                        # ["examples/sample1.wav"],
                        # ["examples/sample2.wav"],
                    ],
                    inputs=audio_input,
                    label="Example Audio Files"
                )

            with gr.Column():
                transcription_output = gr.Textbox(
                    label="Transcription",
                    placeholder="Transcription will appear here...",
                    lines=10,
                    max_lines=20
                )

                with gr.Accordion("Model Information", open=False):
                    gr.Markdown(
                        """
                        **Model Architecture:**
                        - Encoder: Conformer (12 layers, 512 hidden dim)
                        - Decoder: SmolLM2-1.7B with LoRA adapters
                        - Audio Projector: Deep projector with cross-attention

                        **Training:**
                        - Dataset: LibriSpeech (clean + other)
                        - Batch size: 128
                        - Learning rate: 1e-4 with cosine schedule

                        **Source:**
                        - Hugging Face: [mazesmazes/asr](https://huggingface.co/mazesmazes/asr)
                        """
                    )

        # Event handlers
        transcribe_btn.click(
            fn=asr_demo.transcribe_audio,
            inputs=[audio_input],
            outputs=[transcription_output]
        )

        clear_btn.click(
            fn=lambda: (None, ""),
            inputs=[],
            outputs=[audio_input, transcription_output]
        )

        # Auto-transcribe on audio change (optional)
        # audio_input.change(
        #     fn=asr_demo.transcribe_audio,
        #     inputs=[audio_input],
        #     outputs=[transcription_output]
        # )

    return demo


def main():
    """Main function to launch the demo."""
    import argparse

    parser = argparse.ArgumentParser(description="Launch ASR Gradio demo")
    parser.add_argument(
        "--model_path",
        type=str,
        default="mazesmazes/asr",
        help="Model path (HF repo or local checkpoint)"
    )
    parser.add_argument(
        "--use_local",
        action="store_true",
        help="Use local checkpoint instead of HF Hub"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create public link for sharing"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run the demo on"
    )

    args = parser.parse_args()

    # Create and launch demo
    demo = create_demo(model_path=args.model_path, use_local=args.use_local)

    print(f"\n🚀 Launching Gradio demo on port {args.port}...")
    print(f"Model: {args.model_path}")

    demo.launch(
        server_port=args.port,
        share=args.share,
        server_name="0.0.0.0" if args.share else "127.0.0.1"
    )


if __name__ == "__main__":
    main()