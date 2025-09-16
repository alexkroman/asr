#!/usr/bin/env python3
"""
Simple Gradio demo using Hugging Face transformers pipeline.
"""

import gradio as gr
import torch
from transformers import pipeline

# Determine device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")

# Initialize the pipeline directly from Hugging Face
# This will download and cache the model automatically
pipe = pipeline(
    "automatic-speech-recognition",
    model="mazesmazes/asr",
    device=device if device != "mps" else -1,  # MPS uses -1 in pipeline
    trust_remote_code=True  # Allow custom code from the repo
)


def transcribe_audio(audio_input):
    """Transcribe audio using HF pipeline."""
    if audio_input is None:
        return "Please record or upload an audio file."

    try:
        # The pipeline handles various input formats automatically
        if isinstance(audio_input, tuple):
            sample_rate, audio_data = audio_input
            result = pipe({"array": audio_data, "sampling_rate": sample_rate})
        else:
            result = pipe(audio_input)

        # Extract transcription
        if isinstance(result, dict):
            return result.get('text', str(result))
        else:
            return str(result)

    except Exception as e:
        return f"Error: {str(e)}"


# Create Gradio interface
with gr.Blocks(title="ASR Demo") as demo:
    gr.Markdown(
        """
        # 🎤 ASR Demo using Transformers Pipeline

        Record or upload audio to transcribe using the model from Hugging Face.

        Model: [mazesmazes/asr](https://huggingface.co/mazesmazes/asr)
        """
    )

    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(
                sources=["microphone", "upload"],
                type="numpy",
                label="Audio Input"
            )

            with gr.Row():
                clear_btn = gr.Button("Clear", variant="secondary")
                transcribe_btn = gr.Button("Transcribe", variant="primary")

        with gr.Column():
            output = gr.Textbox(
                label="Transcription",
                placeholder="Transcription will appear here...",
                lines=5
            )

    transcribe_btn.click(
        fn=transcribe_audio,
        inputs=[audio_input],
        outputs=[output]
    )

    clear_btn.click(
        fn=lambda: (None, ""),
        inputs=[],
        outputs=[audio_input, output]
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="Create public link")
    parser.add_argument("--port", type=int, default=7860, help="Port to run on")
    args = parser.parse_args()

    print(f"\n🚀 Launching demo on port {args.port}...")
    print(f"Model: mazesmazes/asr")
    print(f"Device: {device}")

    demo.launch(
        server_port=args.port,
        share=args.share,
        server_name="0.0.0.0" if args.share else "127.0.0.1"
    )