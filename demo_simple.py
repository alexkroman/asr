#!/usr/bin/env python3
"""
Simple Gradio demo using Hugging Face Inference API.
"""

import gradio as gr
import numpy as np
from huggingface_hub import InferenceClient

# Initialize the inference client
client = InferenceClient(model="mazesmazes/asr")


def transcribe_audio(audio_input):
    """Transcribe audio using HF Inference API."""
    if audio_input is None:
        return "Please record or upload an audio file."

    try:
        # Handle Gradio audio input (tuple: sample_rate, numpy array)
        if isinstance(audio_input, tuple):
            sample_rate, audio_data = audio_input

            # The Inference API expects audio bytes
            # Convert numpy array to bytes
            import io
            import scipy.io.wavfile as wavfile

            # Create WAV file in memory
            buffer = io.BytesIO()
            wavfile.write(buffer, sample_rate, audio_data)
            audio_bytes = buffer.getvalue()

        elif isinstance(audio_input, str):
            # If it's a file path, read the file
            with open(audio_input, 'rb') as f:
                audio_bytes = f.read()
        else:
            audio_bytes = audio_input

        # Call the inference API
        result = client.automatic_speech_recognition(audio_bytes)

        # Extract transcription from result
        if isinstance(result, dict):
            return result.get('text', str(result))
        else:
            return str(result)

    except Exception as e:
        return f"Error: {str(e)}"


# Create Gradio interface
with gr.Blocks(title="ASR Demo - Simple") as demo:
    gr.Markdown(
        """
        # 🎤 Simple ASR Demo using Hugging Face Inference API

        Record or upload audio to transcribe using the model hosted on Hugging Face.
        """
    )

    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(
                sources=["microphone", "upload"],
                type="numpy",
                label="Audio Input"
            )
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


if __name__ == "__main__":
    demo.launch()