"""Custom handler for Hugging Face Inference API."""

import torch
import torchaudio
import numpy as np
from typing import Dict, List, Any
from transformers import AutoTokenizer


class EndpointHandler:
    def __init__(self, path=""):
        """Initialize the handler with model and tokenizer."""
        # Load your custom model
        from train import ASRModel, ModelArguments

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ASRModel.from_pretrained(path).to(self.device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(path)

        # Audio processing parameters
        self.sample_rate = 16000

    def preprocess_audio(self, audio_data, sampling_rate):
        """Preprocess audio to the required format."""
        # Convert to tensor if needed
        if isinstance(audio_data, (list, np.ndarray)):
            audio = torch.tensor(audio_data, dtype=torch.float32)
        else:
            audio = audio_data

        # Resample if necessary
        if sampling_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sampling_rate,
                new_freq=self.sample_rate
            )
            audio = resampler(audio)

        # Ensure correct shape
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        return audio

    def __call__(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process the input data and return transcription.

        Args:
            data: Dictionary containing:
                - inputs: Audio data (numpy array or list)
                - sampling_rate: Audio sampling rate

        Returns:
            List containing dictionary with transcription
        """
        # Get inputs
        inputs = data.get("inputs", data)
        sampling_rate = data.get("sampling_rate", 16000)

        # Handle different input formats
        if isinstance(inputs, dict):
            audio_data = inputs.get("array", inputs.get("values"))
            sampling_rate = inputs.get("sampling_rate", sampling_rate)
        else:
            audio_data = inputs

        # Preprocess audio
        audio = self.preprocess_audio(audio_data, sampling_rate)
        audio = audio.to(self.device)

        # Run inference
        with torch.no_grad():
            transcription = self.model.transcribe(audio)

        return [{"transcription": transcription}]