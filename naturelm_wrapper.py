"""Fixed NatureLM wrapper for Lambda H100."""

import os
import sys
from typing import Dict

import torch
import librosa

from base_model import AudioCaptioningModel


class NatureLMWrapper(AudioCaptioningModel):
    """Wrapper for NatureLM-audio model."""

    def __init__(
        self,
        model_name: str = "NatureLM-audio",
        device: str = "cuda",
        hf_token: str = None,
    ):
        super().__init__(model_name)
        self.device = device if torch.cuda.is_available() else "cpu"
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        self.model = None
        self.pipeline = None

    def load_model(self) -> None:
        if self._is_loaded:
            print(f"[OK] {self.model_name} already loaded")
            return

        print(f"[LOADING] Loading {self.model_name}...")

        # Login to HuggingFace
        if self.hf_token:
            from huggingface_hub import login
            login(token=self.hf_token, add_to_git_credential=False)

        # Import NatureLM
        from NatureLM.models import NatureLM
        from NatureLM.infer import Pipeline
        from huggingface_hub import hf_hub_download

        # Load model
        self.model = NatureLM.from_pretrained(
            "EarthSpeciesProject/NatureLM-audio",
            device=self.device
        )
        self.model = self.model.to(self.device).to(torch.bfloat16)
        self.model = self.model.eval()

        # Load pipeline config
        cfg_path = hf_hub_download("EarthSpeciesProject/NatureLM-audio", "inference.yml")
        self.pipeline = Pipeline(model=self.model, cfg_path=cfg_path)

        self._is_loaded = True
        print(f"[OK] {self.model_name} loaded successfully")

    def generate_caption(self, audio_path: str, prompt: str) -> str:
        if not self._is_loaded:
            raise RuntimeError(f"{self.model_name} not loaded")

        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio not found: {audio_path}")

        # Load audio
        audio, sr = librosa.load(audio_path, sr=16000)

        # Run inference
        result = self.pipeline(
            audios=[audio],
            queries=prompt,
            input_sample_rate=sr
        )

        # Extract text from result
        if isinstance(result, list):
            result = result[0]

        # Clean up timestamp prefixes if present
        if isinstance(result, str) and "#" in result:
            # Remove timestamp like "#0.00s - 10.00s#: "
            parts = result.split("#: ", 1)
            if len(parts) > 1:
                result = parts[1]

        return result.strip()

    def get_memory_requirements(self) -> Dict[str, float]:
        return {"min_vram_gb": 8.0, "peak_vram_gb": 10.0}

    def unload(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
        self._is_loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[OK] {self.model_name} unloaded")
