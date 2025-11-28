"""Fixed Qwen2-Audio wrapper using correct conversation API."""

import os
import warnings
from typing import Dict
import torch
import librosa

from base_model import AudioCaptioningModel


class QwenAudioWrapper(AudioCaptioningModel):
    """Wrapper for Qwen2-Audio-7B-Instruct model."""

    def __init__(
        self,
        model_name: str = "Qwen2-Audio-7B",
        device: str = "cuda",
        use_4bit: bool = False,
        max_new_tokens: int = 256,
    ):
        super().__init__(model_name)
        self.device = device if torch.cuda.is_available() else "cpu"
        self.use_4bit = use_4bit and torch.cuda.is_available()
        self.max_new_tokens = max_new_tokens
        self.sampling_rate = None
        self.model = None
        self.processor = None

    def load_model(self) -> None:
        if self._is_loaded:
            print(f"[OK] {self.model_name} already loaded")
            return

        print(f"[LOADING] Loading {self.model_name}...")

        from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2-Audio-7B-Instruct",
            trust_remote_code=True
        )
        self.sampling_rate = self.processor.feature_extractor.sampling_rate

        # Load model
        load_kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": "auto",
            "trust_remote_code": True,
        }

        if self.use_4bit:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-Audio-7B-Instruct",
            **load_kwargs
        )
        self.model.eval()
        self._is_loaded = True
        print(f"[OK] {self.model_name} loaded successfully")

    def generate_caption(self, audio_path: str, prompt: str) -> str:
        if not self._is_loaded:
            raise RuntimeError(f"{self.model_name} not loaded. Call load_model() first.")

        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Load audio
        audio, sr = librosa.load(audio_path, sr=self.sampling_rate)

        # Create conversation format
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        # Process with chat template
        text = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False
        )
        inputs = self.processor(
            text=text,
            audios=[audio],
            return_tensors="pt",
            sampling_rate=self.sampling_rate
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens
            )

        # Decode - skip input tokens
        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        response = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]

        return response

    def get_memory_requirements(self) -> Dict[str, float]:
        if self.use_4bit:
            return {"min_vram_gb": 7.0, "peak_vram_gb": 8.0}
        return {"min_vram_gb": 12.0, "peak_vram_gb": 14.0}

    def unload(self) -> None:
        if hasattr(self, "model") and self.model is not None:
            del self.model
            self.model = None
        if hasattr(self, "processor") and self.processor is not None:
            del self.processor
            self.processor = None
        self._is_loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[OK] {self.model_name} unloaded")
