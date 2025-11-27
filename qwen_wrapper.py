"""
Qwen-Audio model wrapper for audio understanding and captioning.

This module provides a production-ready wrapper for Qwen2-Audio-7B,
implementing the AudioCaptioningModel interface.

Model Details:
    - Repository: Qwen/Qwen2-Audio-7B
    - Architecture: 7B parameter audio-language model with BF16 precision
    - Specialization: General audio understanding, audio captioning, audio Q&A
    - VRAM (BF16): ~14GB | VRAM (4-bit): ~8GB

Key Features:
    - BF16 precision by default (~14GB VRAM on A100)
    - Optional 4-bit quantization for memory efficiency
    - Multi-format audio support (WAV, MP3, FLAC, OGG)
    - Automatic audio preprocessing with librosa

References:
    - HuggingFace: https://huggingface.co/Qwen/Qwen2-Audio-7B
    - GitHub: https://github.com/QwenLM/Qwen2-Audio
"""

import os
import warnings
from typing import Dict, Optional
from pathlib import Path

import torch
import numpy as np
import librosa

from base_model import AudioCaptioningModel


class QwenAudioWrapper(AudioCaptioningModel):
    """Wrapper for Qwen2-Audio-7B model.

    This class handles the complete lifecycle of Qwen2-Audio inference:
    1. Load model from HuggingFace Hub
    2. Process audio files with librosa at correct sampling rate
    3. Generate captions from natural language prompts
    4. Memory-efficient cleanup for sequential evaluations

    Example:
        >>> model = QwenAudioWrapper()
        >>> if model.check_memory_availability():
        ...     model.load_model()
        ...     result = model.generate_caption(
        ...         "audio.wav",
        ...         "Describe the audio content in detail."
        ...     )
        ...     print(result)
        ...     model.unload()

    Attributes:
        model: Qwen2AudioForConditionalGeneration model instance
        processor: AutoProcessor for handling audio and text inputs
        use_4bit: Whether to use 4-bit quantization (default: False)
        sampling_rate: Audio sampling rate required by the model
    """

    def __init__(
        self,
        model_name: str = "Qwen2-Audio-7B",
        device: str = "cuda",
        use_4bit: bool = False,
        max_length: int = 256,
        num_beams: int = 1,
        temperature: float = 1.0
    ):
        """Initialize Qwen2-Audio wrapper.

        Args:
            model_name: Human-readable identifier for logging
            device: Target device ('cuda' or 'cpu')
            use_4bit: Enable 4-bit quantization for memory efficiency
                     (automatically disabled if CUDA not available)
            max_length: Maximum generation length
            num_beams: Number of beams for beam search (1 = greedy decoding)
            temperature: Sampling temperature (1.0 = neutral)
        """
        super().__init__(model_name)
        self.device = device if torch.cuda.is_available() else "cpu"
        self.use_4bit_requested = use_4bit  # Store original request
        self.use_4bit = use_4bit and torch.cuda.is_available()
        self.max_length = max_length
        self.num_beams = num_beams
        self.temperature = temperature
        self.sampling_rate = None  # Will be set after processor loads

        if self.use_4bit:
            print(f"[CONFIG] 4-bit quantization enabled - VRAM: ~8GB")
        else:
            print(f"[CONFIG] Full precision (BF16) - VRAM: ~14GB")

    def _verify_dependencies(self) -> None:
        """Verify required dependencies are installed.

        Qwen2-Audio requires:
        - transformers >= 4.35.0 (with Qwen2AudioForConditionalGeneration)
        - bitsandbytes (for 4-bit quantization)
        - librosa (for audio preprocessing)

        Raises:
            ImportError: If critical dependencies are missing
        """
        missing_packages = []

        try:
            import transformers
            from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
        except ImportError:
            missing_packages.append("transformers>=4.35.0")

        try:
            import bitsandbytes
        except ImportError:
            if self.use_4bit:
                missing_packages.append("bitsandbytes")

        try:
            import librosa
        except ImportError:
            missing_packages.append("librosa")

        if missing_packages:
            raise ImportError(
                f"Missing required packages: {', '.join(missing_packages)}\n"
                f"Install with: pip install {' '.join(missing_packages)}\n"
                f"For latest transformers: pip install git+https://github.com/huggingface/transformers"
            )

    def load_model(self) -> None:
        """Load Qwen2-Audio model with optional 4-bit quantization.

        This method:
        1. Verifies dependencies (transformers, bitsandbytes, librosa)
        2. Checks VRAM availability (~14GB for FP16, ~8GB for 4-bit)
        3. Loads model from HuggingFace Hub
        4. Initializes AutoProcessor for audio preprocessing
        5. Performs warm-up inference to stabilize memory

        Raises:
            RuntimeError: If insufficient VRAM or model loading fails
            ImportError: If required packages are missing
        """
        if self._is_loaded:
            print(f"[OK] {self.model_name} already loaded")
            return

        # Pre-flight VRAM check
        if not self.check_memory_availability():
            raise RuntimeError(
                f"Insufficient VRAM for {self.model_name}. "
                f"Required: {self.get_memory_requirements()['peak_vram_gb']:.1f}GB"
            )

        print(f"[LOADING] Loading {self.model_name} from HuggingFace...")

        try:
            # Step 1: Verify dependencies
            self._verify_dependencies()

            # Step 2: Import Qwen2-Audio components
            try:
                from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
            except ImportError as e:
                raise ImportError(
                    f"Failed to import Qwen2-Audio components: {str(e)}\n"
                    f"Install latest transformers: pip install git+https://github.com/huggingface/transformers"
                ) from e

            # Step 3: Configure 4-bit quantization if requested
            if self.use_4bit:
                from transformers import BitsAndBytesConfig

                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                print("[CONFIG] 4-bit quantization configured (NF4)")
            else:
                quantization_config = None

            # Step 4: Load processor first to get sampling rate
            print("[LOADING] Loading processor...")
            self.processor = AutoProcessor.from_pretrained(
                "Qwen/Qwen2-Audio-7B",
                trust_remote_code=True
            )

            # Extract sampling rate from processor
            self.sampling_rate = self.processor.feature_extractor.sampling_rate
            print(f"[CONFIG] Audio sampling rate: {self.sampling_rate}Hz")

            # Step 5: Load model from HuggingFace
            print("[LOADING] Downloading model weights from HuggingFace...")

            load_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.bfloat16,
                "low_cpu_mem_usage": True
            }

            if self.use_4bit:
                load_kwargs["quantization_config"] = quantization_config
                load_kwargs["device_map"] = "auto"
            elif torch.cuda.is_available():
                load_kwargs["device_map"] = "auto"

            self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
                "Qwen/Qwen2-Audio-7B",
                **load_kwargs
            )

            # Set eval mode
            self.model.eval()

            print(f"[OK] {self.model_name} loaded successfully")

            # Step 6: Warm-up inference
            self._warmup_inference()

            self._is_loaded = True

        except Exception as e:
            # Clean up on failure
            self.unload()
            raise RuntimeError(
                f"Failed to load {self.model_name}: {str(e)}\n"
                f"If using 4-bit quantization, ensure bitsandbytes is installed."
            ) from e

    def _warmup_inference(self) -> None:
        """Perform warm-up inference to stabilize memory and compile kernels.

        Creates a dummy audio file (1 second of silence) and runs inference
        to trigger JIT compilation and stabilize VRAM usage.
        """
        print("[WARMUP] Running warm-up inference...")

        try:
            import tempfile
            import soundfile as sf

            # Create dummy audio
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                dummy_audio = np.zeros(self.sampling_rate, dtype=np.float32)  # 1s silence
                sf.write(tmp.name, dummy_audio, self.sampling_rate)
                dummy_path = tmp.name

            # Run inference
            _ = self.generate_caption(dummy_path, "Warmup query")

            # Clean up
            os.remove(dummy_path)

            if torch.cuda.is_available():
                peak_vram = torch.cuda.max_memory_allocated() / 1e9
                print(f"[OK] Warm-up complete. Peak VRAM: {peak_vram:.2f}GB")

        except Exception as e:
            warnings.warn(f"Warm-up inference failed (non-critical): {str(e)}")

    def generate_caption(self, audio_path: str, prompt: str) -> str:
        """Generate caption for audio file using natural language prompt.

        Qwen2-Audio supports various audio understanding tasks:
        - Audio captioning and description
        - Audio question answering
        - Audio classification
        - Audio event detection

        Args:
            audio_path: Path to audio file (WAV, MP3, FLAC, OGG)
            prompt: Natural language query (e.g., "What sounds are in this audio?")

        Returns:
            Generated caption string

        Raises:
            RuntimeError: If model is not loaded
            FileNotFoundError: If audio_path does not exist
        """
        if not self._is_loaded:
            raise RuntimeError(
                f"{self.model_name} is not loaded. Call load_model() first."
            )

        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        try:
            # Load audio with librosa at correct sampling rate
            audio, sr = librosa.load(audio_path, sr=self.sampling_rate)

            # Prepare inputs using processor
            inputs = self.processor(
                text=prompt,
                audios=audio,
                return_tensors="pt",
                sampling_rate=self.sampling_rate
            )

            # Move inputs to device (if not using device_map="auto")
            if not self.use_4bit and torch.cuda.is_available():
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate caption
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_length=self.max_length,
                    num_beams=self.num_beams,
                    temperature=self.temperature
                )

            # Decode output
            generated_text = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]

            return generated_text

        except Exception as e:
            raise RuntimeError(
                f"Inference failed for {audio_path}: {str(e)}"
            ) from e

    def get_memory_requirements(self) -> Dict[str, float]:
        """Return VRAM requirements for Qwen2-Audio-7B.

        Based on 7B parameter model with BF16 precision:
        - Full precision (BF16): ~14GB VRAM
        - 4-bit quantization: ~8GB VRAM

        Returns:
            {'min_vram_gb': X, 'peak_vram_gb': Y}
        """
        if self.use_4bit:
            return {
                'min_vram_gb': 7.0,
                'peak_vram_gb': 8.0
            }
        else:
            return {
                'min_vram_gb': 12.0,
                'peak_vram_gb': 14.0
            }
