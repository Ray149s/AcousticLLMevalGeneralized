"""
Abstract base class for audio captioning models.

This module defines the interface that all audio captioning model wrappers
must implement to ensure consistent behavior across different model architectures
(NatureLM, SALMONN, Qwen-Audio, etc.).
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional
import torch
import gc


class AudioCaptioningModel(ABC):
    """Abstract interface for all audio captioning models.

    This base class enforces a consistent API for loading models, generating
    captions, and managing GPU memory - critical for sequential evaluation
    workflows in resource-constrained environments like Google Colab.

    Attributes:
        model_name (str): Human-readable identifier for the model
        model: The loaded model instance (implementation-specific)
        processor: The loaded processor/tokenizer (implementation-specific)
    """

    def __init__(self, model_name: str):
        """Initialize the model wrapper.

        Args:
            model_name: Human-readable identifier for logging and reporting
        """
        self.model_name = model_name
        self.model = None
        self.processor = None
        self._is_loaded = False

    @abstractmethod
    def load_model(self) -> None:
        """Initialize the model and load weights onto the GPU.

        Implementation requirements:
        - Download/load model weights from HuggingFace Hub or local cache
        - Move model to CUDA device with appropriate dtype (bfloat16 recommended)
        - Load any required processors, tokenizers, or feature extractors
        - Perform warm-up inference to JIT-compile kernels and stabilize memory
        - Set self._is_loaded = True on success

        Raises:
            RuntimeError: If VRAM is insufficient (checked via check_memory_availability)
            ValueError: If model files are corrupted or inaccessible
        """
        pass

    @abstractmethod
    def generate_caption(self, audio_path: str, prompt: str) -> str:
        """Process audio and text prompt to generate a caption.

        Args:
            audio_path: Absolute path to audio file (WAV, MP3, OGG, FLAC supported)
            prompt: Natural language query or instruction for the model

        Returns:
            Generated caption as a string. May include timestamp annotations
            (e.g., "#0.00s - 10.00s#: Green Treefrog") depending on model behavior.

        Raises:
            FileNotFoundError: If audio_path does not exist
            RuntimeError: If model is not loaded (call load_model() first)
            Exception: Model-specific inference errors (e.g., audio format issues)
        """
        pass

    @abstractmethod
    def get_memory_requirements(self) -> Dict[str, float]:
        """Return VRAM requirements for this model.

        Returns:
            Dictionary with keys:
                - 'min_vram_gb': Minimum VRAM needed to load model (float)
                - 'peak_vram_gb': Peak VRAM during inference (float)

        Example:
            {'min_vram_gb': 8.5, 'peak_vram_gb': 10.2}
        """
        pass

    def check_memory_availability(self) -> bool:
        """Verify VRAM before load() - prevent mid-evaluation OOM crashes.

        Returns:
            True if sufficient VRAM is available with 10% safety margin,
            False otherwise.

        Example:
            >>> model = NatureLMWrapper()
            >>> if model.check_memory_availability():
            ...     model.load_model()
            ... else:
            ...     print("Insufficient VRAM!")
        """
        if not torch.cuda.is_available():
            print("[WARNING] CUDA not available - CPU inference will be very slow")
            return True  # Allow CPU fallback

        available_bytes, total_bytes = torch.cuda.mem_get_info()
        available_gb = available_bytes / 1e9
        required_gb = self.get_memory_requirements()['peak_vram_gb']
        required_with_margin = required_gb * 1.1  # 10% safety margin

        print(f"[VRAM CHECK] {available_gb:.2f}GB available, "
              f"{required_with_margin:.2f}GB required (with 10% margin)")

        return available_gb >= required_with_margin

    def unload(self) -> None:
        """Aggressively frees VRAM. Mandatory for Colab execution loops.

        This method ensures complete memory cleanup between sequential model
        evaluations. Without this, VRAM fragmentation causes OOM errors when
        switching between models (e.g., NatureLM -> SALMONN).

        Deletes:
            - self.model
            - self.processor
            - self.tokenizer (if exists)
            - Any other model-specific attributes

        Then triggers Python GC and CUDA cache clearing.
        """
        print(f"[UNLOAD] Unloading {self.model_name}...")

        # Delete model components
        if hasattr(self, 'model') and self.model is not None:
            del self.model
            self.model = None

        if hasattr(self, 'processor') and self.processor is not None:
            del self.processor
            self.processor = None

        if hasattr(self, 'tokenizer') and self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        # Mark as unloaded
        self._is_loaded = False

        # Aggressive memory cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            freed_bytes = torch.cuda.mem_get_info()[0]
            print(f"[OK] VRAM cleared: {freed_bytes / 1e9:.2f}GB now available")
        else:
            print("[OK] Memory cleared (CPU mode)")

    def is_loaded(self) -> bool:
        """Check if model is currently loaded in memory.

        Returns:
            True if model is loaded and ready for inference, False otherwise.
        """
        return self._is_loaded

    def __repr__(self) -> str:
        """String representation for debugging."""
        status = "loaded" if self._is_loaded else "unloaded"
        return f"{self.__class__.__name__}(model_name='{self.model_name}', status='{status}')"
