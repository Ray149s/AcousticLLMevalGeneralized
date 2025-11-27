"""
NatureLM-audio model wrapper for bioacoustic captioning.

This module provides a production-ready wrapper for EarthSpeciesProject's
NatureLM-audio model, implementing the AudioCaptioningModel interface.

Model Details:
    - Repository: EarthSpeciesProject/NatureLM-audio
    - Architecture: Llama-3.1-8B-Instruct fine-tuned on audio-text pairs
    - Specialization: Bioacoustics (species classification, captioning)
    - VRAM: ~10GB peak with bfloat16 precision

References:
    - HuggingFace: https://huggingface.co/EarthSpeciesProject/NatureLM-audio
    - GitHub: https://github.com/earthspecies/NatureLM-audio
    - Paper: Robinson et al. (2025), ICLR 2025
"""

import os
import warnings
from typing import Dict, List, Optional
import torch

from base_model import AudioCaptioningModel


class NatureLMWrapper(AudioCaptioningModel):
    """Wrapper for NatureLM-audio model.

    This class handles the complete lifecycle of NatureLM inference:
    1. Loading model from HuggingFace Hub
    2. Processing audio files with sliding window approach
    3. Generating captions from natural language prompts
    4. Memory-efficient cleanup for sequential evaluations

    Example:
        >>> model = NatureLMWrapper()
        >>> if model.check_memory_availability():
        ...     model.load_model()
        ...     result = model.generate_caption(
        ...         "audio.mp3",
        ...         "What species is vocalizing in this audio?"
        ...     )
        ...     print(result)
        ...     model.unload()

    Attributes:
        model: NatureLM model instance (loaded via NatureLM.from_pretrained)
        pipeline: Inference pipeline for processing audio
        window_length_seconds: Audio chunk size for sliding window (default: 10.0)
        hop_length_seconds: Overlap between windows (default: 10.0)
    """

    def __init__(
        self,
        model_name: str = "NatureLM-audio",
        window_length_seconds: float = 10.0,
        hop_length_seconds: float = 10.0,
        device: str = "cuda"
    ):
        """Initialize NatureLM wrapper.

        Args:
            model_name: Human-readable identifier for logging
            window_length_seconds: Audio window size for inference (seconds)
            hop_length_seconds: Stride between consecutive windows (seconds)
            device: Target device ('cuda' or 'cpu')
        """
        super().__init__(model_name)
        self.window_length_seconds = window_length_seconds
        self.hop_length_seconds = hop_length_seconds
        self.device = device if torch.cuda.is_available() else "cpu"
        self.pipeline = None

        # Validate Llama access
        self._check_llama_access()

    def _check_llama_access(self) -> None:
        """Verify user has access to Llama-3.1-8B-Instruct.

        NatureLM is based on Llama-3.1, which requires explicit access approval
        from Meta via HuggingFace. This check warns users if they haven't
        requested access yet.

        Raises:
            Warning if HF_TOKEN is not set in environment
        """
        if "HF_TOKEN" not in os.environ and "HUGGING_FACE_HUB_TOKEN" not in os.environ:
            warnings.warn(
                "[WARNING] No HuggingFace token detected in environment.\n"
                "NatureLM requires access to Llama-3.1-8B-Instruct.\n"
                "Please:\n"
                "  1. Request access: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct\n"
                "  2. Set HF_TOKEN environment variable with your HuggingFace token\n"
                "  3. Run: huggingface-cli login"
            )

    def load_model(self) -> None:
        """Load NatureLM model and initialize inference pipeline.

        This method:
        1. Checks VRAM availability (10GB required)
        2. Downloads model from HuggingFace (if not cached)
        3. Loads model in bfloat16 precision for memory efficiency
        4. Initializes Pipeline for audio processing
        5. Performs warm-up inference to stabilize memory

        Raises:
            RuntimeError: If insufficient VRAM or model loading fails
            ImportError: If NatureLM package is not installed
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
            # Import NatureLM components
            try:
                from NatureLM.models import NatureLM
                from NatureLM.infer import Pipeline
            except ImportError as e:
                raise ImportError(
                    "NatureLM package not found. Install with:\n"
                    "  git clone https://github.com/earthspecies/NatureLM-audio\n"
                    "  cd NatureLM-audio\n"
                    "  pip install -e .[gpu]  # or pip install -e . for CPU"
                ) from e

            # Load model
            self.model = NatureLM.from_pretrained("EarthSpeciesProject/NatureLM-audio")

            # Use bfloat16 for memory efficiency (Llama-3.1 supports this natively)
            self.model = self.model.to(dtype=torch.bfloat16, device=self.device)
            self.model.eval()  # Set to evaluation mode

            # Initialize inference pipeline
            self.pipeline = Pipeline(model=self.model)

            print(f"[OK] {self.model_name} loaded successfully")

            # Warm-up inference to JIT-compile kernels and stabilize VRAM
            self._warmup_inference()

            self._is_loaded = True

        except Exception as e:
            # Clean up on failure
            self.unload()
            raise RuntimeError(f"Failed to load {self.model_name}: {str(e)}") from e

    def _warmup_inference(self) -> None:
        """Perform warm-up inference to stabilize memory and compile kernels.

        This prevents first-inference slowdowns and ensures accurate peak
        VRAM measurements. Uses a dummy audio path with a simple query.
        """
        print("[WARMUP] Running warm-up inference...")
        try:
            # Create a minimal dummy audio (1 second of silence)
            import numpy as np
            import soundfile as sf
            import tempfile

            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                dummy_audio = np.zeros(16000, dtype=np.float32)  # 1s at 16kHz
                sf.write(tmp.name, dummy_audio, 16000)
                dummy_path = tmp.name

            # Run inference
            _ = self.pipeline(
                [dummy_path],
                ["Warmup query"],
                window_length_seconds=1.0,
                hop_length_seconds=1.0
            )

            # Clean up
            os.remove(dummy_path)

            if torch.cuda.is_available():
                peak_vram = torch.cuda.max_memory_allocated() / 1e9
                print(f"[OK] Warm-up complete. Peak VRAM: {peak_vram:.2f}GB")

        except Exception as e:
            warnings.warn(f"Warm-up inference failed (non-critical): {str(e)}")

    def generate_caption(self, audio_path: str, prompt: str) -> str:
        """Generate caption for audio file using natural language prompt.

        Args:
            audio_path: Path to audio file (WAV, MP3, OGG, FLAC)
            prompt: Natural language query (e.g., "What species is this?")

        Returns:
            Generated caption string. Format: "#start_s - end_s#: caption"
            Example: "#0.00s - 10.00s#: Green Treefrog\n"

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

        # Run inference through pipeline
        results = self.pipeline(
            [audio_path],
            [prompt],
            window_length_seconds=self.window_length_seconds,
            hop_length_seconds=self.hop_length_seconds
        )

        # Pipeline returns list of results (one per audio file)
        if results and len(results) > 0:
            return results[0]
        else:
            return ""

    def generate_captions_batch(
        self,
        audio_paths: List[str],
        prompts: List[str]
    ) -> List[str]:
        """Generate captions for multiple audio files in batch.

        This is more efficient than sequential calls when processing
        multiple files with the same model.

        Args:
            audio_paths: List of audio file paths
            prompts: List of prompts (one per audio file)

        Returns:
            List of generated captions (same order as inputs)

        Raises:
            ValueError: If len(audio_paths) != len(prompts)
        """
        if len(audio_paths) != len(prompts):
            raise ValueError(
                f"Mismatch: {len(audio_paths)} audio files but {len(prompts)} prompts"
            )

        if not self._is_loaded:
            raise RuntimeError(f"{self.model_name} is not loaded")

        # Validate all paths exist
        for path in audio_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Audio file not found: {path}")

        # Batch inference
        results = self.pipeline(
            audio_paths,
            prompts,
            window_length_seconds=self.window_length_seconds,
            hop_length_seconds=self.hop_length_seconds
        )

        return results

    def get_memory_requirements(self) -> Dict[str, float]:
        """Return VRAM requirements for NatureLM-audio.

        Based on empirical measurements with Llama-3.1-8B in bfloat16:
        - Min VRAM: ~8.5GB (model weights alone)
        - Peak VRAM: ~10.0GB (with inference buffers and audio processing)

        Returns:
            {'min_vram_gb': 8.5, 'peak_vram_gb': 10.0}
        """
        return {
            'min_vram_gb': 8.5,
            'peak_vram_gb': 10.0
        }

    def unload(self) -> None:
        """Unload model and free VRAM.

        Extends base class unload() to also clear the inference pipeline.
        """
        if hasattr(self, 'pipeline') and self.pipeline is not None:
            del self.pipeline
            self.pipeline = None

        # Call parent unload for model/processor cleanup
        super().unload()
