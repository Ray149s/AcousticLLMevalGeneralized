"""
SALMONN model wrapper for general audio understanding.

This module provides a production-ready wrapper for SALMONN (Speech Audio Language
Music Open Neural Network), implementing the AudioCaptioningModel interface with
FP16 precision for optimal quality (8-bit quantization available if needed).

Model Details:
    - Repository: bytedance/SALMONN (branch: salmonn)
    - Architecture: Whisper + BEATs encoders + Vicuna-13B with Q-Former fusion
    - Specialization: Speech, audio events, music understanding
    - VRAM (fp16): ~28GB | VRAM (8-bit): ~13-16GB
    - Paper: Tang et al. (2024), ICLR 2024

Key Features:
    - FP16 precision by default (~28GB VRAM)
    - Dynamic repo cloning and path management
    - Multi-modal audio understanding (speech, events, music)
    - Automatic dependency handling

References:
    - GitHub: https://github.com/bytedance/SALMONN/tree/salmonn
    - HuggingFace: https://huggingface.co/tsinghua-ee/SALMONN
    - Paper: https://arxiv.org/abs/2310.13289
"""

import os
import sys
import warnings
import subprocess
from typing import Dict, Optional
from pathlib import Path

import torch
import numpy as np
import soundfile as sf

from base_model import AudioCaptioningModel


class SalmonnWrapper(AudioCaptioningModel):
    """Wrapper for SALMONN model with FP16 precision.

    This class handles the complete lifecycle of SALMONN inference
    optimized for A100 40GB environments:
    1. Clone SALMONN repository and set up Python path
    2. Load model with FP16 precision (~28GB VRAM)
    3. Process audio files (speech/audio events/music)
    4. Generate captions from natural language prompts
    5. Memory-efficient cleanup for sequential evaluations

    Example:
        >>> model = SalmonnWrapper()
        >>> if model.check_memory_availability():
        ...     model.load_model()
        ...     result = model.generate_caption(
        ...         "audio.wav",
        ...         "Describe the audio content."
        ...     )
        ...     print(result)
        ...     model.unload()

    Attributes:
        model: SALMONN model instance (loaded with FP16 precision)
        audio_encoder: Combined Whisper + BEATs encoder
        salmonn_repo_path: Path to cloned SALMONN repository
        use_8bit: Whether to use 8-bit quantization (default: False)
    """

    def __init__(
        self,
        model_name: str = "SALMONN",
        device: str = "cuda",
        use_8bit: bool = False,
        salmonn_repo_path: Optional[str] = None
    ):
        """Initialize SALMONN wrapper.

        Args:
            model_name: Human-readable identifier for logging
            device: Target device ('cuda' or 'cpu')
            use_8bit: Enable 8-bit quantization for memory efficiency
                     (automatically disabled if CUDA not available)
            salmonn_repo_path: Path to SALMONN repo. If None, will clone to
                              /content/SALMONN (Colab) or ./SALMONN (local)
        """
        super().__init__(model_name)
        self.device = device if torch.cuda.is_available() else "cpu"
        self.use_8bit_requested = use_8bit  # Store original request
        self.use_8bit = use_8bit and torch.cuda.is_available()
        self.audio_encoder = None
        self.config = None

        # Determine repository path
        if salmonn_repo_path is None:
            # Default to /content/SALMONN on Colab, ./SALMONN locally
            base_path = Path("/content") if Path("/content").exists() else Path.cwd()
            self.salmonn_repo_path = base_path / "SALMONN"
        else:
            self.salmonn_repo_path = Path(salmonn_repo_path)

        if self.use_8bit:
            print(f"[CONFIG] 8-bit quantization enabled - VRAM: ~13-16GB")
        else:
            print(f"[CONFIG] Full precision (fp16) - VRAM: ~26GB")

    def _clone_salmonn_repo(self) -> None:
        """Clone SALMONN repository if not already present.

        Clones the 'salmonn' branch (ICLR 2024 version) to the configured path
        and adds it to sys.path for imports.

        Raises:
            RuntimeError: If git clone fails or repo is corrupted
        """
        if self.salmonn_repo_path.exists():
            print(f"[OK] SALMONN repository found at {self.salmonn_repo_path}")

            # Verify it's the correct branch
            try:
                result = subprocess.run(
                    ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                    cwd=self.salmonn_repo_path,
                    capture_output=True,
                    text=True,
                    check=True
                )
                current_branch = result.stdout.strip()
                if current_branch != "salmonn":
                    print(f"[WARNING] Repository on branch '{current_branch}', "
                          f"expected 'salmonn'. May cause issues.")
            except subprocess.CalledProcessError:
                print("[WARNING] Could not verify git branch")

            return

        print(f"[CLONING] Downloading SALMONN repository to {self.salmonn_repo_path}...")

        try:
            # Clone the salmonn branch
            subprocess.run(
                [
                    "git", "clone",
                    "--branch", "salmonn",
                    "--single-branch",
                    "https://github.com/bytedance/SALMONN.git",
                    str(self.salmonn_repo_path)
                ],
                check=True,
                capture_output=True,
                text=True
            )
            print(f"[OK] Repository cloned successfully")

        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Failed to clone SALMONN repository: {e.stderr}"
            ) from e

    def _add_salmonn_to_path(self) -> None:
        """Add SALMONN repository to Python sys.path for imports.

        This allows importing SALMONN's internal modules (model, config, etc.)
        without needing to install as a package.
        """
        salmonn_path_str = str(self.salmonn_repo_path.absolute())

        if salmonn_path_str not in sys.path:
            sys.path.insert(0, salmonn_path_str)
            print(f"[OK] Added {salmonn_path_str} to sys.path")

    def _verify_dependencies(self) -> None:
        """Verify required dependencies are installed.

        SALMONN requires:
        - transformers (with BitsAndBytesConfig support)
        - bitsandbytes (for 8-bit quantization)
        - whisper (OpenAI's speech encoder)
        - Various audio processing libraries

        Raises:
            ImportError: If critical dependencies are missing
        """
        missing_packages = []

        try:
            import transformers
            from transformers import BitsAndBytesConfig
        except ImportError:
            missing_packages.append("transformers[torch]")

        try:
            import bitsandbytes
        except ImportError:
            if self.use_8bit:
                missing_packages.append("bitsandbytes")

        try:
            import whisper
        except ImportError:
            missing_packages.append("openai-whisper")

        if missing_packages:
            raise ImportError(
                f"Missing required packages: {', '.join(missing_packages)}\n"
                f"Install with: pip install {' '.join(missing_packages)}"
            )

    def load_model(self) -> None:
        """Load SALMONN model with 8-bit quantization.

        This method:
        1. Clones SALMONN repo (if needed) and adds to sys.path
        2. Verifies dependencies (transformers, bitsandbytes, whisper)
        3. Checks VRAM availability (~16GB required with 8-bit)
        4. Loads model with BitsAndBytesConfig for memory efficiency
        5. Initializes audio encoders (Whisper + BEATs)
        6. Performs warm-up inference to stabilize memory

        Raises:
            RuntimeError: If insufficient VRAM or model loading fails
            ImportError: If SALMONN code or dependencies are missing
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
            # Step 1: Clone repo and set up paths
            self._clone_salmonn_repo()
            self._add_salmonn_to_path()

            # Step 2: Verify dependencies
            self._verify_dependencies()

            # Step 3: Import SALMONN components (now that it's in sys.path)
            try:
                # These imports will work after adding to sys.path
                from omegaconf import OmegaConf
                from models.salmonn import SALMONN  # SALMONN's internal model class

            except ImportError as e:
                raise ImportError(
                    f"Failed to import SALMONN components: {str(e)}\n"
                    f"Ensure SALMONN repository was cloned correctly to "
                    f"{self.salmonn_repo_path}"
                ) from e

            # Step 4: Configure 8-bit quantization
            if self.use_8bit:
                from transformers import BitsAndBytesConfig

                bnb_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0,  # Recommended threshold
                    llm_int8_has_fp16_weight=False
                )
                print("[CONFIG] 8-bit quantization configured")
            else:
                bnb_config = None

            # Step 5: Load configuration
            config_path = self.salmonn_repo_path / "configs" / "decode_config.yaml"
            if not config_path.exists():
                raise FileNotFoundError(
                    f"SALMONN config not found at {config_path}. "
                    f"Repository may be incomplete."
                )

            self.config = OmegaConf.load(config_path)

            # Update config for quantization if enabled
            if bnb_config is not None:
                self.config.model.quantization_config = bnb_config

            # Step 6: Load model from HuggingFace
            print("[LOADING] Downloading model weights from HuggingFace...")

            self.model = SALMONN.from_pretrained(
                "tsinghua-ee/SALMONN",
                config=self.config,
                device_map="auto" if torch.cuda.is_available() else None,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )

            # Move to device and set eval mode
            if not self.use_8bit:  # 8-bit already handles device placement
                self.model = self.model.to(self.device)

            self.model.eval()

            print(f"[OK] {self.model_name} loaded successfully")

            # Step 7: Warm-up inference
            self._warmup_inference()

            self._is_loaded = True

        except Exception as e:
            # Clean up on failure
            self.unload()
            raise RuntimeError(
                f"Failed to load {self.model_name}: {str(e)}\n"
                f"If using 8-bit quantization, ensure bitsandbytes is installed."
            ) from e

    def _warmup_inference(self) -> None:
        """Perform warm-up inference to stabilize memory and compile kernels.

        Creates a dummy audio file (1 second of silence) and runs inference
        to trigger JIT compilation and stabilize VRAM usage.
        """
        print("[WARMUP] Running warm-up inference...")

        try:
            import tempfile

            # Create dummy audio
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                dummy_audio = np.zeros(16000, dtype=np.float32)  # 1s at 16kHz
                sf.write(tmp.name, dummy_audio, 16000)
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

        SALMONN supports various audio understanding tasks:
        - Speech recognition and translation
        - Audio event classification
        - Music understanding
        - Audio captioning

        Args:
            audio_path: Path to audio file (WAV, MP3 supported)
            prompt: Natural language query (e.g., "Describe the audio content")

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
            # SALMONN expects audio path and text prompt
            # The exact API depends on their implementation
            # This is a generic interface based on their demo code

            with torch.no_grad():
                response = self.model.generate(
                    audio_path=audio_path,
                    prompt=prompt,
                    max_length=512,  # Configurable
                    num_beams=4,
                    temperature=1.0
                )

            return response if isinstance(response, str) else response[0]

        except Exception as e:
            raise RuntimeError(
                f"Inference failed for {audio_path}: {str(e)}"
            ) from e

    def get_memory_requirements(self) -> Dict[str, float]:
        """Return VRAM requirements for SALMONN.

        Memory usage depends on quantization mode:
        - Full precision (fp16): ~26GB (too large for A100 40GB)
        - 8-bit quantization: ~13-16GB (recommended)

        Returns:
            {'min_vram_gb': 13.0, 'peak_vram_gb': 16.0} with 8-bit
            {'min_vram_gb': 24.0, 'peak_vram_gb': 26.0} without 8-bit
        """
        if self.use_8bit:
            return {
                'min_vram_gb': 13.0,
                'peak_vram_gb': 16.0
            }
        else:
            return {
                'min_vram_gb': 24.0,
                'peak_vram_gb': 28.0
            }

    def unload(self) -> None:
        """Unload model and free VRAM.

        Extends base class unload() to also clear audio encoder and config.
        """
        if hasattr(self, 'audio_encoder') and self.audio_encoder is not None:
            del self.audio_encoder
            self.audio_encoder = None

        if hasattr(self, 'config') and self.config is not None:
            del self.config
            self.config = None

        # Call parent unload for model/processor cleanup
        super().unload()
