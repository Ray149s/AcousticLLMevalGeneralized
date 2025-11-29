"""
SALMONN model wrapper - FIXED VERSION.

This module provides a corrected wrapper for SALMONN that matches the official
inference code from bytedance/SALMONN repository (salmonn branch).

CRITICAL: SALMONN requires specific older library versions:
    - transformers==4.28.0
    - peft==0.3.0
    - torch==2.0.1
    - torchaudio==2.0.2
    - accelerate==0.20.3
    - bitsandbytes==0.35.0

The garbled output issue is caused by:
1. Version mismatch (newer transformers/peft break LoRA loading)
2. Incorrect API usage (from_pretrained vs from_config)
3. Missing audio preprocessing (prepare_one_sample)
4. Wrong prompt formatting (must use "<Speech><SpeechHere></Speech>")

Model Details:
    - Repository: bytedance/SALMONN (branch: salmonn)
    - HuggingFace: tsinghua-ee/SALMONN
    - Architecture: Whisper + BEATs + Q-Former + Vicuna-13B + LoRA
    - VRAM: ~28GB (fp16) or ~16GB (8-bit)
"""

import os
import sys
import warnings
import subprocess
from typing import Dict, Optional, Any
from pathlib import Path

import torch
import numpy as np
import soundfile as sf

from base_model import AudioCaptioningModel


class SalmonnWrapperFixed(AudioCaptioningModel):
    """Fixed wrapper for SALMONN model matching official inference code.

    This wrapper correctly implements SALMONN inference by:
    1. Using from_config() instead of from_pretrained()
    2. Properly formatting prompts with <Speech><SpeechHere></Speech> tags
    3. Using prepare_one_sample() for audio preprocessing
    4. Correctly calling model.generate() with samples and config

    IMPORTANT: Requires specific library versions. See module docstring.
    """

    # SALMONN's expected paths - these are downloaded from HuggingFace
    HF_REPO = "tsinghua-ee/SALMONN"
    BEATS_REPO = "Bencr/beats-checkpoints"

    # Default prompt template matching SALMONN training
    PROMPT_TEMPLATE = "USER: {}\nASSISTANT:"
    SPEECH_TOKEN = "<Speech><SpeechHere></Speech>"

    def __init__(
        self,
        model_name: str = "SALMONN",
        device: str = "cuda",
        use_8bit: bool = False,
        salmonn_repo_path: Optional[str] = None,
        vicuna_path: Optional[str] = None,
        whisper_path: Optional[str] = None,
        beats_path: Optional[str] = None,
        ckpt_path: Optional[str] = None,
    ):
        """Initialize SALMONN wrapper.

        Args:
            model_name: Human-readable identifier for logging
            device: Target device ('cuda' or 'cpu')
            use_8bit: Enable 8-bit quantization (reduces VRAM to ~16GB)
            salmonn_repo_path: Path to cloned SALMONN repo (auto-clones if None)
            vicuna_path: Path to Vicuna-13B-v1.1 weights
            whisper_path: Path to Whisper Large v2 weights
            beats_path: Path to BEATs checkpoint
            ckpt_path: Path to SALMONN checkpoint (salmonn_v1.pth)
        """
        super().__init__(model_name)
        self.device = device if torch.cuda.is_available() else "cpu"
        self.use_8bit = use_8bit and torch.cuda.is_available()

        # Paths - will be resolved in load_model()
        self.salmonn_repo_path = Path(salmonn_repo_path) if salmonn_repo_path else None
        self.vicuna_path = vicuna_path
        self.whisper_path = whisper_path
        self.beats_path = beats_path
        self.ckpt_path = ckpt_path

        # Runtime components
        self.wav_processor = None
        self.config = None
        self.generate_cfg = None

        # Version check warning
        self._check_versions()

    def _check_versions(self) -> None:
        """Check if library versions are compatible with SALMONN."""
        try:
            import transformers
            import peft

            tf_version = transformers.__version__
            peft_version = peft.__version__

            # SALMONN requires transformers==4.28.0 and peft==0.3.0
            if not tf_version.startswith("4.28"):
                print(f"[WARNING] transformers=={tf_version} detected. "
                      f"SALMONN requires transformers==4.28.0")
                print("[WARNING] Garbled output is likely. Consider version downgrade.")

            if not peft_version.startswith("0.3"):
                print(f"[WARNING] peft=={peft_version} detected. "
                      f"SALMONN requires peft==0.3.0")
                print("[WARNING] LoRA weights may not load correctly.")

        except ImportError as e:
            print(f"[WARNING] Could not check versions: {e}")

    def _clone_salmonn_repo(self) -> Path:
        """Clone SALMONN repository if not already present."""
        if self.salmonn_repo_path is None:
            # Default path
            base = Path("/content") if Path("/content").exists() else Path.home()
            self.salmonn_repo_path = base / "SALMONN"

        if self.salmonn_repo_path.exists():
            print(f"[OK] SALMONN repo found at {self.salmonn_repo_path}")
            return self.salmonn_repo_path

        print(f"[CLONING] Downloading SALMONN repository...")
        subprocess.run([
            "git", "clone",
            "--branch", "salmonn",
            "--single-branch",
            "https://github.com/bytedance/SALMONN.git",
            str(self.salmonn_repo_path)
        ], check=True)

        return self.salmonn_repo_path

    def _patch_qformer_imports(self) -> None:
        """Apply monkey-patches for transformers version compatibility.

        The Qformer.py file imports apply_chunking_to_forward from
        transformers.modeling_utils, but in transformers>=4.31 it moved to
        transformers.pytorch_utils.
        """
        qformer_path = self.salmonn_repo_path / "models" / "Qformer.py"

        if not qformer_path.exists():
            return

        content = qformer_path.read_text()

        # Check if patch is needed
        if "from transformers.modeling_utils import" in content and \
           "apply_chunking_to_forward" in content:

            # Apply patch
            patched = content.replace(
                "from transformers.modeling_utils import",
                "from transformers.pytorch_utils import apply_chunking_to_forward\n"
                "from transformers.modeling_utils import"
            )
            patched = patched.replace(
                "apply_chunking_to_forward,",
                ""
            )

            qformer_path.write_text(patched)
            print("[PATCH] Applied Qformer.py import fix for transformers>=4.31")

    def _download_checkpoints(self) -> Dict[str, str]:
        """Download required checkpoints from HuggingFace."""
        from huggingface_hub import hf_hub_download, snapshot_download

        paths = {}

        # Download SALMONN checkpoint
        if self.ckpt_path is None:
            print("[DOWNLOADING] SALMONN checkpoint...")
            self.ckpt_path = hf_hub_download(
                self.HF_REPO,
                "salmonn_v1.pth",
                local_dir=str(self.salmonn_repo_path / "checkpoints")
            )
        paths["ckpt"] = self.ckpt_path
        print(f"[OK] SALMONN checkpoint: {self.ckpt_path}")

        # Download Whisper Large v2
        if self.whisper_path is None:
            print("[DOWNLOADING] Whisper Large v2...")
            self.whisper_path = snapshot_download(
                "openai/whisper-large-v2",
                local_dir=str(self.salmonn_repo_path / "whisper-large-v2")
            )
        paths["whisper"] = self.whisper_path
        print(f"[OK] Whisper: {self.whisper_path}")

        # Download BEATs - try multiple sources
        if self.beats_path is None:
            print("[DOWNLOADING] BEATs checkpoint...")
            # Check common cache locations first
            beats_cache_paths = [
                Path.home() / ".cache" / "salmonn" / "BEATs_iter3_plus_AS2M.pt",
                Path.home() / ".cache" / "salmonn" / "BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt",
                self.salmonn_repo_path / "beats" / "BEATs_iter3_plus_AS2M.pt",
            ]
            for cache_path in beats_cache_paths:
                if cache_path.exists():
                    print(f"[OK] Found cached BEATs at: {cache_path}")
                    self.beats_path = str(cache_path)
                    break

            # If not found, try downloading from HuggingFace
            if self.beats_path is None:
                try:
                    # Try tsinghua-ee/SALMONN repo first (has beats folder)
                    self.beats_path = hf_hub_download(
                        "tsinghua-ee/SALMONN",
                        "beats/BEATs_iter3_plus_AS2M.pt",
                        local_dir=str(self.salmonn_repo_path)
                    )
                except Exception as e:
                    print(f"[WARNING] Could not download BEATs from HF: {e}")
                    # Create directory and download directly from GitHub releases
                    beats_dir = self.salmonn_repo_path / "beats"
                    beats_dir.mkdir(parents=True, exist_ok=True)
                    beats_url = "https://github.com/microsoft/unilm/releases/download/beats/BEATs_iter3_plus_AS2M.pt"
                    beats_file = beats_dir / "BEATs_iter3_plus_AS2M.pt"
                    print(f"[DOWNLOADING] BEATs from GitHub: {beats_url}")
                    import urllib.request
                    urllib.request.urlretrieve(beats_url, beats_file)
                    self.beats_path = str(beats_file)
        paths["beats"] = self.beats_path
        print(f"[OK] BEATs: {self.beats_path}")

        # Download Vicuna-13B-v1.1
        if self.vicuna_path is None:
            print("[DOWNLOADING] Vicuna-13B-v1.1 (this may take a while)...")
            self.vicuna_path = snapshot_download(
                "lmsys/vicuna-13b-v1.1",
                local_dir=str(self.salmonn_repo_path / "vicuna-13b-v1.1")
            )
        paths["vicuna"] = self.vicuna_path
        print(f"[OK] Vicuna: {self.vicuna_path}")

        return paths

    def _build_config(self) -> Dict[str, Any]:
        """Build configuration dict matching SALMONN's expected format."""
        return {
            "model": {
                "llama_path": self.vicuna_path,
                "whisper_path": self.whisper_path,
                "beats_path": self.beats_path,
                "ckpt": self.ckpt_path,

                # Encoder settings
                "freeze_whisper": True,
                "freeze_beats": True,

                # Q-Former settings
                "use_speech_Qformer": True,
                "freeze_speech_QFormer": False,
                "window_level_Qformer": True,
                "num_speech_query_token": 1,
                "second_per_window": 0.333333,
                "second_stride": 0.333333,

                # LoRA settings
                "lora": True,
                "lora_rank": 8,
                "lora_alpha": 32,
                "lora_dropout": 0.1,

                # Prompt settings
                "multi_prompt": False,
                "prompt_template": self.PROMPT_TEMPLATE,
                "max_txt_len": 300,
                "end_sym": "</s>",

                # Memory settings
                "low_resource": self.use_8bit,
                "device_8bit": 0 if self.use_8bit else -1,
            },
            "generate": {
                "max_new_tokens": 200,
                "num_beams": 4,
                "do_sample": False,
                "temperature": 1.0,
                "top_p": 0.9,
                "repetition_penalty": 1.0,
                "length_penalty": 1.0,
            }
        }

    def _prepare_one_sample(
        self,
        wav_path: str,
        cuda_enabled: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Prepare audio sample for SALMONN (matching official utils.py).

        This replicates the prepare_one_sample function from SALMONN's utils.py.
        """
        # Load audio
        audio, sr = sf.read(wav_path)

        # Stereo to mono
        if len(audio.shape) == 2:
            audio = audio[:, 0]

        # Ensure minimum 1 second
        if len(audio) < sr:
            silence = np.zeros(sr - len(audio), dtype=np.float32)
            audio = np.concatenate([audio, silence])

        # Truncate to 30 seconds max
        audio = audio[:sr * 30]

        # Convert to float32 if needed
        audio = audio.astype(np.float32)

        # Extract spectrogram using Whisper feature extractor
        spectrogram = self.wav_processor(
            audio,
            sampling_rate=sr,
            return_tensors="pt"
        )["input_features"]

        # Build sample dict
        samples = {
            "spectrogram": spectrogram,
            "raw_wav": torch.from_numpy(audio).unsqueeze(0),
            "padding_mask": torch.zeros(len(audio), dtype=torch.bool).unsqueeze(0),
        }

        # Move to GPU if enabled
        if cuda_enabled and torch.cuda.is_available():
            samples = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in samples.items()
            }

        return samples

    def _format_prompt(self, user_prompt: str) -> str:
        """Format prompt with speech tokens matching SALMONN training.

        SALMONN expects prompts in this format:
        "USER: <Speech><SpeechHere></Speech> {user_prompt}\nASSISTANT:"
        """
        # Combine speech token with user prompt
        full_prompt = f"{self.SPEECH_TOKEN} {user_prompt.strip()}"

        # Apply template
        return self.PROMPT_TEMPLATE.format(full_prompt)

    def load_model(self) -> None:
        """Load SALMONN model using official from_config method."""
        if self._is_loaded:
            print(f"[OK] {self.model_name} already loaded")
            return

        if not self.check_memory_availability():
            raise RuntimeError(
                f"Insufficient VRAM for {self.model_name}. "
                f"Required: {self.get_memory_requirements()['peak_vram_gb']:.1f}GB"
            )

        print(f"[LOADING] Loading {self.model_name}...")

        try:
            # Step 1: Clone repo and add to path
            self._clone_salmonn_repo()
            self._patch_qformer_imports()

            salmonn_path = str(self.salmonn_repo_path.absolute())
            if salmonn_path not in sys.path:
                sys.path.insert(0, salmonn_path)
                print(f"[OK] Added {salmonn_path} to sys.path")

            # Step 2: Download checkpoints
            self._download_checkpoints()

            # Step 3: Build config
            config = self._build_config()
            self.generate_cfg = config["generate"]

            # Step 4: Load Whisper feature extractor
            from transformers import WhisperFeatureExtractor
            self.wav_processor = WhisperFeatureExtractor.from_pretrained(
                self.whisper_path
            )
            print("[OK] Whisper feature extractor loaded")

            # Step 5: Load SALMONN model
            from models.salmonn import SALMONN

            # Create OmegaConf-like config object
            from omegaconf import OmegaConf
            omega_config = OmegaConf.create(config)

            print("[LOADING] Initializing SALMONN model...")
            self.model = SALMONN.from_config(omega_config.model)

            # Move to device
            self.model.to(self.device)
            self.model.eval()

            print(f"[OK] {self.model_name} loaded successfully")

            # Step 6: Verify LoRA loaded
            self._verify_lora_weights()

            # Step 7: Warm-up
            self._warmup_inference()

            self._is_loaded = True

        except Exception as e:
            self.unload()
            raise RuntimeError(f"Failed to load {self.model_name}: {e}") from e

    def _verify_lora_weights(self) -> None:
        """Verify LoRA adapter weights are loaded and non-zero."""
        lora_found = False
        zero_weights = []

        for name, param in self.model.named_parameters():
            if 'lora' in name.lower():
                lora_found = True
                if param.abs().sum() == 0:
                    zero_weights.append(name)

        if not lora_found:
            print("[WARNING] No LoRA parameters found - adapter may not be loaded")
        elif zero_weights:
            print(f"[WARNING] {len(zero_weights)} LoRA parameters are zeros:")
            for name in zero_weights[:5]:
                print(f"  - {name}")
            print("[WARNING] This indicates LoRA weights failed to load correctly")
        else:
            print("[OK] LoRA weights verified non-zero")

    def _warmup_inference(self) -> None:
        """Run warm-up inference to stabilize memory."""
        print("[WARMUP] Running warm-up inference...")

        try:
            import tempfile

            # Create 1s dummy audio
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                dummy = np.zeros(16000, dtype=np.float32)
                sf.write(f.name, dummy, 16000)
                dummy_path = f.name

            # Run inference
            _ = self.generate_caption(dummy_path, "Test")

            os.remove(dummy_path)

            if torch.cuda.is_available():
                peak = torch.cuda.max_memory_allocated() / 1e9
                print(f"[OK] Warm-up complete. Peak VRAM: {peak:.2f}GB")

        except Exception as e:
            warnings.warn(f"Warm-up failed (non-critical): {e}")

    def generate_caption(self, audio_path: str, prompt: str) -> str:
        """Generate caption using SALMONN's official inference pipeline.

        Args:
            audio_path: Path to audio file (WAV recommended, 16kHz)
            prompt: Natural language query

        Returns:
            Generated caption string
        """
        if not self._is_loaded:
            raise RuntimeError(f"{self.model_name} not loaded. Call load_model() first.")

        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio not found: {audio_path}")

        try:
            # Prepare audio sample
            samples = self._prepare_one_sample(audio_path, cuda_enabled=True)

            # Format prompt
            formatted_prompt = self._format_prompt(prompt)

            # Generate with mixed precision
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    outputs = self.model.generate(
                        samples,
                        self.generate_cfg,
                        prompts=[formatted_prompt]
                    )

            # Extract text (returns list)
            response = outputs[0] if isinstance(outputs, list) else outputs

            # Clean up response
            response = self._clean_response(response)

            return response

        except Exception as e:
            raise RuntimeError(f"Inference failed for {audio_path}: {e}") from e

    def _clean_response(self, text: str) -> str:
        """Clean up model output."""
        if not isinstance(text, str):
            text = str(text)

        # Remove common artifacts
        text = text.strip()

        # Remove repeated patterns (symptom of broken LoRA)
        lines = text.split('\n')
        unique_lines = []
        for line in lines:
            if line.strip() and line not in unique_lines:
                unique_lines.append(line)
        text = '\n'.join(unique_lines)

        return text

    def get_memory_requirements(self) -> Dict[str, float]:
        """Return VRAM requirements."""
        if self.use_8bit:
            return {'min_vram_gb': 13.0, 'peak_vram_gb': 16.0}
        return {'min_vram_gb': 24.0, 'peak_vram_gb': 28.0}

    def unload(self) -> None:
        """Unload model and free VRAM."""
        if hasattr(self, 'wav_processor') and self.wav_processor is not None:
            del self.wav_processor
            self.wav_processor = None

        if hasattr(self, 'config'):
            self.config = None

        if hasattr(self, 'generate_cfg'):
            self.generate_cfg = None

        super().unload()


# Alias for backward compatibility
SalmonnWrapper = SalmonnWrapperFixed
