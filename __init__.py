"""
AcousticLLMevalGeneralized - Unified Audio Captioning Model Evaluation Framework

This package provides a standardized interface for evaluating multiple audio
captioning models (NatureLM, SALMONN, Qwen-Audio) on bioacoustic datasets.

Quick Start:
    >>> from model_registry import get_model
    >>>
    >>> # Instantiate model
    >>> model = get_model("naturelm")
    >>>
    >>> # Load and run inference
    >>> model.load_model()
    >>> caption = model.generate_caption("audio.mp3", "What species is this?")
    >>> print(caption)
    >>>
    >>> # Clean up VRAM
    >>> model.unload()

Available Models:
    - naturelm: NatureLM-audio (Llama-3.1-8B for bioacoustics)
    - salmonn: SALMONN (Whisper+BEATs+Vicuna-13B with 8-bit quantization)
    - qwen-audio: Qwen2-Audio-7B (7B parameter audio-language model)
"""

__version__ = "0.2.0"
__author__ = "AcousticLLMEval Research Team"

from base_model import AudioCaptioningModel
from naturelm_wrapper import NatureLMWrapper
from salmonn_wrapper import SalmonnWrapper
from qwen_wrapper import QwenAudioWrapper
from model_registry import ModelRegistry, get_model, list_models, get_implemented_models

__all__ = [
    "AudioCaptioningModel",
    "NatureLMWrapper",
    "SalmonnWrapper",
    "QwenAudioWrapper",
    "ModelRegistry",
    "get_model",
    "list_models",
    "get_implemented_models",
]
