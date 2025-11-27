"""
Model registry for managing available audio captioning models.

This module provides a centralized registry for discovering and instantiating
audio captioning models. It enables dynamic model selection based on availability,
VRAM constraints, and research requirements.

Usage:
    >>> from model_registry import ModelRegistry
    >>>
    >>> # List available models
    >>> registry = ModelRegistry()
    >>> print(registry.list_models())
    >>>
    >>> # Get model instance
    >>> model = registry.get_model("naturelm")
    >>> model.load_model()
    >>> caption = model.generate_caption("audio.mp3", "What species is this?")
"""

from typing import Dict, List, Optional, Type
import warnings

from base_model import AudioCaptioningModel
from naturelm_wrapper import NatureLMWrapper
from salmonn_wrapper import SalmonnWrapper
from qwen_wrapper import QwenAudioWrapper


class ModelRegistry:
    """Central registry for audio captioning models.

    This class manages model metadata and provides factory methods for
    instantiating model wrappers. It supports:
    - Discovery of available models
    - VRAM-based filtering
    - Implementation status tracking
    - Consistent model initialization

    Attributes:
        _models: Dictionary mapping model IDs to metadata and wrapper classes
    """

    def __init__(self):
        """Initialize the model registry with available models."""
        self._models: Dict[str, Dict] = {
            "naturelm": {
                "name": "NatureLM-audio",
                "class": NatureLMWrapper,
                "status": "implemented",
                "description": "Llama-3.1-8B fine-tuned for bioacoustics. "
                               "State-of-the-art on BEANS-Zero benchmark.",
                "vram_gb": 10.0,
                "paper": "Robinson et al. (2025), ICLR 2025",
                "huggingface": "EarthSpeciesProject/NatureLM-audio",
                "specialization": [
                    "Species classification",
                    "Audio captioning",
                    "Lifestage identification",
                    "Bioacoustic event detection"
                ],
                "supported_formats": ["WAV", "MP3", "OGG", "FLAC"],
                "notes": "Requires Llama-3.1 access via HuggingFace"
            },
            "salmonn": {
                "name": "SALMONN",
                "class": SalmonnWrapper,
                "status": "implemented",
                "description": "Speech Audio Language Music Open Neural Network. "
                               "Multi-modal model supporting speech, audio events, and music.",
                "vram_gb": 16.0,  # With 8-bit quantization (26GB without)
                "paper": "Tang et al. (2024), ICLR 2024",
                "huggingface": "tsinghua-ee/SALMONN",
                "specialization": [
                    "General audio understanding",
                    "Speech recognition",
                    "Music understanding",
                    "Audio event classification",
                    "Audio captioning",
                    "Speech translation"
                ],
                "supported_formats": ["WAV", "MP3"],
                "notes": "Uses 8-bit quantization by default. Requires cloning bytedance/SALMONN repo."
            },
            "qwen-audio": {
                "name": "Qwen2-Audio-7B",
                "class": QwenAudioWrapper,
                "status": "implemented",
                "description": "Qwen's 8B parameter audio-language model for general audio understanding. "
                               "Supports audio captioning, Q&A, and classification tasks.",
                "vram_gb": 14.0,  # With BF16 precision (8GB with 4-bit quantization)
                "paper": "Qwen Team (2024)",
                "huggingface": "Qwen/Qwen2-Audio-7B",
                "specialization": [
                    "Audio captioning",
                    "Audio question answering",
                    "Audio classification",
                    "Audio event detection",
                    "General audio understanding"
                ],
                "supported_formats": ["WAV", "MP3", "FLAC", "OGG"],
                "notes": "BF16 precision by default (~14GB). Optional 4-bit quantization reduces to ~8GB. "
                         "Requires transformers>=4.35.0 and librosa for audio preprocessing."
            },
        }

    def list_models(
        self,
        status_filter: Optional[str] = None,
        max_vram_gb: Optional[float] = None
    ) -> List[Dict]:
        """List available models with optional filtering.

        Args:
            status_filter: Filter by implementation status
                          ('implemented', 'planned', or None for all)
            max_vram_gb: Only include models requiring <= this VRAM

        Returns:
            List of model metadata dictionaries

        Example:
            >>> registry = ModelRegistry()
            >>>
            >>> # List all models
            >>> all_models = registry.list_models()
            >>>
            >>> # List only implemented models
            >>> ready = registry.list_models(status_filter="implemented")
            >>>
            >>> # List models that fit in 12GB VRAM
            >>> small = registry.list_models(max_vram_gb=12.0)
        """
        filtered = []

        for model_id, metadata in self._models.items():
            # Apply status filter
            if status_filter and metadata["status"] != status_filter:
                continue

            # Apply VRAM filter
            if max_vram_gb and metadata["vram_gb"] > max_vram_gb:
                continue

            # Add model_id to metadata for convenience
            model_info = {"id": model_id, **metadata}
            filtered.append(model_info)

        return filtered

    def get_model(self, model_id: str, **kwargs) -> AudioCaptioningModel:
        """Instantiate a model wrapper by ID.

        Args:
            model_id: Model identifier (e.g., "naturelm", "salmonn")
            **kwargs: Additional arguments passed to model constructor

        Returns:
            Instantiated model wrapper (subclass of AudioCaptioningModel)

        Raises:
            KeyError: If model_id is not registered
            NotImplementedError: If model is planned but not yet implemented

        Example:
            >>> registry = ModelRegistry()
            >>> model = registry.get_model("naturelm")
            >>> model.load_model()
        """
        if model_id not in self._models:
            available = ", ".join(self._models.keys())
            raise KeyError(
                f"Model '{model_id}' not found in registry. "
                f"Available: {available}"
            )

        metadata = self._models[model_id]

        # Check implementation status
        if metadata["status"] != "implemented":
            raise NotImplementedError(
                f"Model '{model_id}' is registered but not yet implemented. "
                f"Status: {metadata['status']}"
            )

        # Get wrapper class
        wrapper_class = metadata["class"]
        if wrapper_class is None:
            raise NotImplementedError(
                f"Wrapper class for '{model_id}' is not defined"
            )

        # Instantiate and return
        return wrapper_class(**kwargs)

    def get_model_info(self, model_id: str) -> Dict:
        """Get detailed metadata for a specific model.

        Args:
            model_id: Model identifier

        Returns:
            Dictionary with model metadata

        Raises:
            KeyError: If model_id is not registered
        """
        if model_id not in self._models:
            available = ", ".join(self._models.keys())
            raise KeyError(
                f"Model '{model_id}' not found. Available: {available}"
            )

        return {"id": model_id, **self._models[model_id]}

    def get_implemented_models(self) -> List[str]:
        """Get list of model IDs that are fully implemented.

        Returns:
            List of model IDs (strings) with status='implemented'

        Example:
            >>> registry = ModelRegistry()
            >>> implemented = registry.get_implemented_models()
            >>> print(implemented)
            ['naturelm']
        """
        return [
            model_id for model_id, metadata in self._models.items()
            if metadata["status"] == "implemented"
        ]

    def check_model_compatibility(
        self,
        model_id: str,
        available_vram_gb: Optional[float] = None
    ) -> Dict[str, bool]:
        """Check if a model is compatible with system requirements.

        Args:
            model_id: Model identifier
            available_vram_gb: Available VRAM in GB (auto-detected if None)

        Returns:
            Dictionary with compatibility checks:
                {
                    'is_implemented': bool,
                    'vram_sufficient': bool,
                    'can_run': bool  # Overall compatibility
                }

        Example:
            >>> registry = ModelRegistry()
            >>> compat = registry.check_model_compatibility("naturelm", 12.0)
            >>> if compat['can_run']:
            ...     model = registry.get_model("naturelm")
        """
        import torch

        if model_id not in self._models:
            return {
                'is_implemented': False,
                'vram_sufficient': False,
                'can_run': False
            }

        metadata = self._models[model_id]

        # Check implementation status
        is_implemented = metadata["status"] == "implemented"

        # Check VRAM
        if available_vram_gb is None and torch.cuda.is_available():
            available_vram_gb = torch.cuda.mem_get_info()[0] / 1e9

        vram_sufficient = (
            available_vram_gb is not None and
            available_vram_gb >= metadata["vram_gb"] * 1.1  # 10% margin
        )

        return {
            'is_implemented': is_implemented,
            'vram_sufficient': vram_sufficient,
            'can_run': is_implemented and vram_sufficient
        }

    def print_model_summary(self, model_id: Optional[str] = None) -> None:
        """Print human-readable summary of model(s).

        Args:
            model_id: Specific model ID, or None to print all models

        Example:
            >>> registry = ModelRegistry()
            >>> registry.print_model_summary()  # All models
            >>> registry.print_model_summary("naturelm")  # Specific model
        """
        if model_id:
            models_to_print = [model_id]
        else:
            models_to_print = list(self._models.keys())

        for mid in models_to_print:
            if mid not in self._models:
                print(f"[!] Model '{mid}' not found in registry\n")
                continue

            metadata = self._models[mid]

            # Status indicator
            status_indicator = "[OK]" if metadata["status"] == "implemented" else "[PLANNED]"

            print(f"{status_indicator} {metadata['name']} (ID: {mid})")
            print(f"   Status: {metadata['status'].upper()}")
            print(f"   VRAM: {metadata['vram_gb']:.1f}GB")
            print(f"   Description: {metadata['description']}")

            if metadata.get('huggingface'):
                print(f"   HuggingFace: {metadata['huggingface']}")

            if metadata.get('specialization'):
                print(f"   Specialization:")
                for spec in metadata['specialization']:
                    print(f"      - {spec}")

            if metadata.get('notes'):
                print(f"   Notes: {metadata['notes']}")

            print()  # Blank line between models

    def __repr__(self) -> str:
        """String representation of registry."""
        total = len(self._models)
        implemented = len(self.get_implemented_models())
        return (
            f"ModelRegistry(total={total}, implemented={implemented}, "
            f"planned={total - implemented})"
        )


# Convenience function for quick model instantiation
def get_model(model_id: str, **kwargs) -> AudioCaptioningModel:
    """Quick factory function for model instantiation.

    This is a convenience wrapper around ModelRegistry.get_model().

    Args:
        model_id: Model identifier (e.g., "naturelm")
        **kwargs: Additional arguments for model constructor

    Returns:
        Instantiated model wrapper

    Example:
        >>> from model_registry import get_model
        >>> model = get_model("naturelm")
        >>> model.load_model()
    """
    registry = ModelRegistry()
    return registry.get_model(model_id, **kwargs)


# Module-level registry instance for convenience
_global_registry = ModelRegistry()


def list_models(**kwargs) -> List[Dict]:
    """Convenience function to list models from global registry."""
    return _global_registry.list_models(**kwargs)


def get_implemented_models() -> List[str]:
    """Convenience function to get implemented models from global registry."""
    return _global_registry.get_implemented_models()


if __name__ == "__main__":
    # Demo usage when run as script
    print("=== Audio Captioning Model Registry ===\n")

    registry = ModelRegistry()
    print(f"{registry}\n")

    print("All Models:")
    print("-" * 60)
    registry.print_model_summary()

    print("\nImplemented Models:")
    print("-" * 60)
    implemented = registry.get_implemented_models()
    print(f"Ready for use: {', '.join(implemented)}\n")

    print("\nVRAM-Constrained (<=12GB):")
    print("-" * 60)
    small_models = registry.list_models(max_vram_gb=12.0)
    for model in small_models:
        print(f"  - {model['name']} ({model['vram_gb']}GB) [{model['status']}]")
