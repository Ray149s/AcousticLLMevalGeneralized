# AcousticLLMevalGeneralized

**Unified Audio Captioning Model Evaluation Framework**

A production-ready abstraction layer for evaluating multiple audio captioning models (NatureLM, SALMONN) on bioacoustic datasets. Designed for resource-constrained environments like Google Colab with aggressive VRAM management.

---

## Features

- **Unified Interface**: Abstract base class (`AudioCaptioningModel`) ensures consistent API across all models
- **Memory-Safe**: Built-in VRAM checks and aggressive cleanup for sequential evaluations
- **Model Registry**: Centralized discovery and instantiation of available models
- **Production-Ready**: Proper error handling, warm-up inference, and comprehensive documentation

---

## Project Structure

```
AcousticLLMevalGeneralized/
├── base_model.py                      # Abstract base class for all models
├── naturelm_wrapper.py                # NatureLM-audio implementation
├── salmonn_wrapper.py                 # SALMONN implementation
├── model_registry.py                  # Model discovery and factory
├── universal_evaluator.py             # Model-agnostic evaluation framework
├── colab_orchestrator.ipynb           # Production Colab notebook
├── validate_pipeline.py               # Pipeline validation suite
├── animalspeak_spider_benchmark.jsonl # Benchmark dataset (500 samples)
├── requirements.txt                   # Dependencies
├── __init__.py                        # Package initialization
└── README.md                          # This file
```

---

## Installation

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (10GB+ VRAM recommended)
- HuggingFace account with Llama-3.1 access

### Setup

```bash
# 1. Clone NatureLM repository
git clone https://github.com/earthspecies/NatureLM-audio
cd NatureLM-audio

# 2. Install dependencies
pip install -e .[gpu]  # or pip install -e . for CPU-only

# 3. Set up HuggingFace authentication
export HF_TOKEN="your_token_here"
huggingface-cli login

# 4. Request Llama-3.1 access
# Visit: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
```

---

## Quick Start

### Basic Usage

```python
from model_registry import get_model

# Instantiate model
model = get_model("naturelm")

# Check VRAM availability
if model.check_memory_availability():
    # Load model
    model.load_model()

    # Generate caption
    caption = model.generate_caption(
        "path/to/audio.mp3",
        "What is the common name for the focal species in the audio?"
    )
    print(caption)
    # Output: "#0.00s - 10.00s#: Green Treefrog\n"

    # Clean up VRAM
    model.unload()
else:
    print("Insufficient VRAM!")
```

### Batch Processing

```python
from model_registry import get_model

model = get_model("naturelm")
model.load_model()

audio_paths = ["audio1.mp3", "audio2.mp3", "audio3.mp3"]
prompts = ["What species is this?"] * 3

captions = model.generate_captions_batch(audio_paths, prompts)

for path, caption in zip(audio_paths, captions):
    print(f"{path}: {caption}")

model.unload()
```

### Model Discovery

```python
from model_registry import ModelRegistry

registry = ModelRegistry()

# List all models
registry.print_model_summary()

# Filter by VRAM constraint
small_models = registry.list_models(max_vram_gb=12.0)

# Check compatibility
compat = registry.check_model_compatibility("naturelm", available_vram_gb=10.0)
if compat['can_run']:
    print("Model is compatible!")
```

---

## Implemented Models

### NatureLM-audio

- **Status**: ✅ Implemented
- **Architecture**: Llama-3.1-8B-Instruct fine-tuned on bioacoustic data
- **VRAM**: ~10GB (peak with bfloat16)
- **Specialization**: Species classification, audio captioning, lifestage identification
- **Reference**: Robinson et al. (2025), ICLR 2025
- **HuggingFace**: [EarthSpeciesProject/NatureLM-audio](https://huggingface.co/EarthSpeciesProject/NatureLM-audio)

### SALMONN

- **Status**: ✅ Implemented
- **Architecture**: Multi-modal audio understanding (speech, music, events)
- **VRAM**: ~16GB (with 8-bit quantization)
- **Specialization**: General audio understanding, speech recognition, audio event classification
- **Reference**: Tang et al. (2024), ICLR 2024
- **HuggingFace**: [tsinghua-ee/SALMONN](https://huggingface.co/tsinghua-ee/SALMONN)

---

## API Reference

### `AudioCaptioningModel` (Abstract Base Class)

**Methods:**

- `load_model()` - Load model weights and initialize inference
- `generate_caption(audio_path, prompt)` - Generate caption for single audio file
- `get_memory_requirements()` - Return VRAM requirements dict
- `check_memory_availability()` - Verify VRAM before loading
- `unload()` - Aggressively free VRAM
- `is_loaded()` - Check if model is currently loaded

### `NatureLMWrapper`

Extends `AudioCaptioningModel` with NatureLM-specific features:

- `generate_captions_batch(audio_paths, prompts)` - Batch inference
- Sliding window audio processing (configurable window/hop length)
- Automatic warm-up inference for stable VRAM

**Constructor Args:**
- `model_name` (str): Human-readable identifier
- `window_length_seconds` (float): Audio chunk size (default: 10.0)
- `hop_length_seconds` (float): Stride between windows (default: 10.0)
- `device` (str): Target device ('cuda' or 'cpu')

### `ModelRegistry`

**Methods:**

- `list_models(status_filter, max_vram_gb)` - List models with filtering
- `get_model(model_id, **kwargs)` - Instantiate model wrapper
- `get_model_info(model_id)` - Get detailed metadata
- `get_implemented_models()` - List ready-to-use models
- `check_model_compatibility(model_id, available_vram_gb)` - Compatibility check
- `print_model_summary(model_id)` - Human-readable summary

---

## Design Principles

### Memory Safety

All models implement aggressive VRAM cleanup:

```python
def unload(self):
    # Delete model components
    del self.model
    del self.processor

    # Force garbage collection
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
```

This prevents OOM errors in sequential evaluation loops (critical for Colab).

### Warm-Up Inference

Models perform warm-up inference during `load_model()`:

- JIT-compiles CUDA kernels
- Stabilizes VRAM usage
- Ensures accurate peak memory measurements
- Prevents first-inference slowdowns

### Error Handling

All methods include comprehensive error handling:

- Pre-flight VRAM checks before loading
- File existence validation
- Model state verification
- Informative error messages with resolution steps

---

## Memory Requirements

| Model      | Min VRAM | Peak VRAM | Precision |
|------------|----------|-----------|-----------|
| NatureLM   | 8.5GB    | 10.0GB    | bfloat16  |
| SALMONN    | TBD      | ~16GB     | TBD       |

**Note:** All requirements include 10% safety margin for stability.

---

## Troubleshooting

### HuggingFace Authentication Errors

```
ERROR: Cannot access EarthSpeciesProject/NatureLM-audio
```

**Solution:**
1. Request Llama-3.1 access: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
2. Set environment variable: `export HF_TOKEN="your_token"`
3. Run: `huggingface-cli login`

### Out of Memory (OOM)

```
RuntimeError: CUDA out of memory
```

**Solution:**
1. Check available VRAM: `model.check_memory_availability()`
2. Use smaller models or reduce batch size
3. Ensure `model.unload()` is called between evaluations
4. Restart Python kernel to clear fragmented memory

### Import Errors

```
ImportError: No module named 'NatureLM'
```

**Solution:**
Install NatureLM package:
```bash
git clone https://github.com/earthspecies/NatureLM-audio
cd NatureLM-audio
pip install -e .[gpu]
```

---

## Roadmap

### Phase 1: Core Abstraction (COMPLETE ✅)
- ✅ Abstract base class (`base_model.py`)
- ✅ NatureLM wrapper (`naturelm_wrapper.py`)
- ✅ Model registry (`model_registry.py`)

### Phase 2: SALMONN Integration (COMPLETE ✅)
- ✅ SALMONN wrapper implementation (`salmonn_wrapper.py`)
- ✅ 8-bit quantization support for A100
- ✅ Validation and testing

### Phase 3: Universal Evaluator (COMPLETE ✅)
- ✅ Model-agnostic evaluation framework (`universal_evaluator.py`)
- ✅ Checkpoint/resume functionality
- ✅ AnimalSpeak SPIDEr benchmark integration (500 samples)

### Phase 4: Colab Orchestration (COMPLETE ✅)
- ✅ Production-ready Colab notebook (`colab_orchestrator.ipynb`)
- ✅ Pipeline validation suite (`validate_pipeline.py`)
- ✅ Automated memory management and cleanup

---

## License

This project is part of the AcousticLLMEval research initiative.

---

## References

- **NatureLM-audio**: Robinson, D., Miron, M., Hagiwara, M., & Pietquin, O. (2025). NatureLM-audio: An Audio-Language Foundation Model for Bioacoustics. ICLR 2025.
- **HuggingFace Model**: https://huggingface.co/EarthSpeciesProject/NatureLM-audio
- **GitHub Repository**: https://github.com/earthspecies/NatureLM-audio

---

## Contact

For issues or questions about this framework, please refer to the main AcousticLLMEval project documentation.
