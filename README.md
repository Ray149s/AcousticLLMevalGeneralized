# AcousticLLMevalGeneralized

**Multi-Model Bioacoustic Captioning Evaluation Framework**

A generalized framework for evaluating audio-language models on bioacoustic captioning tasks. Extends the Gemini evaluation pipeline to open-weights models (Qwen2-Audio, NatureLM).

---

## Evaluation Status

| Model | Status | Configs | Samples | Total Predictions |
|-------|--------|---------|---------|-------------------|
| **Qwen2-Audio-7B** | Complete | 12 | 500 | 6,000 |
| **NatureLM-audio** | Complete | 12 | 500 | 6,000 |
| SALMONN | Future Work | - | - | - |

**Total: 24,000 predictions** across 24 configurations (2 models x 4 prompts x 3 shots)

---

## Project Structure

```
AcousticLLMevalGeneralized/
├── base_model.py                      # Abstract interface for model wrappers
├── qwen_wrapper.py                    # Qwen2-Audio-7B-Instruct wrapper
├── naturelm_wrapper.py                # NatureLM-audio wrapper
├── prompt_config.py                   # 4 prompt roles + 3 shot configurations
├── run_full_evaluation.py             # Main evaluation runner
├── colab_orchestrator.ipynb           # Google Colab notebook for evaluation
├── compute_spider_colab.ipynb         # SPIDEr score computation (multi-model)
├── animalspeak_spider_benchmark.jsonl # Benchmark dataset (500 samples)
├── requirements_eval_env.txt          # Tested dependencies (Lambda H100)
├── outputs/
│   └── lambda_full_eval/              # Evaluation results (24 JSON files)
└── salmonn_future_work/               # SALMONN implementation (future work)
```

---

## Supported Models

### Qwen2-Audio-7B-Instruct
- **Provider**: Alibaba Qwen
- **VRAM**: ~14GB (BF16)
- **Status**: Fully implemented and tested
- **HuggingFace**: [Qwen/Qwen2-Audio-7B-Instruct](https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct)

### NatureLM-audio
- **Provider**: Earth Species Project
- **VRAM**: ~10GB (BF16)
- **Status**: Fully implemented and tested
- **HuggingFace**: [EarthSpeciesProject/NatureLM-audio](https://huggingface.co/EarthSpeciesProject/NatureLM-audio)

### SALMONN (Future Work)
- **Provider**: ByteDance/Tsinghua
- **VRAM**: ~29GB (FP16)
- **Status**: Requires separate environment due to dependency conflicts
- **Documentation**: See `salmonn_future_work/SALMONN_FIX_DOCUMENTATION.md`

---

## Evaluation Configuration

### Prompt Roles (4)
| Role | Description |
|------|-------------|
| `baseline` | Standard captioning prompt |
| `ornithologist` | Expert birder persona |
| `skeptical` | Conservative, uncertainty-aware |
| `multi-taxa` | Multi-species detection focus |

### Shot Configurations (3)
- **0-shot**: No examples
- **3-shot**: 3 text-only examples
- **5-shot**: 5 text-only examples

---

## Quick Start

### Prerequisites
- Python 3.10+
- CUDA GPU with 16GB+ VRAM (or use Google Colab A100)
- HuggingFace account with model access

### Local Evaluation

```bash
# Install dependencies
pip install -r requirements_eval_env.txt

# Run evaluation (both models, all configs)
python run_full_evaluation.py --models qwen naturelm --output-dir ./outputs/eval

# Run specific configuration
python run_full_evaluation.py --models qwen --prompts baseline --shots 3 --max-samples 100
```

### Google Colab

1. Upload `colab_orchestrator.ipynb` to Google Colab
2. Set runtime to A100 GPU
3. Run all cells (restart after Step 2 for Pillow fix)
4. Results saved to Google Drive

### SPIDEr Score Computation

1. Upload `compute_spider_colab.ipynb` to Google Colab
2. Upload result JSON files from `outputs/lambda_full_eval/`
3. Run all cells
4. Download SPIDEr scores

---

## Output Format

Each evaluation produces a JSON file:

```json
{
  "model": "qwen",
  "prompt_version": "baseline",
  "n_shots": 3,
  "samples_tested": 500,
  "successful": 500,
  "avg_latency": 0.97,
  "results": [
    {
      "id": 15107,
      "species": "Squirrel Treefrog",
      "reference": "a chorus of squirrel treefrogs...",
      "prediction": "Frogs calling near water...",
      "latency": 0.74,
      "success": true
    }
  ]
}
```

---

## Results Summary

### Average Latency by Model
| Model | 0-shot | 3-shot | 5-shot |
|-------|--------|--------|--------|
| Qwen2-Audio | 0.74s | 0.97s | 1.50s |
| NatureLM | 0.97s | 1.23s | 1.35s |

### Success Rate
All configurations achieved **100% success rate** (500/500 samples).

---

## API Reference

### Base Model Interface

```python
from base_model import AudioCaptioningModel

class MyModelWrapper(AudioCaptioningModel):
    def load_model(self) -> None:
        """Load model weights onto GPU."""
        pass

    def generate_caption(self, audio_path: str, prompt: str) -> str:
        """Generate caption for audio file."""
        pass

    def get_memory_requirements(self) -> dict:
        """Return VRAM requirements."""
        return {"min_vram_gb": 10.0, "peak_vram_gb": 12.0}

    def unload(self) -> None:
        """Free GPU memory."""
        pass
```

---

## Known Issues

### NatureLM Timestamp Annotations
NatureLM outputs include timestamp annotations for longer audio:
```
American Woodcock calling...
#10.00s - 20.00s#: American Woodcock
#20.00s - 30.00s#: American Woodcock calling...
```
The SPIDEr notebook automatically preprocesses these before scoring.

### Colab Fixes Required
1. **Pillow**: Upgrade to >=10.0.0, restart runtime
2. **NatureLM imports**: Patch Qformer.py for transformers>=4.40

See `colab_orchestrator.ipynb` Step 2-3 for automatic fixes.

---

## Related Projects

- **Parent Project**: [AcousticLLMEval](https://github.com/Ray149s/AcousticLLMEval) - Gemini Flash/Pro evaluation
- **NatureLM**: [earthspecies/NatureLM-audio](https://github.com/earthspecies/NatureLM-audio)
- **Qwen2-Audio**: [QwenLM/Qwen2-Audio](https://github.com/QwenLM/Qwen2-Audio)

---

## License

This project is part of the AcousticLLMEval research initiative.

---

## References

- **NatureLM-audio**: Robinson, D., et al. (2025). NatureLM-audio: An Audio-Language Foundation Model for Bioacoustics. ICLR 2025.
- **Qwen2-Audio**: Chu, Y., et al. (2024). Qwen2-Audio Technical Report. arXiv:2407.10759.
- **SALMONN**: Tang, C., et al. (2024). SALMONN: Towards Generic Hearing Abilities for Large Language Models. ICLR 2024.
