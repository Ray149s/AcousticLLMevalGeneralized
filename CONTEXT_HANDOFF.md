# Context Handoff Summary

## Project Status: AcousticLLMevalGeneralized — IN PROGRESS

**GitHub Repo:** https://github.com/Ray149s/AcousticLLMevalGeneralized

**Last Updated:** 2025-11-27 (end of session)

---

## What Was Built
A generalized bioacoustic LLM evaluation framework for Google Colab A100, supporting:
- **NatureLM** (~10GB, bfloat16) - Earth Species Project
- **SALMONN** (~28GB, FP16) - ByteDance/Tsinghua
- **Qwen2-Audio-7B** (~14GB, BF16) - Alibaba Qwen

## Key Files
| File | Purpose |
|------|---------|
| `base_model.py` | Abstract interface all wrappers inherit from |
| `naturelm_wrapper.py` | NatureLM implementation |
| `salmonn_wrapper.py` | SALMONN implementation (FIXED: import path) |
| `qwen_wrapper.py` | Qwen2-Audio-7B implementation |
| `universal_evaluator.py` | Model-agnostic evaluation with checkpoint/resume |
| `model_registry.py` | Centralized model management |
| `colab_orchestrator.ipynb` | Main Colab notebook |
| `animalspeak_spider_benchmark.jsonl` | 500-sample benchmark dataset |

---

## Current Evaluation Status (10-sample proof of concept)

| Model | Status | Issue | Fix Applied to GitHub? |
|-------|--------|-------|------------------------|
| **Qwen-Audio** | ✅ SUCCESS | None | N/A |
| **NatureLM** | ❌ FAILED | `wrapper_args` had wrong `model_path` key | ✅ Yes (notebook) |
| **SALMONN** | ❌ FAILED | Import was `from model` instead of `from models.salmonn` | ✅ Yes (wrapper) |

**Qwen-Audio Results:** 10/10 samples, 6.52s average latency

---

## Fixes Pushed to GitHub (2025-11-27)

### Fix 1: NatureLM Config (colab_orchestrator.ipynb)
```python
# BEFORE (wrong):
"NatureLM": {
    "wrapper_args": {"model_path": "EarthSpeciesProject/NatureLM-audio"},
}

# AFTER (correct):
"NatureLM": {
    "wrapper_args": {},  # Empty - uses defaults from wrapper
}
```

### Fix 2: SALMONN Import Path (salmonn_wrapper.py line 249)
```python
# BEFORE (wrong):
from model import SALMONN

# AFTER (correct):
from models.salmonn import SALMONN
```

### Fix 3: Dependencies (colab_orchestrator.ipynb Cell 2)
- Added `pip install -q openai-whisper` for SALMONN
- Made `flash-attn` installation non-blocking

---

## RESUME INSTRUCTIONS FOR TOMORROW

The user has NOT yet applied the fixes in Colab. Run these steps:

### Step 1: Pull fixes from GitHub
```python
!cd /content/AcousticLLMevalGeneralized && git pull
```

### Step 2: Apply runtime fixes (new cell after Cell 4)
```python
# Fix NatureLM config
MODEL_REGISTRY["NatureLM"]["wrapper_args"] = {}

# Reload SALMONN wrapper with fixed import
import importlib
import salmonn_wrapper
importlib.reload(salmonn_wrapper)
from salmonn_wrapper import SalmonnWrapper
MODEL_REGISTRY["SALMONN"]["wrapper_class"] = SalmonnWrapper

print("✓ Both fixes applied")
```

### Step 3: Re-run evaluation
Re-run Cell 7 (Main Evaluation Loop)

---

## Technical Context

### SALMONN Repository Structure
The bytedance/SALMONN repo (branch: `salmonn`) has this structure:
```
/content/SALMONN/
├── models/
│   ├── __init__.py
│   ├── salmonn.py      <-- SALMONN class is HERE
│   ├── modeling_llama.py
│   ├── modeling_whisper.py
│   ├── Qformer.py
│   └── beats/
├── configs/
├── cli_inference.py
└── ...
```

### VRAM Requirements (A100 40GB)
- NatureLM: ~10GB (bfloat16)
- Qwen2-Audio: ~14GB (BF16)
- SALMONN: ~28GB (FP16)
- Total sequential: OK (models load/unload one at a time)

### Decisions Made
- Qwen-Omni excluded (78GB+ too large for A100 40GB)
- SALMONN runs FP16 (~28GB), not 8-bit quantized
- Sequential model loading to prevent OOM
- Zero-shot evaluation first (no personas/shots yet)

---

## Parent Project
Original Gemini evaluation code: `C:\Users\Raymond\Documents\School\UCSD\Courses\2025 Fall\GenAI\AcousticLLMEval`

## Role Structure
- User: Project owner
- Claude: CEO coordinating research-orchestrator agent and specialized team
- Goal: Generalize Gemini pipeline to support open-weights models on Colab A100
