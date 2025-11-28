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
| `salmonn_wrapper.py` | SALMONN implementation |
| `qwen_wrapper.py` | Qwen2-Audio-7B implementation |
| `universal_evaluator.py` | Model-agnostic evaluation with checkpoint/resume |
| `model_registry.py` | Centralized model management |
| `colab_orchestrator.ipynb` | Main Colab notebook (ALL FIXES APPLIED) |
| `animalspeak_spider_benchmark.jsonl` | 500-sample benchmark dataset |

---

## Current Evaluation Status (10-sample proof of concept)

| Model | Status | Notes |
|-------|--------|-------|
| **Qwen-Audio** | ✅ SUCCESS | 10/10 samples, 6.52s avg latency |
| **NatureLM** | 🔧 READY TO TEST | All fixes applied to GitHub |
| **SALMONN** | 🔧 READY TO TEST | All fixes applied to GitHub |

---

## ALL FIXES NOW IN GITHUB

The notebook and wrapper files have ALL fixes applied:

1. **NatureLM Config** (colab_orchestrator.ipynb Cell 4/Config)
   - `wrapper_args: {}` (empty, no model_path)

2. **SALMONN Import Path** (salmonn_wrapper.py line 249)
   - `from models.salmonn import SALMONN`

3. **SALMONN Branch** (colab_orchestrator.ipynb Cell 3/Clone)
   - Auto-checkouts `salmonn` branch instead of `main`

4. **Dependencies** (colab_orchestrator.ipynb Cell 2)
   - Added `pip install -q openai-whisper`
   - Made `flash-attn` non-blocking

5. **GitHub Username** (colab_orchestrator.ipynb Cell 3)
   - Changed `YOUR_USERNAME` to `Ray149s`

---

## RESUME INSTRUCTIONS FOR TOMORROW

### Option A: Fresh Colab (Recommended)
1. Open new Colab notebook
2. Change runtime to A100 GPU
3. Run all cells in order (no manual fixes needed)

### Option B: Existing Colab Session
1. Pull the latest fixes:
```python
!cd /content/AcousticLLMevalGeneralized && git pull
```
2. Restart runtime: Runtime → Restart runtime
3. Re-run from Cell 1

---

## Technical Context

### SALMONN Repository Structure
```
/content/SALMONN/  (branch: salmonn)
├── models/
│   ├── salmonn.py      <-- SALMONN class
│   ├── modeling_llama.py
│   ├── modeling_whisper.py
│   └── Qformer.py
├── configs/
└── ...
```

### VRAM Requirements (A100 40GB)
- NatureLM: ~10GB (bfloat16)
- Qwen2-Audio: ~14GB (BF16)
- SALMONN: ~28GB (FP16)
- Sequential loading prevents OOM

### Key Decisions
- Qwen-Omni excluded (78GB+ too large)
- SALMONN runs FP16, not 8-bit quantized
- Zero-shot evaluation first (no personas/shots yet)

---

## Parent Project
Original Gemini evaluation: `C:\Users\Raymond\Documents\School\UCSD\Courses\2025 Fall\GenAI\AcousticLLMEval`

## Role Structure
- User: Project owner
- Claude: CEO coordinating research-orchestrator agent
- Goal: Generalize Gemini pipeline to open-weights models on Colab A100
