# Context Handoff Summary

## Project Status: AcousticLLMevalGeneralized — FULL EVALUATION COMPLETE

**GitHub Repo:** https://github.com/Ray149s/AcousticLLMevalGeneralized

**Last Updated:** 2025-11-29 (Full evaluation completed)

---

## What Was Built
A generalized bioacoustic LLM evaluation framework supporting:
- **NatureLM** (~10GB, bfloat16) - Earth Species Project ✅ FULL EVAL COMPLETE
- **SALMONN** (~29GB, FP16) - ByteDance/Tsinghua ⚠️ FUTURE WORK
- **Qwen2-Audio-7B** (~14GB, BF16) - Alibaba Qwen ✅ FULL EVAL COMPLETE

## Key Files
| File | Purpose |
|------|---------|
| `base_model.py` | Abstract interface all wrappers inherit from |
| `naturelm_wrapper.py` | NatureLM implementation (FIXED) |
| `salmonn_wrapper.py` | SALMONN implementation (marked as future work) |
| `qwen_wrapper.py` | Qwen2-Audio-7B implementation (FIXED) |
| `universal_evaluator.py` | Model-agnostic evaluation with checkpoint/resume |
| `model_registry.py` | Centralized model management |
| `colab_orchestrator.ipynb` | Main Colab notebook (UPDATED with all fixes) |
| `animalspeak_spider_benchmark.jsonl` | 500-sample benchmark dataset |
| `prompt_config.py` | 4 prompt roles + 3 shot configs (matches Gemini) |
| `run_full_evaluation.py` | Full 24-config evaluation runner |
| `SETUP_INSTRUCTIONS.md` | Lambda H100 setup with all dependency fixes |
| `requirements_eval_env.txt` | Frozen requirements from working environment |

---

## FULL EVALUATION RESULTS (Completed 2025-11-29)

### Summary
| Metric | Value |
|--------|-------|
| **Total Configurations** | 24 (2 models × 4 prompts × 3 shots) |
| **Samples per Config** | 500 |
| **Total Predictions** | 24,000 |
| **Success Rate** | 100% (all 500/500) |
| **Runtime** | ~4 hours on Lambda H100 |

### Results by Model

**Qwen2-Audio-7B-Instruct (12 configs):**
| Config | Samples | Avg Latency |
|--------|---------|-------------|
| qwen_baseline_0shot | 500/500 | 0.74s |
| qwen_baseline_3shot | 500/500 | 0.97s |
| qwen_baseline_5shot | 500/500 | 1.50s |
| qwen_ornithologist_0shot | 500/500 | 0.97s |
| qwen_ornithologist_3shot | 500/500 | 0.97s |
| qwen_ornithologist_5shot | 500/500 | 1.50s |
| qwen_skeptical_0shot | 500/500 | 1.36s |
| qwen_skeptical_3shot | 500/500 | 0.63s |
| qwen_skeptical_5shot | 500/500 | 0.57s |
| qwen_multi-taxa_0shot | 500/500 | 1.37s |
| qwen_multi-taxa_3shot | 500/500 | 1.49s |
| qwen_multi-taxa_5shot | 500/500 | 2.06s |

**NatureLM-audio (12 configs):**
| Config | Samples | Avg Latency |
|--------|---------|-------------|
| naturelm_baseline_0shot | 500/500 | 0.97s |
| naturelm_baseline_3shot | 500/500 | 1.23s |
| naturelm_baseline_5shot | 500/500 | 1.28s |
| naturelm_ornithologist_0shot | 500/500 | 1.43s |
| naturelm_ornithologist_3shot | 500/500 | 1.32s |
| naturelm_ornithologist_5shot | 500/500 | 1.38s |
| naturelm_skeptical_0shot | 500/500 | 1.06s |
| naturelm_skeptical_3shot | 500/500 | 1.26s |
| naturelm_skeptical_5shot | 500/500 | 1.35s |
| naturelm_multi-taxa_0shot | 500/500 | 1.29s |
| naturelm_multi-taxa_3shot | 500/500 | 1.38s |
| naturelm_multi-taxa_5shot | 500/500 | 1.43s |

### Results Location
**Local:** `C:\Users\Raymond\Documents\School\UCSD\Courses\2025 Fall\GenAI\AcousticLLMevalGeneralized\outputs\lambda_full_eval\`

**Files:**
- 24 result JSON files (one per config)
- 1 evaluation_manifest.json

---

## Colab Notebook Fixes (2025-11-29)

### Issue 1: Pillow Version
**Error:** `cannot import name '_Ink' from 'PIL._typing'`
**Fix:** Step 2 now installs `Pillow>=10.0.0` and prompts for runtime restart

### Issue 2: NatureLM transformers Compatibility
**Error:** `cannot import name 'apply_chunking_to_forward' from 'transformers.modeling_utils'`
**Root Cause:** transformers>=4.40 moved functions from `modeling_utils` to `pytorch_utils`

**Fix:** Step 3 now patches NatureLM's Qformer.py:
```python
# Functions moved to pytorch_utils:
- apply_chunking_to_forward
- find_pruneable_heads_and_indices
- prune_linear_layer
```

### Colab Workflow (Working)
1. Step 1 - GPU check, mount Drive
2. Step 2 - Install deps → **RESTART RUNTIME**
3. Step 3 - Clone repos + auto-patch NatureLM
4. Step 4 - HuggingFace auth
5. Step 5 - Verify imports (both should pass)
6. Step 6 - Configure evaluation
7. Step 7 - Run evaluation

---

## Testing Infrastructure

### Lambda Labs H100 (Used for full evaluation)
- **SSH:** `ssh ubuntu@192.222.52.233` (instance may be terminated)
- **Virtual env:** `source ~/AcousticLLMevalGeneralized/eval_env/bin/activate`
- **Results:** `~/AcousticLLMevalGeneralized/outputs/full_eval/`

### Environment Setup (Lambda)
```bash
cd ~/AcousticLLMevalGeneralized
python3 -m venv eval_env
source eval_env/bin/activate
pip install -r requirements_eval_env.txt
pip install --no-deps -e ~/NatureLM-audio
```

---

## Fixes Applied

### 1. Qwen-Audio Wrapper (CRITICAL FIX)
**Problem:** Original wrapper used wrong API - processor ignored `audios` parameter

**Fix:** Complete rewrite using conversation format
```python
conversation = [{"role": "user", "content": [
    {"type": "audio", "audio": audio},
    {"type": "text", "text": prompt}
]}]
text = self.processor.apply_chat_template(conversation, ...)
inputs = self.processor(text=text, audios=[audio], ...)
```

### 2. NatureLM Wrapper (CRITICAL FIX)
**Problem:** Device/dtype mismatch + transformers API changes

**Fix:**
- Explicit device and dtype handling
- Patch Qformer.py imports for transformers>=4.40

### 3. SALMONN (FUTURE WORK)
Root cause identified (version mismatch) but requires separate environment.
See `SALMONN_FIX_DOCUMENTATION.md` for details.

---

## VRAM Requirements

| Model | VRAM | Device Tested |
|-------|------|---------------|
| Qwen2-Audio | ~14GB | Lambda H100 80GB |
| NatureLM | ~10GB | Lambda H100 80GB |
| SALMONN | ~29GB | Lambda H100 80GB |

---

## Next Steps

### P2: Compute SPIDEr Scores (PENDING)
- Use `compute_spider_colab.ipynb` or local script
- Requires SPICE/METEOR dependencies (Java)
- Compare against Gemini Pro/Flash baselines

### P3: Generate Final Report (PENDING)
- Publication-quality figures
- Statistical analysis of model performance
- Comparison tables across all configurations

---

## Parent Project
Original Gemini evaluation: `C:\Users\Raymond\Documents\School\UCSD\Courses\2025 Fall\GenAI\AcousticLLMEval`

## Project Goal
Generalize Gemini evaluation pipeline to open-weights models (Qwen, NatureLM, SALMONN)

---

## Session History

| Date | Accomplishments |
|------|-----------------|
| 2025-11-27 | Initial framework, Colab setup, identified model issues |
| 2025-11-28 | Lambda H100 testing, fixed Qwen & NatureLM, SALMONN diagnosed |
| 2025-11-28 | Prompt roles & shots implemented (P0.5 complete) |
| 2025-11-29 | **FULL EVALUATION COMPLETE** - 24,000 predictions, results downloaded |

## Execution Plan Status

| Priority | Task | Status |
|----------|------|--------|
| P0 | SALMONN Fix | FUTURE WORK |
| P0.5 | Prompt Roles & Shots | ✅ COMPLETE |
| P1 | Full Evaluation (500 samples × 24 configs) | ✅ COMPLETE |
| P2 | SPIDEr Score Computation | PENDING |
| P3 | Final Report & Figures | PENDING |

---

## Quick Reference

### Run Full Evaluation (Lambda)
```bash
source ~/AcousticLLMevalGeneralized/eval_env/bin/activate
cd ~/AcousticLLMevalGeneralized
python run_full_evaluation.py --models qwen naturelm --output-dir ./outputs/full_eval
```

### Download Results to Local
```bash
scp -r ubuntu@192.222.52.233:~/AcousticLLMevalGeneralized/outputs/full_eval/* ./outputs/lambda_full_eval/
```

### Colab Quick Start
1. Open `colab_orchestrator.ipynb` in Google Colab
2. Set runtime to A100 GPU
3. Run all cells in order (restart after Step 2)
