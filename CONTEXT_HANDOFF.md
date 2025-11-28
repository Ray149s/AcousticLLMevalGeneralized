# Context Handoff Summary

## Project Status: AcousticLLMevalGeneralized — UPDATED

**GitHub Repo:** https://github.com/Ray149s/AcousticLLMevalGeneralized

## What Was Built
A generalized bioacoustic LLM evaluation framework for Google Colab A100, supporting:
- **NatureLM** (~10GB, bfloat16)
- **SALMONN** (~28GB, FP16)
- **Qwen2-Audio-7B** (~14GB, BF16)

## Key Files (13 total)
- `base_model.py` — Abstract interface
- `naturelm_wrapper.py`, `salmonn_wrapper.py`, `qwen_wrapper.py` — Model wrappers
- `universal_evaluator.py` — Model-agnostic evaluation with checkpoint/resume
- `model_registry.py` — Centralized model management
- `colab_orchestrator.ipynb` — Main notebook (FIXED: NatureLM config, added openai-whisper)
- `animalspeak_spider_benchmark.jsonl` — 500-sample dataset

## Proof of Concept Results (10 samples)
| Model | Status | Notes |
|-------|--------|-------|
| Qwen-Audio | SUCCESS | 10/10 samples, 6.52s avg latency |
| NatureLM | FIXED | Config now uses empty wrapper_args |
| SALMONN | FIXED | openai-whisper added to dependencies |

## Fixes Applied (2025-11-27)
1. **NatureLM**: Changed `wrapper_args` from `{"model_path": "..."}` to `{}`
2. **SALMONN**: Added `pip install -q openai-whisper` to Cell 2
3. **flash-attn**: Made non-blocking with fallback message

## Next Steps for User
1. Push updated notebook to GitHub: `git add . && git commit && git push`
2. In Colab: `!git pull` to get the fixes
3. Re-run from Cell 2 (dependencies) then Cell 7 (evaluation)

## Parent Project
Original Gemini evaluation code is in: `C:\Users\Raymond\Documents\School\UCSD\Courses\2025 Fall\GenAI\AcousticLLMEval`
