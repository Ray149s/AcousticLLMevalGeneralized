# Context Handoff Summary

## Project Status: AcousticLLMevalGeneralized — MODELS TESTED

**GitHub Repo:** https://github.com/Ray149s/AcousticLLMevalGeneralized

**Last Updated:** 2025-11-28 (end of session)

---

## What Was Built
A generalized bioacoustic LLM evaluation framework supporting:
- **NatureLM** (~10GB, bfloat16) - Earth Species Project ✅ WORKING
- **SALMONN** (~29GB, FP16) - ByteDance/Tsinghua ⚠️ NEEDS INVESTIGATION
- **Qwen2-Audio-7B** (~14GB, BF16) - Alibaba Qwen ✅ WORKING

## Key Files
| File | Purpose |
|------|---------|
| `base_model.py` | Abstract interface all wrappers inherit from |
| `naturelm_wrapper.py` | NatureLM implementation (FIXED) |
| `salmonn_wrapper.py` | SALMONN implementation (loads but garbled output) |
| `qwen_wrapper.py` | Qwen2-Audio-7B implementation (FIXED) |
| `universal_evaluator.py` | Model-agnostic evaluation with checkpoint/resume |
| `model_registry.py` | Centralized model management |
| `colab_orchestrator.ipynb` | Main Colab notebook |
| `animalspeak_spider_benchmark.jsonl` | 500-sample benchmark dataset |

---

## Current Evaluation Status (10-sample proof of concept)

| Model | Status | Samples | Avg Latency | Notes |
|-------|--------|---------|-------------|-------|
| **Qwen-Audio** | ✅ WORKING | 10/10 | 0.9s | Correct species ID |
| **NatureLM** | ✅ WORKING | 10/10 | 1.54s | Correct species ID |
| **SALMONN** | ⚠️ GARBLED OUTPUT | 10/10 | 5.39s | Model loads, output corrupted |

### Sample Outputs

**Qwen-Audio (Working):**
```
Input: Squirrel Treefrog audio
Output: "The audio contains the sounds of frogs, specifically tree frogs..."
```

**NatureLM (Working):**
```
Input: Squirrel Treefrog audio
Output: "Squirrel Treefrog"

Input: Alder Flycatcher audio
Output: "The sound of an Alder Flycatcher."
```

**SALMONN (Garbled):**
```
Input: Squirrel Treefrog audio
Output: "\n\n\n```\n\n\n\n\n\n..."

Input: Northern Mockingbird audio
Output: "</Speak>\n</Speak>\n</England</Speak>..."
```

---

## Testing Infrastructure

### Lambda Labs H100 (Used for debugging)
- **SSH:** `ssh ubuntu@192.222.52.233` (instance may be terminated)
- **Virtual env:** `source ~/venv/bin/activate`
- **HF Token:** Set via `export HF_TOKEN="your_token"` (request from user)

### Test Scripts on Lambda
```
~/AcousticLLMevalGeneralized/
├── run_qwen_test.py      # 10-sample Qwen test
├── run_naturelm_test.py  # 10-sample NatureLM test
├── run_salmonn_test.py   # 10-sample SALMONN test
└── *_test_results.json   # Results files
```

---

## Fixes Applied (Nov 28, 2025)

### 1. Qwen-Audio Wrapper (CRITICAL FIX)
**Problem:** Original wrapper used wrong API - processor ignored `audios` parameter

**Fix:** Complete rewrite using conversation format
```python
# OLD (broken)
inputs = self.processor(text=prompt, audios=audio, ...)

# NEW (working)
conversation = [{"role": "user", "content": [
    {"type": "audio", "audio": audio},
    {"type": "text", "text": prompt}
]}]
text = self.processor.apply_chat_template(conversation, ...)
inputs = self.processor(text=text, audios=[audio], ...)
```

Also switched from `Qwen2-Audio-7B` to `Qwen2-Audio-7B-Instruct`.

### 2. NatureLM Wrapper (CRITICAL FIX)
**Problem:** Device/dtype mismatch - BEATs weights on CPU, input on CUDA

**Fix:**
```python
# Added explicit device and dtype handling
self.model = self.model.to(self.device).to(torch.bfloat16)
```

Also patched `Qformer.py` import:
```python
# OLD (broken in new transformers)
from transformers.modeling_utils import apply_chunking_to_forward

# NEW (working)
from transformers.pytorch_utils import apply_chunking_to_forward
```

### 3. SALMONN Wrapper (PARTIAL)
**What works:**
- Downloads BEATs checkpoint from `Bencr/beats-checkpoints` (dataset repo)
- Downloads SALMONN checkpoint from `tsinghua-ee/SALMONN`
- Model loads successfully (~29GB VRAM)
- Patched Qformer.py imports

**What's broken:**
- Model generates garbage tokens (HTML fragments, repeated characters)
- Suspected cause: Incompatibility with newer `transformers`/`peft` versions
- SALMONN was trained with older library versions

---

## VRAM Requirements

| Model | VRAM | Device Tested |
|-------|------|---------------|
| Qwen2-Audio | ~14GB | Lambda H100 80GB |
| NatureLM | ~10GB | Lambda H100 80GB |
| SALMONN | ~29GB | Lambda H100 80GB |

---

## Next Steps

### Immediate (Ready Now)
1. Run full 500-sample evaluation with **Qwen-Audio** and **NatureLM**
2. Use `compute_spider_colab.ipynb` for SPIDEr score calculation
3. Compare against Gemini Pro/Flash baselines from parent project

### SALMONN Investigation (Future)
1. Pin older library versions:
   ```
   transformers==4.36.0
   peft==0.7.0
   ```
2. Verify checkpoint loading with official demo audio
3. Check if LoRA weights are being applied correctly

### Full Evaluation Command (Lambda)
```bash
source ~/venv/bin/activate
cd ~/AcousticLLMevalGeneralized
export HF_TOKEN="your_huggingface_token"

# Edit MAX_SAMPLES in scripts: None for full 500
python run_qwen_test.py
python run_naturelm_test.py
```

---

## Parent Project
Original Gemini evaluation: `C:\Users\Raymond\Documents\School\UCSD\Courses\2025 Fall\GenAI\AcousticLLMEval`

## Role Structure
- User: Project owner
- Claude: CEO coordinating research-orchestrator agent
- Goal: Generalize Gemini pipeline to open-weights models

---

## Session History

| Date | Accomplishments |
|------|-----------------|
| 2025-11-27 | Initial framework, Colab setup, identified model issues |
| 2025-11-28 | Lambda H100 testing, fixed Qwen & NatureLM, SALMONN partial |
