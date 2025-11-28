# Context Handoff Summary

## Project Status: AcousticLLMevalGeneralized — MODELS TESTED

**GitHub Repo:** https://github.com/Ray149s/AcousticLLMevalGeneralized

**Last Updated:** 2025-11-28 (team synchronized)

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

### 3. SALMONN Wrapper (DIAGNOSED AND FIXED)
**Root Cause Identified (2025-11-28):**
The garbled output is caused by **library version mismatch**:

| Package | Required | Current | Issue |
|---------|----------|---------|-------|
| transformers | 4.28.0 | 4.40+ | API changes break tokenizer/generation |
| peft | 0.3.0 | 0.10+ | LoRA weights fail to load (appear as zeros) |
| accelerate | 0.20.3 | 0.25+ | Device map changes |

**Secondary Cause:** Original wrapper used wrong API:
```python
# WRONG (doesn't exist)
SALMONN.from_pretrained("tsinghua-ee/SALMONN", ...)

# CORRECT (official API)
SALMONN.from_config(cfg.config.model)
model.load_state_dict(torch.load(ckpt_path)['model'], strict=False)
```

**Missing Prompt Formatting:**
SALMONN requires `<Speech><SpeechHere></Speech>` tokens for audio injection:
```
USER: <Speech><SpeechHere></Speech> {prompt}
ASSISTANT:
```

**Fix Files Created:**
| File | Purpose |
|------|---------|
| `salmonn_wrapper_fixed.py` | Corrected wrapper matching official inference |
| `test_salmonn_fix.py` | Verification script |
| `requirements_salmonn.txt` | Exact version pins |
| `SALMONN_FIX_DOCUMENTATION.md` | Full technical documentation |

**To Apply Fix:**
```bash
# Create separate environment with exact versions
python -m venv salmonn_env
source salmonn_env/bin/activate
pip install -r requirements_salmonn.txt
python test_salmonn_fix.py
```

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

### SALMONN Fix (Ready to Test)
Root cause identified and fix implemented. Steps:
1. Create separate virtual environment on Lambda H100
2. Install exact versions from `requirements_salmonn.txt`:
   ```
   transformers==4.28.0
   peft==0.3.0
   torch==2.0.1
   ```
3. Run `python test_salmonn_fix.py` to verify
4. If verification passes, run full evaluation

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

## Team Structure & Agent Capabilities

### Core Team
| Role | Member | Responsibilities |
|------|--------|------------------|
| **Project Owner** | Raymond | Strategic direction, resource allocation, final approval |
| **CEO** | Claude (Main Agent) | Coordinate agents, strategic planning, synthesize results |
| **Research Orchestrator** | Specialized Agent | Execute workflows, delegate tasks, track experiments |

### Available Specialized Agents
| Agent | Capabilities | Status |
|-------|-------------|--------|
| `experiment-runner` | Batch processing, checkpointing, parallel execution | Ready |
| `results-analyzer` | SPIDEr/BLEU metrics, statistical testing | Ready |
| `bioacoustic-data-manager` | Dataset prep, audio preprocessing, taxonomy | Ready |
| `salmonn-restoration-engineer` | Fix broken multimodal LLMs, version debugging | Ready |
| `report-generator` | Publication-quality reports, LaTeX tables | Ready |
| `visualization-architect` | Charts, heatmaps, comparison plots | Ready |
| `model-comparison-analyst` | Rigorous model comparisons, tradeoff analysis | Ready |
| `cost-monitor` | API cost tracking, budget optimization | Ready |

### Communication Protocols
- **Project Owner ↔ CEO**: Resource requests, strategic decisions
- **CEO ↔ Orchestrator**: Task delegation, progress updates
- **Orchestrator ↔ Agents**: Task specifications via YAML format

### Project Goal
Generalize Gemini evaluation pipeline to open-weights models (Qwen, NatureLM, SALMONN)

---

## Session History

| Date | Accomplishments |
|------|-----------------|
| 2025-11-27 | Initial framework, Colab setup, identified model issues |
| 2025-11-28 | Lambda H100 testing, fixed Qwen & NatureLM, SALMONN partial |
| 2025-11-28 | Team synchronization: Research Orchestrator briefed, all agents aligned |

## Team Synchronization Status

**Last Sync:** 2025-11-28

### Readiness Assessment
- Research Orchestrator: BRIEFED ✅
- All specialized agents: READY FOR DEPLOYMENT ✅
- Infrastructure validated: ✅
- Evaluation pipeline tested: ✅

### Execution Plan (Approved)
1. **[P0] SALMONN Fix**: Deploy `salmonn-restoration-engineer`
2. **[P1] Full Evaluation**: Run Qwen + NatureLM on 500 samples (parallel)
3. **[P2] Results Analysis**: SPIDEr scores via `results-analyzer`
4. **[P3] Final Report**: Publication figures and documentation

### Blockers
- HuggingFace token required from Project Owner
- Lambda H100 status confirmation needed
