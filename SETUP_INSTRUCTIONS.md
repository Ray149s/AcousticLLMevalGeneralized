# Environment Setup Instructions

## Prerequisites
- Lambda H100 (or GPU with 80GB+ VRAM)
- Python 3.10+
- NatureLM-audio cloned at `~/NatureLM-audio`

## Quick Setup (Tested on Lambda H100, Nov 28 2025)

```bash
cd ~/AcousticLLMevalGeneralized

# 1. Create virtual environment
python3 -m venv eval_env
source eval_env/bin/activate

# 2. Upgrade pip
pip install --upgrade pip

# 3. Install PyTorch stack (CUDA 12.x)
pip install torch torchvision torchaudio

# 4. Install core ML packages
pip install transformers accelerate

# 5. Install audio processing
pip install librosa soundfile requests

# 6. Install Pillow (needed for transformers image utils)
pip install --upgrade Pillow

# 7. Install NatureLM dependencies
pip install peft einops omegaconf cloudpathlib
pip install google-cloud-storage tensorboardx wandb timm
pip install pydantic-settings pydub resampy
pip install pandas mir-eval levenshtein memoization plumbum tensorboard

# 8. Install NatureLM from local clone (--no-deps to avoid missing packages)
pip install --no-deps -e ~/NatureLM-audio
```

## Key Package Versions (Working)

| Package | Version | Notes |
|---------|---------|-------|
| torch | 2.9.1 | CUDA 12.x |
| transformers | 4.57.3 | Has Qwen2Audio built-in |
| accelerate | 1.12.0 | Required for device_map |
| peft | 0.18.0 | Required for NatureLM LoRA |
| librosa | 0.11.0 | Audio loading |
| soundfile | 0.13.1 | WAV writing |
| Pillow | 12.0.0 | Must be 9.0+ for Resampling |
| torchvision | 0.24.1 | Must match torch version |

## Common Issues & Fixes

### Issue 1: `Qwen2AudioForConditionalGeneration` import fails
**Cause**: Old Pillow version (< 9.0) missing `PIL.Image.Resampling`
**Fix**: `pip install --upgrade Pillow`

### Issue 2: `torchvision::nms does not exist`
**Cause**: torchvision/torch version mismatch
**Fix**: `pip install --upgrade torchvision`

### Issue 3: `numpy.core._multiarray_umath failed to import`
**Cause**: NumPy 2.x incompatibility with system TensorFlow
**Fix**: Use clean virtual environment (not system Python)

### Issue 4: `No module named 'accelerate'`
**Cause**: Missing accelerate package for device_map
**Fix**: `pip install accelerate`

### Issue 5: NatureLM missing dependencies
**Cause**: `pip install naturelm-audio` fails with missing deps
**Fix**: Install deps manually, then `pip install --no-deps -e ~/NatureLM-audio`

### Issue 6: `No module named 'peft'` (NatureLM)
**Fix**: `pip install peft`

### Issue 7: `No module named 'cloudpathlib'` (NatureLM)
**Fix**: `pip install cloudpathlib`

### Issue 8: `No module named 'pydantic_settings'` (NatureLM)
**Fix**: `pip install pydantic-settings`

## Verification Commands

```bash
# Test Qwen2-Audio imports
python -c "from transformers import Qwen2AudioForConditionalGeneration, Qwen2AudioProcessor; print('Qwen2Audio OK')"

# Test NatureLM imports
python -c "from NatureLM.models import NatureLM; from NatureLM.infer import Pipeline; print('NatureLM OK')"

# Run 10-sample test
python run_full_evaluation.py --models qwen naturelm --max-samples 10 --output-dir ./outputs/test
```

## Full Evaluation Command

```bash
# Full 500-sample evaluation (both models, all configs)
python run_full_evaluation.py --models qwen naturelm --output-dir ./outputs/full_eval

# Single model
python run_full_evaluation.py --models qwen --output-dir ./outputs/qwen_full

# Specific configs
python run_full_evaluation.py --models qwen --prompts baseline ornithologist --shots 0 3 --max-samples 100
```

## Expected Output Structure

```
outputs/
├── full_eval/
│   ├── evaluation_manifest.json
│   ├── qwen_baseline_0shot_results.json
│   ├── qwen_baseline_3shot_results.json
│   ├── qwen_baseline_5shot_results.json
│   ├── qwen_ornithologist_0shot_results.json
│   ├── ... (24 files for 2 models × 4 prompts × 3 shots)
│   └── audio_cache/
```
