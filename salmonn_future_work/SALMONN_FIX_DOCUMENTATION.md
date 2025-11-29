# SALMONN Garbled Output Fix - Technical Documentation

## Problem Statement

SALMONN model loads successfully but produces garbage tokens instead of coherent audio captions:

```
Input: Squirrel Treefrog audio
Output: "\n\n\n```\n\n\n\n\n\n..."

Input: Northern Mockingbird audio
Output: "</Speak>\n</Speak>\n</England</Speak>..."
```

## Root Cause Analysis

### Primary Cause: Library Version Mismatch

SALMONN was trained with specific library versions that have since undergone breaking changes:

| Package | Required Version | Current (Breaking) | Issue |
|---------|-----------------|-------------------|-------|
| transformers | 4.28.0 | 4.40+ | API changes, tokenizer drift |
| peft | 0.3.0 | 0.10+ | Complete API restructuring |
| accelerate | 0.20.3 | 0.25+ | Device map changes |
| bitsandbytes | 0.35.0 | 0.41+ | Quantization API changes |

**Critical Breaking Changes:**

1. **PEFT 0.3.0 to 0.10+**: The LoRA adapter loading mechanism was completely rewritten. Checkpoints saved with peft 0.3.0 cannot be loaded correctly with newer versions - the weights appear as zeros.

2. **Transformers 4.28.0 to 4.40+**:
   - `apply_chunking_to_forward` moved from `transformers.modeling_utils` to `transformers.pytorch_utils`
   - Tokenizer vocabulary may differ slightly
   - Generation API signature changes

### Secondary Cause: Incorrect API Usage

The original wrapper attempted to use a non-existent API pattern:

```python
# WRONG - This API does not exist in SALMONN
self.model = SALMONN.from_pretrained("tsinghua-ee/SALMONN", ...)
response = self.model.generate(audio_path=audio_path, prompt=prompt, ...)
```

The correct pattern from SALMONN's official code:

```python
# CORRECT - Official inference pattern
model = SALMONN.from_config(cfg.config.model)
model.load_state_dict(torch.load(ckpt_path)['model'], strict=False)

samples = prepare_one_sample(wav_path, wav_processor)
prompt = prompt_template.format("<Speech><SpeechHere></Speech> " + user_prompt)

with torch.cuda.amp.autocast(dtype=torch.float16):
    response = model.generate(samples, generate_cfg, prompts=[prompt])[0]
```

### Tertiary Cause: Missing Prompt Formatting

SALMONN requires prompts to be formatted with special speech tokens:

```
USER: <Speech><SpeechHere></Speech> {user_prompt}
ASSISTANT:
```

Without the `<Speech><SpeechHere></Speech>` token, the model does not know where to inject the audio embeddings.

## Solution

### Option 1: Version Pinning (Recommended)

Create a separate virtual environment with exact versions:

```bash
python -m venv salmonn_env
source salmonn_env/bin/activate
pip install -r requirements_salmonn.txt
```

Contents of `requirements_salmonn.txt`:
```
torch==2.0.1
torchaudio==2.0.2
transformers==4.28.0
peft==0.3.0
accelerate==0.20.3
bitsandbytes==0.35.0
sentencepiece==0.1.97
soundfile
librosa
openai-whisper
omegaconf
huggingface_hub
```

### Option 2: Fixed Wrapper Implementation

The fixed wrapper (`salmonn_wrapper_fixed.py`) includes:

1. **Correct API usage**: Uses `from_config()` instead of `from_pretrained()`
2. **Proper audio preprocessing**: Implements `prepare_one_sample()` matching official code
3. **Correct prompt formatting**: Wraps prompts with `<Speech><SpeechHere></Speech>`
4. **Version warnings**: Alerts when incompatible versions are detected
5. **LoRA verification**: Checks if LoRA weights loaded correctly

### Option 3: Monkey-Patching for Newer Versions

If you cannot downgrade versions, apply these patches:

```python
# Fix Qformer.py import
# In SALMONN/models/Qformer.py, change:
from transformers.modeling_utils import apply_chunking_to_forward
# To:
from transformers.pytorch_utils import apply_chunking_to_forward
```

**WARNING**: Monkey-patching does not fix the LoRA weight loading issue. You will still get garbled output unless using peft==0.3.0.

## Verification Procedure

Run the verification script:

```bash
python test_salmonn_fix.py
```

Expected output for **working** model:
```
[OK] transformers: 4.28.0
[OK] peft: 0.3.0
[OK] LoRA weights verified non-zero
[OK] Response appears coherent
[SUCCESS] SALMONN fix verified
```

Expected output for **broken** model:
```
[WARNING] transformers==4.40.0 detected
[WARNING] peft==0.12.0 detected
[CRITICAL] LoRA parameters are all zeros
[FAIL] Response appears to be garbage
```

### Manual Verification

Check LoRA weights after loading:

```python
# After model.load_model()
for name, param in model.model.named_parameters():
    if 'lora' in name.lower():
        print(f"{name}: mean={param.abs().mean():.6f}")
        # Should be > 0. If all zeros, LoRA not loaded.
```

## Files Created

| File | Purpose |
|------|---------|
| `salmonn_wrapper_fixed.py` | Fixed SALMONN wrapper implementation |
| `test_salmonn_fix.py` | Verification test script |
| `requirements_salmonn.txt` | Exact dependency versions |
| `SALMONN_FIX_DOCUMENTATION.md` | This documentation |

## Lambda H100 Setup Instructions

```bash
# 1. SSH to Lambda
ssh ubuntu@192.222.52.233

# 2. Create virtual environment
python -m venv ~/salmonn_env
source ~/salmonn_env/bin/activate

# 3. Install dependencies
pip install -r requirements_salmonn.txt

# 4. Set HuggingFace token
export HF_TOKEN="your_token_here"

# 5. Run verification
python test_salmonn_fix.py
```

## Architecture Reference

```
SALMONN Pipeline:

Audio File (WAV) ─┬──> Whisper Encoder ──────────────────────┐
                  │    (1024-dim features)                   │
                  │                                          ▼
                  └──> BEATs Encoder ──> Q-Former ──> Speech Embeddings
                       (768-dim)         (cross-attention)   (4096-dim)
                                                             │
                                                             ▼
                                                      ┌──────────────┐
Prompt ──> Tokenizer ──> Token Embeddings ──────────> │  Vicuna-13B  │
           (LLaMA)       (4096-dim)                   │  + LoRA      │
                                                      └──────┬───────┘
                                                             │
                                                             ▼
                                                      Text Response
```

## Known Issues

1. **First inference is slow**: Model needs to JIT compile CUDA kernels
2. **~28GB VRAM required**: Use 8-bit mode for 16GB
3. **30 second audio limit**: SALMONN truncates longer audio
4. **Whisper dependency**: Requires large Whisper model download (~3GB)

## References

- Official repo: https://github.com/bytedance/SALMONN (branch: salmonn)
- HuggingFace: https://huggingface.co/tsinghua-ee/SALMONN
- Paper: https://arxiv.org/abs/2310.13289
- Requirements: https://github.com/bytedance/SALMONN/blob/salmonn/requirements.txt
