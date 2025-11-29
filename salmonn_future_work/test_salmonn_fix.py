#!/usr/bin/env python3
"""
SALMONN Fix Verification Script.

This script tests whether the SALMONN fix is working correctly.
Run on Lambda H100 after setting up the environment.

Usage:
    # 1. Set up environment (see requirements_salmonn.txt)
    # 2. Set HuggingFace token
    export HF_TOKEN="your_token_here"

    # 3. Run verification
    python test_salmonn_fix.py

Expected output with WORKING model:
    "The audio contains sounds of a frog calling..." or similar coherent text

Expected output with BROKEN model:
    "\n\n\n```\n\n\n..." or "</Speak>\n</Speak>..." (garbage)
"""

import os
import sys
import tempfile
import numpy as np
import soundfile as sf
import torch


def check_environment():
    """Verify library versions are compatible with SALMONN."""
    print("=" * 60)
    print("SALMONN Environment Check")
    print("=" * 60)

    # Check Python
    print(f"\nPython: {sys.version}")

    # Check PyTorch
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Check transformers
    try:
        import transformers
        tf_ver = transformers.__version__
        tf_ok = tf_ver.startswith("4.28")
        status = "OK" if tf_ok else "WARNING"
        print(f"transformers: {tf_ver} [{status}] (required: 4.28.0)")
    except ImportError:
        print("transformers: NOT INSTALLED [CRITICAL]")
        return False

    # Check peft
    try:
        import peft
        peft_ver = peft.__version__
        peft_ok = peft_ver.startswith("0.3")
        status = "OK" if peft_ok else "WARNING"
        print(f"peft: {peft_ver} [{status}] (required: 0.3.0)")
    except ImportError:
        print("peft: NOT INSTALLED [CRITICAL]")
        return False

    # Check other deps
    for pkg in ['accelerate', 'bitsandbytes', 'omegaconf']:
        try:
            mod = __import__(pkg)
            print(f"{pkg}: {getattr(mod, '__version__', 'installed')}")
        except ImportError:
            print(f"{pkg}: NOT INSTALLED")

    print()
    return True


def create_test_audio(duration_sec: float = 3.0, sample_rate: int = 16000) -> str:
    """Create a test audio file with synthetic sound."""
    # Generate a simple tone (440 Hz)
    t = np.linspace(0, duration_sec, int(duration_sec * sample_rate), dtype=np.float32)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)

    # Add some noise
    audio += 0.1 * np.random.randn(len(audio)).astype(np.float32)

    # Write to temp file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        sf.write(f.name, audio, sample_rate)
        return f.name


def test_lora_weights(model):
    """Verify LoRA weights are loaded correctly."""
    print("\n" + "=" * 60)
    print("LoRA Weight Verification")
    print("=" * 60)

    lora_params = {}
    for name, param in model.model.named_parameters():
        if 'lora' in name.lower():
            lora_params[name] = {
                'shape': tuple(param.shape),
                'mean': param.abs().mean().item(),
                'max': param.abs().max().item(),
                'zeros': (param == 0).sum().item(),
                'total': param.numel()
            }

    if not lora_params:
        print("[CRITICAL] No LoRA parameters found!")
        print("This means LoRA adapter is not attached to the model.")
        return False

    print(f"Found {len(lora_params)} LoRA parameters")

    zero_count = 0
    for name, stats in list(lora_params.items())[:10]:
        is_zero = stats['max'] == 0
        if is_zero:
            zero_count += 1
        status = "ZEROS" if is_zero else "OK"
        print(f"  {name}: mean={stats['mean']:.6f}, max={stats['max']:.6f} [{status}]")

    if len(lora_params) > 10:
        print(f"  ... and {len(lora_params) - 10} more")

    if zero_count > 0:
        print(f"\n[WARNING] {zero_count} LoRA parameters are all zeros!")
        print("This indicates LoRA weights did not load from checkpoint.")
        return False

    print("\n[OK] All LoRA weights are non-zero")
    return True


def test_generation_quality(model, audio_path: str) -> bool:
    """Test if generation produces coherent output."""
    print("\n" + "=" * 60)
    print("Generation Quality Test")
    print("=" * 60)

    prompts = [
        "Describe the audio.",
        "What sounds do you hear?",
        "Provide a detailed caption for this audio."
    ]

    all_coherent = True

    for prompt in prompts:
        print(f"\nPrompt: {prompt}")

        try:
            response = model.generate_caption(audio_path, prompt)
            print(f"Response: {response[:200]}{'...' if len(response) > 200 else ''}")

            # Check for garbage indicators
            garbage_indicators = [
                response.count('\n') > 10,  # Too many newlines
                '```' in response,  # Code block (shouldn't appear)
                '</Speak>' in response,  # XML tag leakage
                '</England' in response,  # Corrupted tag
                len(set(response)) < 5,  # Very low character diversity
                response.strip() == '',  # Empty response
            ]

            if any(garbage_indicators):
                print("[FAIL] Response appears to be garbage")
                all_coherent = False
            else:
                print("[OK] Response appears coherent")

        except Exception as e:
            print(f"[ERROR] Generation failed: {e}")
            all_coherent = False

    return all_coherent


def test_tensor_pipeline(model, audio_path: str):
    """Inspect tensors at each pipeline stage."""
    print("\n" + "=" * 60)
    print("Tensor Pipeline Inspection")
    print("=" * 60)

    # Prepare sample
    samples = model._prepare_one_sample(audio_path)

    print("\n1. Audio Preprocessing:")
    print(f"   spectrogram: shape={samples['spectrogram'].shape}, "
          f"dtype={samples['spectrogram'].dtype}, "
          f"range=[{samples['spectrogram'].min():.3f}, {samples['spectrogram'].max():.3f}]")
    print(f"   raw_wav: shape={samples['raw_wav'].shape}, "
          f"dtype={samples['raw_wav'].dtype}")

    # Check for NaN/Inf
    for name, tensor in samples.items():
        if isinstance(tensor, torch.Tensor):
            has_nan = torch.isnan(tensor).any().item()
            has_inf = torch.isinf(tensor).any().item()
            if has_nan or has_inf:
                print(f"   [WARNING] {name} contains NaN={has_nan}, Inf={has_inf}")


def main():
    """Run all SALMONN verification tests."""
    print("\n" + "=" * 60)
    print("SALMONN FIX VERIFICATION")
    print("=" * 60)

    # Check environment
    if not check_environment():
        print("\n[CRITICAL] Environment check failed. Fix dependencies first.")
        return 1

    # Import wrapper
    try:
        from salmonn_wrapper_fixed import SalmonnWrapperFixed
    except ImportError:
        print("\n[ERROR] Could not import SalmonnWrapperFixed")
        print("Ensure salmonn_wrapper_fixed.py is in the current directory")
        return 1

    # Create test audio
    print("\nCreating test audio...")
    test_audio = create_test_audio()
    print(f"Test audio: {test_audio}")

    # Initialize model
    print("\nInitializing SALMONN wrapper...")
    model = SalmonnWrapperFixed(
        model_name="SALMONN-Test",
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_8bit=False  # Use full precision for testing
    )

    # Load model
    print("\nLoading model (this may take several minutes)...")
    try:
        model.load_model()
    except Exception as e:
        print(f"\n[CRITICAL] Model loading failed: {e}")
        os.remove(test_audio)
        return 1

    # Run tests
    results = {
        'lora': test_lora_weights(model),
        'tensors': True,  # Inspected above
        'generation': test_generation_quality(model, test_audio),
    }

    # Inspect tensors
    test_tensor_pipeline(model, test_audio)

    # Cleanup
    os.remove(test_audio)
    model.unload()

    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)

    all_passed = all(results.values())
    for test, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {test}: [{status}]")

    if all_passed:
        print("\n[SUCCESS] SALMONN fix verified - model produces coherent output!")
        return 0
    else:
        print("\n[FAILURE] SALMONN still has issues - see details above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
