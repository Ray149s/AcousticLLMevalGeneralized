#!/usr/bin/env python3
"""
Full Generalized Bioacoustic Evaluation.

Runs all configurations (models × prompts × shots) matching Gemini evaluation:
- Models: Qwen2-Audio, NatureLM (SALMONN = future work)
- Prompts: baseline, ornithologist, skeptical, multi-taxa
- Shots: 0-shot, 3-shot, 5-shot

Total: 2 models × 4 prompts × 3 shots = 24 configurations
"""

import os
import sys
import json
import time
import argparse
import tempfile
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

import requests
import librosa
import soundfile as sf

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from prompt_config import (
    build_prompt,
    get_shot_examples,
    get_all_configurations,
    FewShotExample,
    PROMPT_VERSIONS,
    SHOT_CONFIGS,
)


# =============================================================================
# AUDIO UTILITIES
# =============================================================================

def download_audio(url: str, cache_dir: Optional[Path] = None) -> str:
    """Download audio from URL and convert to 16kHz WAV."""
    # Check cache first
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_key = url.split('/')[-1].split('?')[0]
        cache_path = cache_dir / f"{cache_key}.wav"
        if cache_path.exists():
            return str(cache_path)

    # Download
    response = requests.get(url, timeout=60)
    response.raise_for_status()

    # Determine format
    if '.flac' in url:
        ext = '.flac'
    elif '.mp3' in url or '.mpga' in url:
        ext = '.mp3'
    elif '.ogg' in url:
        ext = '.ogg'
    else:
        ext = '.wav'

    # Save raw audio
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as raw_file:
        raw_file.write(response.content)
        raw_path = raw_file.name

    # Convert to 16kHz WAV
    audio, sr = librosa.load(raw_path, sr=16000)

    if cache_dir:
        wav_path = str(cache_path)
    else:
        wav_path = raw_path.replace(ext, '.wav') if ext != '.wav' else raw_path + '.converted.wav'

    sf.write(wav_path, audio, 16000)

    # Cleanup raw file
    if raw_path != wav_path and os.path.exists(raw_path):
        os.remove(raw_path)

    return wav_path


# =============================================================================
# MODEL WRAPPER INTERFACE
# =============================================================================

def get_model_wrapper(model_name: str):
    """Get the appropriate model wrapper."""
    if model_name == 'qwen':
        from qwen_wrapper import QwenAudioWrapper
        return QwenAudioWrapper(model_name="Qwen2-Audio-7B")
    elif model_name == 'naturelm':
        from naturelm_wrapper import NatureLMWrapper
        return NatureLMWrapper(model_name="NatureLM-audio")
    else:
        raise ValueError(f"Unknown model: {model_name}")


# =============================================================================
# EVALUATION RUNNER
# =============================================================================

def load_benchmark(jsonl_path: str, max_samples: Optional[int] = None) -> List[Dict]:
    """Load benchmark samples from JSONL."""
    samples = []
    with open(jsonl_path) as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                samples.append(json.loads(line))

    if max_samples:
        samples = samples[:max_samples]

    return samples


def load_few_shot_examples(jsonl_path: str, n_examples: int = 10) -> List[FewShotExample]:
    """Load few-shot examples from beginning of benchmark."""
    examples = []
    with open(jsonl_path) as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                data = json.loads(line)
                examples.append(FewShotExample(
                    caption=data.get('caption', ''),
                    species=data.get('species_common', ''),
                    environment=data.get('data_source', 'Unknown'),
                ))
                if len(examples) >= n_examples:
                    break
    return examples


def run_single_config(
    model,
    samples: List[Dict],
    prompt_version: str,
    n_shots: int,
    few_shot_pool: List[FewShotExample],
    audio_cache_dir: Path,
) -> Dict:
    """Run evaluation for a single configuration."""
    config_name = f"{prompt_version}_{n_shots}shot"
    print(f"\n  Running: {config_name}")

    # Get few-shot examples
    shot_examples = get_shot_examples(few_shot_pool, n_shots)

    # Build prompt
    prompt = build_prompt(
        prompt_version=prompt_version,
        examples=shot_examples,
    )

    results = []
    latencies = []

    for i, sample in enumerate(samples):
        species = sample.get('species_common', 'Unknown')
        print(f"    [{i+1}/{len(samples)}] {species}", end='', flush=True)

        try:
            # Download audio
            audio_path = download_audio(sample['audio_url'], audio_cache_dir)

            # Generate caption
            start_time = time.time()
            prediction = model.generate_caption(audio_path, prompt)
            latency = time.time() - start_time
            latencies.append(latency)

            results.append({
                'id': sample.get('id'),
                'species': species,
                'reference': sample.get('caption', ''),
                'prediction': prediction,
                'latency': latency,
                'success': True,
            })

            print(f" - {latency:.2f}s")

        except Exception as e:
            print(f" - ERROR: {e}")
            results.append({
                'id': sample.get('id'),
                'species': species,
                'reference': sample.get('caption', ''),
                'prediction': '',
                'error': str(e),
                'success': False,
            })

    # Calculate stats
    successful = [r for r in results if r.get('success')]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0

    return {
        'config_name': config_name,
        'prompt_version': prompt_version,
        'n_shots': n_shots,
        'samples_tested': len(results),
        'successful': len(successful),
        'avg_latency': avg_latency,
        'results': results,
    }


def run_full_evaluation(
    models: List[str] = ['qwen', 'naturelm'],
    prompt_versions: List[str] = None,
    shot_configs: List[int] = None,
    jsonl_path: str = 'animalspeak_spider_benchmark.jsonl',
    max_samples: Optional[int] = None,
    output_dir: str = './outputs/generalized_eval',
    few_shot_pool_size: int = 10,
):
    """
    Run full evaluation across all configurations.

    Args:
        models: List of models to test
        prompt_versions: List of prompt versions (default: all 4)
        shot_configs: List of shot configs (default: [0, 3, 5])
        jsonl_path: Path to benchmark JSONL
        max_samples: Maximum test samples (None = all)
        output_dir: Output directory for results
        few_shot_pool_size: Number of examples for few-shot pool
    """
    prompt_versions = prompt_versions or PROMPT_VERSIONS
    shot_configs = shot_configs or SHOT_CONFIGS

    print("=" * 80)
    print("GENERALIZED BIOACOUSTIC EVALUATION")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Models: {models}")
    print(f"  Prompt versions: {prompt_versions}")
    print(f"  Shot configs: {shot_configs}")
    print(f"  Max samples: {max_samples or 'all'}")

    total_configs = len(models) * len(prompt_versions) * len(shot_configs)
    print(f"  Total configurations: {total_configs}")

    # Setup output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    audio_cache_dir = output_path / 'audio_cache'
    audio_cache_dir.mkdir(exist_ok=True)

    # Load benchmark
    print(f"\n[Step 1] Loading benchmark...")
    samples = load_benchmark(jsonl_path, max_samples)
    few_shot_pool = load_few_shot_examples(jsonl_path, few_shot_pool_size)

    print(f"  Test samples: {len(samples)}")
    print(f"  Few-shot pool: {len(few_shot_pool)} examples")

    # Run evaluations
    print(f"\n[Step 2] Running evaluations...")
    print("=" * 80)

    all_results = {}
    completed = 0
    start_time = time.time()

    for model_name in models:
        print(f"\n{'=' * 40}")
        print(f"MODEL: {model_name.upper()}")
        print("=" * 40)

        # Initialize model
        print(f"  Loading {model_name}...")
        try:
            model = get_model_wrapper(model_name)
            model.load_model()
        except Exception as e:
            print(f"  ERROR loading model: {e}")
            continue

        model_results = {}

        for prompt_version in prompt_versions:
            for n_shots in shot_configs:
                completed += 1
                config_name = f"{model_name}_{prompt_version}_{n_shots}shot"

                print(f"\n[{completed}/{total_configs}] {config_name}")

                result = run_single_config(
                    model=model,
                    samples=samples,
                    prompt_version=prompt_version,
                    n_shots=n_shots,
                    few_shot_pool=few_shot_pool,
                    audio_cache_dir=audio_cache_dir,
                )

                model_results[config_name] = result

                # Save individual result
                result_file = output_path / f"{config_name}_results.json"
                with open(result_file, 'w') as f:
                    json.dump({
                        'model': model_name,
                        'prompt_version': prompt_version,
                        'n_shots': n_shots,
                        'timestamp': datetime.now().isoformat(),
                        **result,
                    }, f, indent=2)

                print(f"    Saved: {result_file.name}")

        # Unload model to free memory
        model.unload()
        all_results[model_name] = model_results

    # Summary
    elapsed_time = time.time() - start_time
    print(f"\n{'=' * 80}")
    print("EVALUATION COMPLETE")
    print(f"{'=' * 80}")
    print(f"Total time: {elapsed_time / 60:.1f} minutes")
    print(f"Configurations completed: {completed}/{total_configs}")

    # Save manifest
    manifest = {
        'timestamp': datetime.now().isoformat(),
        'models': models,
        'prompt_versions': prompt_versions,
        'shot_configs': shot_configs,
        'samples_per_config': len(samples),
        'total_configs': completed,
        'total_time_seconds': elapsed_time,
    }

    manifest_path = output_path / 'evaluation_manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\nManifest saved: {manifest_path}")

    # Print summary table
    print(f"\n{'=' * 80}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 80}")
    print(f"{'Config':<40} {'Success':>10} {'Avg Latency':>12}")
    print("-" * 62)

    for model_name, model_results in all_results.items():
        for config_name, result in model_results.items():
            success_rate = f"{result['successful']}/{result['samples_tested']}"
            avg_lat = f"{result['avg_latency']:.2f}s"
            print(f"{config_name:<40} {success_rate:>10} {avg_lat:>12}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Run generalized bioacoustic evaluation")

    parser.add_argument('--models', nargs='+', default=['qwen', 'naturelm'],
                        choices=['qwen', 'naturelm'],
                        help='Models to evaluate')
    parser.add_argument('--prompts', nargs='+', default=None,
                        choices=['baseline', 'ornithologist', 'skeptical', 'multi-taxa'],
                        help='Prompt versions (default: all)')
    parser.add_argument('--shots', nargs='+', type=int, default=None,
                        help='Shot configs (default: 0 3 5)')
    parser.add_argument('--jsonl', type=str, default='animalspeak_spider_benchmark.jsonl',
                        help='Path to benchmark JSONL')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Max test samples (default: all)')
    parser.add_argument('--output-dir', type=str, default='./outputs/generalized_eval',
                        help='Output directory')

    args = parser.parse_args()

    run_full_evaluation(
        models=args.models,
        prompt_versions=args.prompts,
        shot_configs=args.shots,
        jsonl_path=args.jsonl,
        max_samples=args.max_samples,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
