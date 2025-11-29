#!/usr/bin/env python3
"""Run SALMONN on benchmark samples to get actual metrics."""

import os
import sys
import json
import time
import tempfile
from pathlib import Path

import torch
import requests
import soundfile as sf
import librosa

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from salmonn_wrapper_fixed import SalmonnWrapperFixed

MAX_SAMPLES = 10
BENCHMARK_FILE = "animalspeak_spider_benchmark.jsonl"

def download_audio(url: str) -> str:
    """Download audio from URL to temp file."""
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    # Determine file extension from URL
    if '.flac' in url:
        ext = '.flac'
    elif '.mp3' in url:
        ext = '.mp3'
    elif '.mpga' in url:
        ext = '.mp3'  # mpga is MPEG audio
    elif '.ogg' in url:
        ext = '.ogg'
    else:
        ext = '.wav'

    # Save raw audio to temp file
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as raw_file:
        raw_file.write(response.content)
        raw_path = raw_file.name

    # Load and resample to 16kHz WAV
    audio, sr = librosa.load(raw_path, sr=16000)

    # Save as WAV
    wav_path = raw_path.replace(ext, '.wav') if ext != '.wav' else raw_path + '.converted.wav'
    sf.write(wav_path, audio, 16000)

    # Cleanup raw file if different
    if raw_path != wav_path:
        os.remove(raw_path)

    return wav_path

def main():
    print("=" * 60)
    print("SALMONN BENCHMARK TEST")
    print("=" * 60)

    # Load benchmark
    benchmark_path = Path(__file__).parent / BENCHMARK_FILE
    samples = []
    with open(benchmark_path) as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                samples.append(json.loads(line))

    print(f"\nLoaded {len(samples)} samples, testing {MAX_SAMPLES}")

    # Initialize model
    print("\nInitializing SALMONN...")
    model = SalmonnWrapperFixed(
        model_name="SALMONN-Benchmark",
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_8bit=False
    )
    model.load_model()

    # Run tests
    results = []
    latencies = []

    for i, sample in enumerate(samples[:MAX_SAMPLES]):
        species = sample.get('species_common', sample.get('species', 'Unknown'))
        print(f"\n[{i+1}/{MAX_SAMPLES}] Processing: {species}")

        try:
            # Download audio
            audio_path = download_audio(sample['audio_url'])

            # Generate caption
            prompt = "Describe the sounds in this audio recording."
            start_time = time.time()
            response = model.generate_caption(audio_path, prompt)
            latency = time.time() - start_time
            latencies.append(latency)

            result = {
                'species': species,
                'reference': sample.get('caption', ''),
                'prediction': response,
                'latency': latency
            }
            results.append(result)

            print(f"  Species: {result['species']}")
            print(f"  Reference: {result['reference'][:100]}...")
            print(f"  Prediction: {result['prediction'][:100]}...")
            print(f"  Latency: {latency:.2f}s")

            # Cleanup
            os.remove(audio_path)

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'species': species,
                'error': str(e)
            })

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    successful = [r for r in results if 'prediction' in r]
    print(f"Successful: {len(successful)}/{len(results)}")

    if latencies:
        avg_latency = sum(latencies) / len(latencies)
        print(f"Average latency: {avg_latency:.2f}s")

    # Save results
    output_file = Path(__file__).parent / "salmonn_benchmark_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            'model': 'SALMONN',
            'samples_tested': len(results),
            'successful': len(successful),
            'avg_latency': avg_latency if latencies else None,
            'results': results
        }, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    # Show sample outputs
    print("\n" + "=" * 60)
    print("SAMPLE OUTPUTS")
    print("=" * 60)
    for r in successful[:3]:
        print(f"\nSpecies: {r['species']}")
        print(f"Prediction: {r['prediction']}")

    # Cleanup
    model.unload()

    return 0

if __name__ == "__main__":
    sys.exit(main())
