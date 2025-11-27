"""
AnimalSpeak SPIDEr Benchmark Loader
Loads JSONL dataset for Gemini evaluation
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
import sys
import io

# Configure UTF-8 output for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.experiments.gemini_evaluation import BioacousticExample


def load_animalspeak_jsonl(
    jsonl_path: str,
    max_samples: Optional[int] = None,
    skip_comments: bool = True
) -> List[BioacousticExample]:
    """
    Load AnimalSpeak SPIDEr benchmark from JSONL file

    Args:
        jsonl_path: Path to JSONL file
        max_samples: Maximum number of samples to load (None = all)
        skip_comments: Skip lines starting with '#'

    Returns:
        List of BioacousticExample objects
    """
    print(f"Loading AnimalSpeak dataset from {jsonl_path}...")

    examples = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            # Skip comments
            if skip_comments and line.strip().startswith('#'):
                continue

            # Skip empty lines
            if not line.strip():
                continue

            try:
                data = json.loads(line)

                # Create BioacousticExample
                example = BioacousticExample(
                    audio_path=data['audio_url'],  # Use URL as "path"
                    caption=data['caption'],
                    species=data.get('species_common'),
                    environment=data.get('data_source', 'iNaturalist'),
                    metadata={
                        'id': data.get('id'),
                        'species_scientific': data.get('species_scientific'),
                        'recordist': data.get('recordist'),
                        'source_dataset': data.get('source_dataset'),
                        'subset': data.get('subset'),
                    }
                )
                examples.append(example)

                # Check max samples limit
                if max_samples and len(examples) >= max_samples:
                    break

            except json.JSONDecodeError as e:
                print(f"WARNING: Failed to parse line {line_num}: {e}")
                continue
            except KeyError as e:
                print(f"WARNING: Missing required field on line {line_num}: {e}")
                continue

    print(f"✓ Loaded {len(examples)} samples")

    # Print statistics
    print(f"\nDataset Statistics:")
    print(f"  Total samples: {len(examples)}")

    # Count species
    species_counts = {}
    for ex in examples:
        if ex.species:
            species_counts[ex.species] = species_counts.get(ex.species, 0) + 1

    if species_counts:
        print(f"  Unique species: {len(species_counts)}")
        print(f"\n  Top 10 species:")
        for species, count in sorted(species_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"    - {species}: {count}")

    # Print caption length statistics
    caption_lengths = [len(ex.caption.split()) for ex in examples]
    if caption_lengths:
        print(f"\n  Caption statistics:")
        print(f"    - Mean length: {sum(caption_lengths) / len(caption_lengths):.1f} words")
        print(f"    - Min length: {min(caption_lengths)} words")
        print(f"    - Max length: {max(caption_lengths)} words")

    return examples


def split_train_test(
    examples: List[BioacousticExample],
    train_size: int = 10,
    seed: int = 42
) -> tuple[List[BioacousticExample], List[BioacousticExample]]:
    """
    Split examples into training (for in-context learning) and test sets

    Args:
        examples: List of BioacousticExample objects
        train_size: Number of examples for training
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_examples, test_examples)
    """
    if len(examples) <= train_size:
        print(f"WARNING: Not enough samples to split. Using all {len(examples)} for training.")
        return examples, []

    # Set random seed for reproducibility
    import random
    random.seed(seed)

    # Shuffle examples
    shuffled = examples.copy()
    random.shuffle(shuffled)

    # Split
    train_examples = shuffled[:train_size]
    test_examples = shuffled[train_size:]

    print(f"\nData Split (seed={seed}):")
    print(f"  Training examples: {len(train_examples)}")
    print(f"  Test examples: {len(test_examples)}")

    return train_examples, test_examples


if __name__ == "__main__":
    # Demo usage
    import argparse

    parser = argparse.ArgumentParser(description="Load AnimalSpeak dataset")
    parser.add_argument(
        '--jsonl',
        type=str,
        default='evaluations/gemini/data/processed/animalspeak_spider_benchmark.jsonl',
        help='Path to JSONL file'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum samples to load'
    )

    args = parser.parse_args()

    # Load dataset
    examples = load_animalspeak_jsonl(args.jsonl, max_samples=args.max_samples)

    # Show first few examples
    print("\nFirst 3 examples:")
    for i, ex in enumerate(examples[:3], 1):
        print(f"\n{i}. {ex.species or 'Unknown'}")
        print(f"   Caption: {ex.caption}")
        print(f"   Audio URL: {ex.audio_path}")
        if ex.metadata:
            print(f"   Scientific name: {ex.metadata.get('species_scientific')}")
