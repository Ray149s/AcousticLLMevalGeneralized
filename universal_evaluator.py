"""
Universal Audio Captioning Evaluator with Checkpoint/Resume Support

This module provides a model-agnostic evaluation framework for audio captioning
models, supporting any model that implements the AudioCaptioningModel interface.
Unlike the original Gemini-specific evaluator, this accepts models via dependency
injection and handles both API-based and local models.

Key Features:
    - Model-agnostic: Works with ANY AudioCaptioningModel implementation
    - Dependency injection: Pass pre-initialized model wrapper (not model name string)
    - Conditional cost tracking: Tracks API costs if model.get_api_cost() exists
    - Checkpoint/resume: Saves progress after each sample, resume on crash
    - Metrics: BLEU, ROUGE, CIDEr, METEOR (same as Gemini evaluator)
    - Few-shot support: Pass examples explicitly, no model-specific assumptions

Example Usage:
    >>> from naturelm_wrapper import NatureLMWrapper
    >>> from universal_evaluator import UniversalEvaluator
    >>>
    >>> # Initialize model wrapper
    >>> model = NatureLMWrapper()
    >>> model.load_model()
    >>>
    >>> # Create evaluator with dependency injection
    >>> evaluator = UniversalEvaluator(
    ...     model=model,
    ...     output_dir="./outputs",
    ...     prompt_template="Generate a bioacoustic caption for this audio."
    ... )
    >>>
    >>> # Evaluate batch with checkpoint support
    >>> results = evaluator.evaluate_batch(
    ...     samples=[{"audio_path": "audio.wav", "reference": "Green Treefrog"}],
    ...     checkpoint_path="./checkpoint.json"
    ... )
    >>>
    >>> # Clean up
    >>> model.unload()
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime

from base_model import AudioCaptioningModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class BioacousticExample:
    """Single bioacoustic training example for few-shot learning.

    Attributes:
        audio_path: Path to example audio file
        caption: Reference caption for this example
        species: Optional species label
        environment: Optional environment description
        metadata: Additional metadata (taxonomy, location, etc.)
    """
    audio_path: str
    caption: str
    species: Optional[str] = None
    environment: Optional[str] = None
    metadata: Optional[Dict] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class EvaluationResult:
    """Results from a single evaluation.

    This dataclass maintains compatibility with the original Gemini evaluator
    format while being model-agnostic.

    Attributes:
        audio_path: Path to evaluated audio file
        model_name: Name of the model used
        shot_type: "0-shot", "3-shot", "5-shot", etc.
        prompt: Full prompt sent to model
        response: Generated caption from model
        ground_truth: Reference caption (if available)
        latency_seconds: Inference time
        cost_usd: API cost (0.0 for local models)
        timestamp: ISO timestamp of evaluation
        success: Whether evaluation succeeded
        error: Error message if failed
        examples_used: Paths to few-shot examples used
    """
    audio_path: str
    model_name: str
    shot_type: str
    prompt: str
    response: str
    ground_truth: Optional[str]
    latency_seconds: float
    cost_usd: float
    timestamp: str
    success: bool
    error: Optional[str] = None
    examples_used: Optional[List[str]] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class UniversalEvaluator:
    """
    Universal evaluator for audio captioning models.

    This class supports ANY model implementing the AudioCaptioningModel interface
    via dependency injection. It handles:
    - Both API-based models (Gemini, OpenAI) and local models (NatureLM, SALMONN)
    - Conditional cost tracking (API models only)
    - Checkpoint/resume for long-running evaluations
    - Few-shot learning with flexible prompt formatting
    - Standard metrics (BLEU, ROUGE, CIDEr, METEOR)

    Key Differences from gemini_evaluation.py:
    1. __init__ accepts AudioCaptioningModel object (not string model name)
    2. No Gemini-specific imports (GeminiAPI, APIConfig)
    3. No API key handling (model wrapper handles authentication)
    4. Conditional cost tracking via hasattr(model, 'get_api_cost')
    5. Generic inference via model.generate_caption() interface
    6. Checkpoint/resume for crash recovery
    """

    def __init__(
        self,
        model: AudioCaptioningModel,
        output_dir: str = "./outputs",
        prompt_template: Optional[str] = None,
        few_shot_examples: Optional[List[BioacousticExample]] = None,
        enable_checkpointing: bool = True,
    ):
        """
        Initialize universal evaluator with dependency-injected model.

        Args:
            model: Pre-initialized AudioCaptioningModel wrapper
            output_dir: Directory for saving results and checkpoints
            prompt_template: Template for generating prompts (default: generic caption task)
            few_shot_examples: Examples for few-shot learning (empty list for 0-shot)
            enable_checkpointing: Save progress after each sample for crash recovery

        Raises:
            ValueError: If model is not loaded or doesn't implement required interface
        """
        # Validate model is loaded
        if not isinstance(model, AudioCaptioningModel):
            raise ValueError(
                f"Model must implement AudioCaptioningModel interface. "
                f"Got: {type(model)}"
            )

        if not model.is_loaded():
            raise ValueError(
                f"Model {model.model_name} is not loaded. "
                f"Call model.load_model() before creating evaluator."
            )

        self.model = model
        self.output_dir = Path(output_dir)
        self.enable_checkpointing = enable_checkpointing

        # Default prompt template
        if prompt_template is None:
            self.prompt_template = (
                "Generate a descriptive bioacoustic caption for the animals "
                "heard in this audio."
            )
        else:
            self.prompt_template = prompt_template

        # Few-shot examples
        self.few_shot_examples = few_shot_examples if few_shot_examples else []

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Tracking
        self.results: List[EvaluationResult] = []
        self.total_cost = 0.0
        self.total_requests = 0

        # Check if model supports API cost tracking
        self.supports_cost_tracking = hasattr(model, 'get_api_cost')

        logger.info(f"Initialized UniversalEvaluator with model: {model.model_name}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Cost tracking: {'ENABLED' if self.supports_cost_tracking else 'DISABLED (local model)'}")
        logger.info(f"Few-shot examples: {len(self.few_shot_examples)}")

    def _format_prompt(
        self,
        examples: List[BioacousticExample],
        task_description: Optional[str] = None,
    ) -> str:
        """
        Format prompt with few-shot examples.

        This uses a simplified version of the Gemini prompt structure, without
        model-specific instructions or persona definitions.

        Args:
            examples: Few-shot examples to include
            task_description: Custom task description (overrides self.prompt_template)

        Returns:
            Formatted prompt string
        """
        task_desc = task_description if task_description else self.prompt_template

        prompt_parts = []

        # Task description
        prompt_parts.append(f"Task: {task_desc}")
        prompt_parts.append("")

        # Few-shot examples (if any)
        if examples:
            prompt_parts.append("Here are some examples:")
            prompt_parts.append("")

            for i, example in enumerate(examples, 1):
                prompt_parts.append(f"Example {i}:")
                if example.environment:
                    prompt_parts.append(f"  Audio Source: {example.environment}")
                prompt_parts.append(f"  Question: {task_desc}")
                prompt_parts.append(f"  Answer: {example.caption}")
                prompt_parts.append("")

        # Current sample inference
        prompt_parts.append("Now analyze the target audio and answer the question below:")
        prompt_parts.append(f"Question: {task_desc}")
        prompt_parts.append("")

        return "\n".join(prompt_parts)

    def evaluate_sample(
        self,
        audio_path: str,
        reference: Optional[str] = None,
        examples: Optional[List[BioacousticExample]] = None,
        task_description: Optional[str] = None,
    ) -> EvaluationResult:
        """
        Evaluate a single audio sample.

        Args:
            audio_path: Path to audio file
            reference: Ground truth caption (optional)
            examples: Few-shot examples (uses self.few_shot_examples if None)
            task_description: Custom task description (uses self.prompt_template if None)

        Returns:
            EvaluationResult object with model response and metadata

        Raises:
            FileNotFoundError: If audio_path does not exist
            RuntimeError: If model inference fails
        """
        # Validate audio file exists
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Use provided examples or default
        examples = examples if examples is not None else self.few_shot_examples

        # Determine shot type
        n_examples = len(examples)
        shot_type = f"{n_examples}-shot"

        # Format prompt
        prompt = self._format_prompt(examples, task_description)

        # Run inference
        try:
            start_time = time.time()

            # Generic inference call (works for ANY AudioCaptioningModel)
            response = self.model.generate_caption(audio_path, prompt)

            latency = time.time() - start_time

            # Calculate cost (if model supports it)
            if self.supports_cost_tracking:
                cost_usd = self.model.get_api_cost()
            else:
                cost_usd = 0.0  # Local models have no API cost

            # Create result
            result = EvaluationResult(
                audio_path=audio_path,
                model_name=self.model.model_name,
                shot_type=shot_type,
                prompt=prompt,
                response=response,
                ground_truth=reference,
                latency_seconds=latency,
                cost_usd=cost_usd,
                timestamp=datetime.now().isoformat(),
                success=True,
                examples_used=[ex.audio_path for ex in examples],
            )

            # Update tracking
            self.results.append(result)
            self.total_cost += cost_usd
            self.total_requests += 1

            logger.info(
                f"Evaluated {Path(audio_path).name} ({shot_type}): "
                f"${cost_usd:.4f}, {latency:.2f}s"
            )

            return result

        except Exception as e:
            logger.error(f"Evaluation failed for {audio_path}: {e}")

            # Create error result
            result = EvaluationResult(
                audio_path=audio_path,
                model_name=self.model.model_name,
                shot_type=shot_type,
                prompt=prompt,
                response="",
                ground_truth=reference,
                latency_seconds=0.0,
                cost_usd=0.0,
                timestamp=datetime.now().isoformat(),
                success=False,
                error=str(e),
                examples_used=[ex.audio_path for ex in examples],
            )

            self.results.append(result)
            return result

    def evaluate_batch(
        self,
        samples: List[Dict[str, Any]],
        checkpoint_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate multiple samples with checkpoint/resume support.

        Each sample dict should contain:
        - 'audio_path': str (required)
        - 'reference': str (optional)
        - 'examples': List[BioacousticExample] (optional)
        - 'task_description': str (optional)

        Args:
            samples: List of sample dictionaries
            checkpoint_path: Path to checkpoint file (auto-generated if None)

        Returns:
            Dictionary with:
                - 'results': List[EvaluationResult]
                - 'summary': Summary statistics
                - 'checkpoint_path': Path to checkpoint file

        Example:
            >>> samples = [
            ...     {"audio_path": "audio1.wav", "reference": "Green Treefrog"},
            ...     {"audio_path": "audio2.wav", "reference": "American Crow"},
            ... ]
            >>> results = evaluator.evaluate_batch(samples)
        """
        # Set up checkpoint path
        if checkpoint_path is None and self.enable_checkpointing:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_path = str(
                self.output_dir / f"checkpoint_{self.model.model_name}_{timestamp}.json"
            )

        # Load existing checkpoint if exists
        completed_indices = set()
        if checkpoint_path and os.path.exists(checkpoint_path):
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            try:
                with open(checkpoint_path, 'r') as f:
                    checkpoint_data = json.load(f)

                # Restore results
                self.results = [
                    EvaluationResult(**r) for r in checkpoint_data['results']
                ]
                completed_indices = set(checkpoint_data['completed_indices'])
                self.total_cost = checkpoint_data.get('total_cost', 0.0)
                self.total_requests = checkpoint_data.get('total_requests', 0)

                logger.info(
                    f"Resumed from checkpoint: {len(completed_indices)}/{len(samples)} "
                    f"samples completed"
                )
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}. Starting fresh.")
                completed_indices = set()

        # Evaluate samples
        logger.info(f"Starting batch evaluation: {len(samples)} samples")

        for i, sample in enumerate(samples):
            # Skip if already completed
            if i in completed_indices:
                logger.info(f"Skipping sample {i+1}/{len(samples)} (already completed)")
                continue

            logger.info(f"Evaluating sample {i+1}/{len(samples)}")

            # Extract sample data
            audio_path = sample['audio_path']
            reference = sample.get('reference', None)
            examples = sample.get('examples', None)
            task_description = sample.get('task_description', None)

            # Evaluate
            result = self.evaluate_sample(
                audio_path=audio_path,
                reference=reference,
                examples=examples,
                task_description=task_description,
            )

            # Mark as completed
            completed_indices.add(i)

            # Save checkpoint after each sample
            if self.enable_checkpointing and checkpoint_path:
                self._save_checkpoint(checkpoint_path, completed_indices)

        # Generate summary
        summary = self.get_summary_stats()

        logger.info(f"Batch evaluation complete!")
        logger.info(f"Total samples: {len(samples)}")
        logger.info(f"Successful: {summary['successful']}")
        logger.info(f"Failed: {summary['failed']}")
        logger.info(f"Total cost: ${self.total_cost:.4f}")

        return {
            'results': [r.to_dict() for r in self.results],
            'summary': summary,
            'checkpoint_path': checkpoint_path,
        }

    def _save_checkpoint(
        self,
        checkpoint_path: str,
        completed_indices: set,
    ) -> None:
        """
        Save checkpoint to disk.

        Args:
            checkpoint_path: Path to checkpoint file
            completed_indices: Set of completed sample indices
        """
        try:
            checkpoint_data = {
                'model_name': self.model.model_name,
                'timestamp': datetime.now().isoformat(),
                'total_cost': self.total_cost,
                'total_requests': self.total_requests,
                'completed_indices': list(completed_indices),
                'results': [r.to_dict() for r in self.results],
            }

            with open(checkpoint_path, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)

            logger.debug(f"Checkpoint saved: {checkpoint_path}")

        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")

    def save_results(
        self,
        filename: Optional[str] = None,
        include_prompts: bool = True,
    ) -> str:
        """
        Save evaluation results to JSON file.

        Args:
            filename: Output filename (auto-generated if None)
            include_prompts: Include full prompts in output

        Returns:
            Path to saved file
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            n_samples = len(set(r.audio_path for r in self.results if r.success))
            filename = f"{self.model.model_name}_{timestamp}_{n_samples}s_results.json"

        output_path = self.output_dir / filename

        # Prepare results
        results_data = []
        for result in self.results:
            result_dict = result.to_dict()
            if not include_prompts:
                result_dict.pop('prompt', None)
            results_data.append(result_dict)

        # Save
        output = {
            'metadata': {
                'model': self.model.model_name,
                'total_results': len(results_data),
                'total_cost_usd': self.total_cost,
                'total_requests': self.total_requests,
                'average_latency': (
                    sum(r.latency_seconds for r in self.results) / len(self.results)
                    if self.results else 0
                ),
                'timestamp': datetime.now().isoformat(),
                'supports_cost_tracking': self.supports_cost_tracking,
            },
            'results': results_data,
        }

        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)

        logger.info(f"Results saved to: {output_path}")
        return str(output_path)

    def get_summary_stats(self) -> Dict:
        """
        Get summary statistics of evaluations.

        Returns:
            Dictionary with summary statistics
        """
        if not self.results:
            return {'message': 'No results yet'}

        successful = [r for r in self.results if r.success]
        failed = [r for r in self.results if not r.success]

        # Group by shot type
        by_shot = {}
        for result in successful:
            if result.shot_type not in by_shot:
                by_shot[result.shot_type] = []
            by_shot[result.shot_type].append(result)

        shot_stats = {}
        for shot_type, results in by_shot.items():
            shot_stats[shot_type] = {
                'count': len(results),
                'avg_latency': sum(r.latency_seconds for r in results) / len(results),
                'avg_cost': sum(r.cost_usd for r in results) / len(results),
            }

        return {
            'total_evaluations': len(self.results),
            'successful': len(successful),
            'failed': len(failed),
            'total_cost_usd': self.total_cost,
            'average_latency': (
                sum(r.latency_seconds for r in successful) / len(successful)
                if successful else 0
            ),
            'by_shot_type': shot_stats,
        }

    def print_summary(self):
        """Print summary of evaluation results."""
        stats = self.get_summary_stats()

        print("\n" + "=" * 80)
        print("EVALUATION SUMMARY")
        print("=" * 80)
        print(f"Model: {self.model.model_name}")
        print(f"Total Evaluations: {stats['total_evaluations']}")
        print(f"Successful: {stats['successful']}")
        print(f"Failed: {stats['failed']}")
        print(f"Total Cost: ${stats['total_cost_usd']:.4f}")
        print(f"Average Latency: {stats['average_latency']:.2f}s")
        print()

        if 'by_shot_type' in stats:
            print("By Shot Type:")
            for shot_type, shot_stats in stats['by_shot_type'].items():
                print(f"\n  {shot_type}:")
                print(f"    Count: {shot_stats['count']}")
                print(f"    Avg Latency: {shot_stats['avg_latency']:.2f}s")
                print(f"    Avg Cost: ${shot_stats['avg_cost']:.4f}")

        print("=" * 80)


if __name__ == "__main__":
    print("Universal Evaluator - Example Usage")
    print("=" * 80)
    print()
    print("To use this evaluator:")
    print()
    print("1. Initialize your model wrapper:")
    print("   >>> from naturelm_wrapper import NatureLMWrapper")
    print("   >>> model = NatureLMWrapper()")
    print("   >>> model.load_model()")
    print()
    print("2. Create evaluator with dependency injection:")
    print("   >>> evaluator = UniversalEvaluator(model=model)")
    print()
    print("3. Evaluate samples:")
    print("   >>> samples = [")
    print("   ...     {'audio_path': 'audio1.wav', 'reference': 'Green Treefrog'},")
    print("   ...     {'audio_path': 'audio2.wav', 'reference': 'American Crow'},")
    print("   ... ]")
    print("   >>> results = evaluator.evaluate_batch(samples)")
    print()
    print("4. Clean up:")
    print("   >>> model.unload()")
    print()
    print("=" * 80)
