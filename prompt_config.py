"""
Prompt Configuration for Generalized Bioacoustic Evaluation.

Ports the 4 prompt roles from Gemini evaluation for fair comparison:
- baseline: Generic bioacoustic captioning expert
- ornithologist: Dr. Elena Vasquez - world-renowned ornithologist
- skeptical: Cautious expert avoiding overconfident species naming
- multi-taxa: Expert across birds, frogs, insects, mammals

Shot configurations:
- 0-shot: No examples, just prompt + audio
- 3-shot: 3 text-only in-context examples
- 5-shot: 5 text-only in-context examples
"""

from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class FewShotExample:
    """Single few-shot example (text-only for open-weights models)."""
    caption: str
    species: Optional[str] = None
    environment: Optional[str] = None


# =============================================================================
# EXPERT PERSONAS (matching Gemini evaluation exactly)
# =============================================================================

EXPERT_PERSONAS = {
    'baseline': [
        "You are an expert in bioacoustic captioning.",
        "",
    ],

    'ornithologist': [
        "You are Dr. Elena Vasquez, a world-renowned ornithologist and bioacoustic captioning specialist.",
        "You excel at generating precise, concise captions of bird vocalizations.",
        "",
        "Your expertise includes:",
        "- Identifying call structure, rhythm, and frequency patterns",
        "- Describing bird sounds when exact species is uncertain",
        "- Interpreting complex soundscapes with scientific clarity",
        "",
    ],

    'skeptical': [
        "You are a cautious bioacoustic captioning expert.",
        "You avoid overconfident species naming and focus on accurate sound descriptions.",
        "",
        "Your philosophy:",
        "\"Describe what you hear clearly and conservatively. Only identify species when confident.\"",
        "",
        "Your strengths:",
        "- Distinguishing clear vocalizations from background noise",
        "- Avoiding false positives while still generating meaningful captions",
        "- Providing scientifically grounded descriptions",
        "",
    ],

    'multi-taxa': [
        "You are an expert bioacoustic captioning specialist experienced with birds, frogs, insects, mammals, and environmental soundscapes.",
        "You focus on producing informative captions that describe animal vocalizations and acoustic scenes.",
        "",
        "Your expertise includes:",
        "- Recognizing multiple overlapping taxa",
        "- Describing complex multi-species audio",
        "- Providing natural-language captions suited for scientific analysis",
        "",
    ],
}


# =============================================================================
# RESPONSE INSTRUCTIONS (matching Gemini evaluation exactly)
# =============================================================================

RESPONSE_INSTRUCTIONS = {
    'baseline': [
        "CAPTIONING INSTRUCTIONS:",
        "- Provide a clear, concise caption describing the audible animals or sounds.",
        "- Keep the caption brief (20 words or fewer).",
        "- Only describe what is clearly audible.",
        "",
    ],

    'ornithologist': [
        "CAPTIONING INSTRUCTIONS:",
        "- Produce a brief, precise caption describing the bird vocalizations.",
        "- Identify bird species when acoustic features are distinct.",
        "- When uncertain, describe the bird sound characteristics rather than forcing a species name.",
        "- Keep the caption scientific, clear, and under ~20 words.",
        "",
    ],

    'skeptical': [
        "CAPTIONING INSTRUCTIONS:",
        "- Produce a concise, factual caption describing the animals heard.",
        "- Identify species ONLY when vocalizations are clear and characteristic.",
        "- When uncertain, describe the sound generically (e.g., 'unidentified songbird', 'unknown frog').",
        "- Do NOT guess species names.",
        "- Avoid adding details not supported by audio cues.",
        "",
    ],

    'multi-taxa': [
        "CAPTIONING INSTRUCTIONS:",
        "- Generate a concise caption describing all animals heard.",
        "- Identify species when confident; otherwise describe the taxonomic group (e.g. 'a frog calling', 'a small songbird').",
        "- Mention multiple species/groups when present.",
        "- If audio is unclear, describe sound characteristics rather than returning 'None'.",
        "",
    ],
}


# =============================================================================
# PROMPT BUILDER
# =============================================================================

def build_prompt(
    prompt_version: str = 'baseline',
    examples: Optional[List[FewShotExample]] = None,
    task_description: str = "Generate a descriptive bioacoustic caption for the animals heard in this audio.",
) -> str:
    """
    Build a complete prompt matching Gemini evaluation structure.

    Args:
        prompt_version: One of 'baseline', 'ornithologist', 'skeptical', 'multi-taxa'
        examples: List of few-shot examples (text-only)
        task_description: The captioning task description

    Returns:
        Complete formatted prompt string
    """
    if prompt_version not in EXPERT_PERSONAS:
        raise ValueError(f"Unknown prompt version: {prompt_version}. "
                        f"Choose from: {list(EXPERT_PERSONAS.keys())}")

    prompt_parts = []

    # 1. Expert persona
    prompt_parts.extend(EXPERT_PERSONAS[prompt_version])

    # 2. Task description
    prompt_parts.append(f"Task: {task_description}")
    prompt_parts.append("")

    # 3. Response instructions
    prompt_parts.extend(RESPONSE_INSTRUCTIONS[prompt_version])

    # 4. Few-shot examples (text-only for open-weights models)
    if examples:
        prompt_parts.append("Here are some examples:")
        prompt_parts.append("")

        for i, example in enumerate(examples, 1):
            prompt_parts.append(f"Example {i}:")
            if example.environment:
                prompt_parts.append(f"  Audio Source: {example.environment}")
            prompt_parts.append(f"  Question: {task_description}")
            prompt_parts.append(f"  Answer: {example.caption}")
            prompt_parts.append("")

    # 5. Current sample inference
    prompt_parts.append("Now analyze the target audio and answer the question below:")
    prompt_parts.append(f"Question: {task_description}")
    prompt_parts.append("")

    return "\n".join(prompt_parts)


def get_shot_examples(
    all_examples: List[FewShotExample],
    n_shots: int,
) -> List[FewShotExample]:
    """
    Select examples for n-shot learning.

    Args:
        all_examples: Pool of available examples
        n_shots: Number of examples to select (0, 3, or 5)

    Returns:
        List of selected examples
    """
    if n_shots == 0:
        return []

    if len(all_examples) < n_shots:
        print(f"Warning: Only {len(all_examples)} examples available for {n_shots}-shot")
        return all_examples

    return all_examples[:n_shots]


# =============================================================================
# CONFIGURATION MATRIX
# =============================================================================

PROMPT_VERSIONS = ['baseline', 'ornithologist', 'skeptical', 'multi-taxa']
SHOT_CONFIGS = [0, 3, 5]


def get_all_configurations() -> List[Dict]:
    """
    Get all prompt × shot configurations for evaluation.

    Returns:
        List of configuration dicts with 'prompt_version' and 'n_shots'
    """
    configs = []
    for prompt_version in PROMPT_VERSIONS:
        for n_shots in SHOT_CONFIGS:
            configs.append({
                'prompt_version': prompt_version,
                'n_shots': n_shots,
                'config_name': f"{prompt_version}_{n_shots}shot",
            })
    return configs


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Demo prompt generation
    print("=" * 80)
    print("PROMPT CONFIGURATION DEMO")
    print("=" * 80)

    # Example few-shot examples
    demo_examples = [
        FewShotExample(
            caption="A male northern cardinal singing territorial song with clear whistles",
            species="Northern Cardinal",
            environment="Deciduous forest edge",
        ),
        FewShotExample(
            caption="American crow calls with harsh cawing sounds in urban setting",
            species="American Crow",
            environment="Urban park",
        ),
        FewShotExample(
            caption="Spring peeper chorus with high-pitched peeping calls overlapping",
            species="Spring Peeper",
            environment="Wetland",
        ),
    ]

    # Show each prompt version with 0-shot
    for version in PROMPT_VERSIONS:
        print(f"\n{'=' * 40}")
        print(f"PROMPT VERSION: {version} (0-shot)")
        print("=" * 40)
        prompt = build_prompt(prompt_version=version, examples=[])
        print(prompt)

    # Show 3-shot example
    print(f"\n{'=' * 40}")
    print("PROMPT VERSION: baseline (3-shot)")
    print("=" * 40)
    prompt = build_prompt(prompt_version='baseline', examples=demo_examples)
    print(prompt)

    # Show all configurations
    print(f"\n{'=' * 40}")
    print("ALL CONFIGURATIONS")
    print("=" * 40)
    for config in get_all_configurations():
        print(f"  {config['config_name']}")
