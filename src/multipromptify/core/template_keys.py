"""
Template keys for MultiPromptify templates.

Each key is used in template dictionaries to specify a particular field or configuration.
These constants should be used instead of hardcoded strings for clarity and consistency.

- SYSTEM_PROMPT_TEMPLATE_KEY: The system prompt shown at the top of each prompt (optional, can include placeholders).
- SYSTEM_PROMPT_KEY: List of variation types to apply to the system prompt (e.g., [REWORDING]).
- INSTRUCTION_TEMPLATE_KEY: The main instruction template for each example (usually contains placeholders for fields).
- INSTRUCTION_KEY: List of variation types to apply to the instruction template (e.g., [PARAPHRASE_WITH_LLM]).
"""

# Template keys
INSTRUCTION = "instruction"  # System prompt at the top of each prompt (optional)
INSTRUCTION_VARIATIONS = "instruction variations"  # Variation types for the system prompt
PROMPT_FORMAT = "prompt format"  # Main instruction template for each example
PROMPT_FORMAT_VARIATIONS = "prompt format variations"  # Variation types for the instruction template
QUESTION_KEY = "question"
GOLD_KEY = "gold"
FEW_SHOT_KEY = "few_shot"
OPTIONS_KEY = "options"
CONTEXT_KEY = "context"
PROBLEM_KEY = "problem"
GOLD_FIELD = "gold"

# Variation types (values)
PARAPHRASE_WITH_LLM = "paraphrase_with_llm"  # replaces 'paraphrase'
CONTEXT_VARIATION = "context"
SHUFFLE_VARIATION = "shuffle"
MULTIDOC_VARIATION = "multidoc"
ENUMERATE_VARIATION = "enumerate"
FEW_SHOT_VARIATION = "fewshot"

# New specialized augmenters (refactored from TextSurfaceAugmenter)
FORMAT_STRUCTURE_VARIATION = "format structure"  # For semantic-preserving format changes
TYPOS_AND_NOISE_VARIATION = "typos and noise"  # For noise injection (typos, swaps, etc.)

