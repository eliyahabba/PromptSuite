"""
Generation package: Contains refactored classes for handling variations, prompts, and few-shot examples.
"""

from .variation_generator import VariationGenerator
from .prompt_builder import PromptBuilder
from .few_shot_handler import FewShotHandler

__all__ = [
    'VariationGenerator',
    'PromptBuilder',
    'FewShotHandler'
] 