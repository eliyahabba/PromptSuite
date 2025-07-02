"""
Text-based augmentation modules.
"""

from promptsuite.augmentations.text.context import ContextAugmenter
from promptsuite.augmentations.text.paraphrase import Paraphrase
from .format_structure import FormatStructureAugmenter
from .noise import TextNoiseAugmenter

__all__ = [
    "Paraphrase",
    "ContextAugmenter",
    "FormatStructureAugmenter",  # New semantic-preserving format augmenter
    "TextNoiseAugmenter"  # New noise injection augmenter
]
