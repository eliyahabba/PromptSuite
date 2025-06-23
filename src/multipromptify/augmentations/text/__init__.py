"""
Text-based augmentation modules.
"""

from multipromptify.augmentations.text.context import ContextAugmenter
from multipromptify.augmentations.text.paraphrase import Paraphrase
from .format_structure import FormatStructureAugmenter
from .noise import TextNoiseAugmenter

__all__ = [
    "Paraphrase",
    "ContextAugmenter",
    "FormatStructureAugmenter",  # New semantic-preserving format augmenter
    "TextNoiseAugmenter"  # New noise injection augmenter
]
