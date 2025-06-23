"""
Augmentation modules for MultiPromptify.
"""

# Import all augmenters for easy access
from multipromptify.augmentations.base import BaseAxisAugmenter
# Other augmenters
from multipromptify.augmentations.other import OtherAugmenter
from multipromptify.augmentations.structure.enumerate import EnumeratorAugmenter
# Structure augmenters
from multipromptify.augmentations.structure.fewshot import FewShotAugmenter
from multipromptify.augmentations.structure.shuffle import ShuffleAugmenter
from multipromptify.augmentations.text.context import ContextAugmenter
from multipromptify.augmentations.text.paraphrase import Paraphrase
# Text augmenters
from multipromptify.augmentations.text.format_structure import FormatStructureAugmenter
from multipromptify.augmentations.text.noise import TextNoiseAugmenter

__all__ = [
    "BaseAxisAugmenter",
    "FormatStructureAugmenter",  # New semantic-preserving format augmenter
    "TextNoiseAugmenter",  # New noise injection augmenter
    "Paraphrase",
    "ContextAugmenter",
    "FewShotAugmenter",
    "ShuffleAugmenter",
    "EnumeratorAugmenter",
    "OtherAugmenter"
]
