"""
Augmentation modules for PromptSuiteEngine.
"""

# Import all augmenters for easy access
from promptsuite.augmentations.base import BaseAxisAugmenter
# Other augmenters
from promptsuite.augmentations.other import OtherAugmenter
from promptsuite.augmentations.structure.enumerate import EnumeratorAugmenter
# Structure augmenters
from promptsuite.augmentations.structure.fewshot import FewShotAugmenter
from promptsuite.augmentations.structure.shuffle import ShuffleAugmenter
from promptsuite.augmentations.text.context import ContextAugmenter
from promptsuite.augmentations.text.paraphrase import Paraphrase
# Text augmenters
from promptsuite.augmentations.text.format_structure import FormatStructureAugmenter
from promptsuite.augmentations.text.noise import TextNoiseAugmenter

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
