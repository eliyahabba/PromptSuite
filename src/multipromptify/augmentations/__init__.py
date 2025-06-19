"""
Augmentation modules for MultiPromptify.
"""

# Import all augmenters for easy access
from multipromptify.augmentations.base import BaseAxisAugmenter
from multipromptify.augmentations.pipeline import AugmentationPipeline

# Text augmenters
from multipromptify.augmentations.text.surface import TextSurfaceAugmenter
from multipromptify.augmentations.text.paraphrase import Paraphrase
from multipromptify.augmentations.text.context import ContextAugmenter

# Structure augmenters  
from multipromptify.augmentations.structure.fewshot import FewShotAugmenter
from multipromptify.augmentations.structure.shuffle import ShuffleAugmenter
from multipromptify.augmentations.structure.enumerate import EnumeratorAugmenter

# Other augmenters
from multipromptify.augmentations.other import OtherAugmenter

from multipromptify.template_keys import PARAPHRASE_WITH_LLM, REWORDING

__all__ = [
    "BaseAxisAugmenter",
    "AugmentationPipeline", 
    "TextSurfaceAugmenter",
    "Paraphrase",
    "ContextAugmenter",
    "FewShotAugmenter", 
    "ShuffleAugmenter",
    "EnumeratorAugmenter",

    "OtherAugmenter"
] 