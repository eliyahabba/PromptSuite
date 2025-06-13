"""
Augmentation modules for MultiPromptify.
"""

# Import all augmenters for easy access
from src.multipromptify.augmentations.base import BaseAxisAugmenter
from src.multipromptify.augmentations.pipeline import AugmentationPipeline

# Text augmenters
from src.multipromptify.augmentations.text.surface import TextSurfaceAugmenter
from src.multipromptify.augmentations.text.paraphrase import Paraphrase
from src.multipromptify.augmentations.text.context import ContextAugmenter

# Structure augmenters  
from src.multipromptify.augmentations.structure.fewshot import FewShotAugmenter
from src.multipromptify.augmentations.structure.shuffle import ShuffleAugmenter


# Other augmenters
from src.multipromptify.augmentations.other import OtherAugmenter

__all__ = [
    "BaseAxisAugmenter",
    "AugmentationPipeline", 
    "TextSurfaceAugmenter",
    "Paraphrase",
    "ContextAugmenter",
    "FewShotAugmenter", 
    "ShuffleAugmenter",

    "OtherAugmenter"
] 