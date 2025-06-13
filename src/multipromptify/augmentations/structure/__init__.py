"""
Structure-based augmentation modules.
"""

from src.multipromptify.augmentations.structure.fewshot import FewShotAugmenter
from src.multipromptify.augmentations.structure.shuffle import ShuffleAugmenter


__all__ = ["FewShotAugmenter", "ShuffleAugmenter"] 