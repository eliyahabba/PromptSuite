"""
Structure-based augmentation modules.
"""

from multipromptify.augmentations.structure.fewshot import FewShotAugmenter
from multipromptify.augmentations.structure.shuffle import ShuffleAugmenter
from multipromptify.augmentations.structure.enumerate import EnumeratorAugmenter


__all__ = ["FewShotAugmenter", "ShuffleAugmenter", "EnumeratorAugmenter"] 