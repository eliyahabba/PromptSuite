"""
Structure-based augmentation modules.
"""

from promptsuite.augmentations.structure.fewshot import FewShotAugmenter
from promptsuite.augmentations.structure.shuffle import ShuffleAugmenter
from promptsuite.augmentations.structure.enumerate import EnumeratorAugmenter


__all__ = ["FewShotAugmenter", "ShuffleAugmenter", "EnumeratorAugmenter"] 