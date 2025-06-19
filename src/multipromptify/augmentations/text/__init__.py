"""
Text-based augmentation modules.
"""

from multipromptify.augmentations.text.surface import TextSurfaceAugmenter
from multipromptify.augmentations.text.paraphrase import Paraphrase
from multipromptify.augmentations.text.context import ContextAugmenter
from multipromptify.core.template_keys import PARAPHRASE_WITH_LLM, REWORDING

__all__ = ["TextSurfaceAugmenter", "Paraphrase", "ContextAugmenter"] 