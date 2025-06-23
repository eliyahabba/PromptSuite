"""
Augmenter Factory: Centralized creation of augmenter instances with special handling.
"""

from typing import Dict, Any, Optional

from multipromptify.augmentations.base import BaseAxisAugmenter
from multipromptify.augmentations.structure.enumerate import EnumeratorAugmenter
from multipromptify.augmentations.structure.fewshot import FewShotAugmenter
from multipromptify.augmentations.structure.shuffle import ShuffleAugmenter
from multipromptify.augmentations.text.context import ContextAugmenter
from multipromptify.augmentations.text.format_structure import FormatStructureAugmenter
from multipromptify.augmentations.text.noise import TextNoiseAugmenter
from multipromptify.augmentations.text.paraphrase import Paraphrase
from multipromptify.core.template_keys import (
    PARAPHRASE_WITH_LLM, SHUFFLE_VARIATION, CONTEXT_VARIATION, FEW_SHOT_VARIATION, ENUMERATE_VARIATION,
    FORMAT_STRUCTURE_VARIATION, TYPOS_AND_NOISE_VARIATION
)


class AugmenterFactory:
    """
    Factory class for creating augmenter instances with centralized logic for handling
    different augmenter requirements and configurations.
    """

    _registry = {
        PARAPHRASE_WITH_LLM: Paraphrase,
        CONTEXT_VARIATION: ContextAugmenter,
        SHUFFLE_VARIATION: ShuffleAugmenter,
        FEW_SHOT_VARIATION: FewShotAugmenter,
        ENUMERATE_VARIATION: EnumeratorAugmenter,
        FORMAT_STRUCTURE_VARIATION: FormatStructureAugmenter,  # New semantic-preserving format augmenter
        TYPOS_AND_NOISE_VARIATION: TextNoiseAugmenter,  # New noise injection augmenter
    }

    @classmethod
    def create(
            cls,
            variation_type: str,
            n_augments: int,
            api_key: Optional[str] = None,
            seed: Optional[int] = None,
            **kwargs
    ) -> BaseAxisAugmenter:
        """
        Create an augmenter instance with appropriate configuration.
        
        Args:
            variation_type: Type of augmenter to create
            n_augments: Number of augmentations to generate
            api_key: API key for augmenters that require it (e.g., Paraphrase, ContextAugmenter)
            seed: Random seed for reproducibility
            **kwargs: Additional parameters for specific augmenters
            
        Returns:
            Configured augmenter instance
            
        Raises:
            ValueError: If variation_type is not supported
        """
        if variation_type not in cls._registry:
            # Return TextNoiseAugmenter as default fallback (instead of TextSurfaceAugmenter)
            print(f"⚠️ Unknown variation type '{variation_type}', using TextNoiseAugmenter as fallback")
            return TextNoiseAugmenter(n_augments=n_augments, seed=seed)

        augmenter_class = cls._registry[variation_type]

        # Handle special cases based on augmenter type
        if augmenter_class == Paraphrase:
            # Paraphrase requires api_key
            if api_key:
                return augmenter_class(n_augments=n_augments - 1, api_key=api_key, seed=seed)
            else:
                print(f"⚠️ Paraphrase augmenter requires api_key, using TextNoiseAugmenter as fallback")
                return TextNoiseAugmenter(n_augments=n_augments, seed=seed)

        elif augmenter_class == ContextAugmenter:
            # ContextAugmenter requires api_key
            if api_key:
                print(f"✅ Creating ContextAugmenter with API key")
                return augmenter_class(n_augments=n_augments, seed=seed)
            else:
                print(f"⚠️ ContextAugmenter requires api_key, using TextNoiseAugmenter as fallback")
                print(f"   Context variations add background information but need LLM API access")
                return TextNoiseAugmenter(n_augments=n_augments, seed=seed)

        elif augmenter_class == FewShotAugmenter:
            # FewShotAugmenter might have different parameters
            return augmenter_class(n_augments=n_augments, seed=seed)

        elif augmenter_class == EnumeratorAugmenter:
            # EnumeratorAugmenter can take custom enumeration patterns
            enumeration_patterns = kwargs.get('enumeration_patterns', None)
            if enumeration_patterns:
                return augmenter_class(enumeration_patterns=enumeration_patterns, n_augments=n_augments, seed=seed)
            else:
                return augmenter_class(n_augments=n_augments, seed=seed)

        elif augmenter_class == FormatStructureAugmenter:
            # FormatStructureAugmenter supports seed and uses n_augments for max combinations
            return augmenter_class(n_augments=n_augments, seed=seed)

        elif augmenter_class == TextNoiseAugmenter:
            # TextNoiseAugmenter supports seed and uses n_augments for max combinations
            return augmenter_class(n_augments=n_augments, seed=seed)

        else:
            # Standard augmenters (ShuffleAugmenter, etc.)
            return augmenter_class(n_augments=n_augments, seed=seed)

    @classmethod
    def get_available_types(cls) -> list:
        """
        Get list of available augmenter types.
        
        Returns:
            List of supported variation types
        """
        return list(cls._registry.keys())

    @classmethod
    def requires_api_key(cls, variation_type: str) -> bool:
        """
        Check if a variation type requires an API key.
        
        Args:
            variation_type: Type of augmenter to check
            
        Returns:
            True if API key is required, False otherwise
        """
        augmenter_class = cls._registry.get(variation_type)
        return augmenter_class == Paraphrase or augmenter_class == ContextAugmenter

    @classmethod
    def augment_with_special_handling(
            cls,
            augmenter: BaseAxisAugmenter,
            text: str,
            variation_type: str,
            identification_data: Optional[Dict[str, Any]] = None
    ) -> list:
        """
        Apply augmentation with special handling for different augmenter types.
        
        Args:
            augmenter: The augmenter instance to use
            text: Text to augment
            variation_type: Type of augmenter (for special handling)
            identification_data: Special data needed by some augmenters
            
        Returns:
            List of augmentations (format depends on augmenter type)
        """
        try:
            if variation_type == 'shuffle' and identification_data:
                # ShuffleAugmenter requires identification_data
                return augmenter.augment(text, identification_data)
            elif variation_type == 'fewshot' and identification_data:
                # FewShotAugmenter requires identification_data
                return augmenter.augment(text, identification_data)
            elif variation_type == 'enumerate':
                # EnumeratorAugmenter works with or without identification_data
                return augmenter.augment(text, identification_data)
            else:
                # Standard augmenters
                return augmenter.augment(text)

        except Exception as e:
            print(f"⚠️ Error in {variation_type} augmentation: {e}")
            return [text]  # Return original text as fallback

    @classmethod
    def extract_text_from_result(cls, result: Any, variation_type: str) -> list:
        """
        Extract text strings from augmenter results, handling different return formats.
        
        Args:
            result: Result from augmenter.augment()
            variation_type: Type of augmenter that produced the result
            
        Returns:
            List of text strings
        """
        if not result:
            return []

        if isinstance(result, str):
            return [result]

        if isinstance(result, set):
            # Handle sets (e.g., from FormatStructureAugmenter)
            return list(result)

        if isinstance(result, list):
            # Handle list of strings
            if len(result) > 0 and isinstance(result[0], str):
                return result

            # Handle list of dictionaries (e.g., from ShuffleAugmenter)
            if len(result) > 0 and isinstance(result[0], dict):
                extracted = []
                for item in result:
                    if variation_type == 'shuffle' and 'shuffled_data' in item:
                        extracted.append(item['shuffled_data'])
                    elif 'data' in item:
                        extracted.append(item['data'])
                    elif 'text' in item:
                        extracted.append(item['text'])
                    else:
                        extracted.append(str(item))
                return extracted

        # Fallback: convert to string
        return [str(result)]


def create_augmenter(variation_type: str, n_augments: int, api_key: Optional[str] = None,
                     seed: Optional[int] = None) -> BaseAxisAugmenter:
    """
    Convenience function to create an augmenter instance.
    
    Args:
        variation_type: Type of augmenter to create
        n_augments: Number of augmentations to generate
        api_key: API key for augmenters that require it
        seed: Random seed for reproducibility
        
    Returns:
        Configured augmenter instance
    """
    return AugmenterFactory.create(variation_type, n_augments, api_key, seed)


def get_augmenter_types() -> list:
    """
    Get list of available augmenter types.
    
    Returns:
        List of supported variation types
    """
    return AugmenterFactory.get_available_types()
