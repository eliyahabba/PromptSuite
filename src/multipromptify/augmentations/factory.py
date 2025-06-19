"""
Augmenter Factory: Centralized creation of augmenter instances with special handling.
"""

from typing import Dict, Any, Optional
from multipromptify.augmentations.base import BaseAxisAugmenter
from multipromptify.augmentations.structure.shuffle import ShuffleAugmenter
from multipromptify.augmentations.structure.fewshot import FewShotAugmenter
from multipromptify.augmentations.structure.enumerate import EnumeratorAugmenter
from multipromptify.augmentations.text.context import ContextAugmenter
from multipromptify.augmentations.text.paraphrase import Paraphrase
from multipromptify.augmentations.text.surface import TextSurfaceAugmenter
from multipromptify.core.template_keys import (
    PARAPHRASE_WITH_LLM, REWORDING, SHUFFLE_VARIATION
)


class AugmenterFactory:
    """
    Factory class for creating augmenter instances with centralized logic for handling
    different augmenter requirements and configurations.
    """
    
    _registry = {
        PARAPHRASE_WITH_LLM: Paraphrase,
        REWORDING: TextSurfaceAugmenter,
        "context": ContextAugmenter,
        SHUFFLE_VARIATION: ShuffleAugmenter,
        "fewshot": FewShotAugmenter,
        "enumerate": EnumeratorAugmenter,
    }

    @classmethod
    def create(
        cls, 
        variation_type: str, 
        n_augments: int, 
        api_key: Optional[str] = None,
        **kwargs
    ) -> BaseAxisAugmenter:
        """
        Create an augmenter instance with appropriate configuration.
        
        Args:
            variation_type: Type of augmenter to create
            n_augments: Number of augmentations to generate
            api_key: API key for augmenters that require it (e.g., Paraphrase)
            **kwargs: Additional parameters for specific augmenters
            
        Returns:
            Configured augmenter instance
            
        Raises:
            ValueError: If variation_type is not supported
        """
        if variation_type not in cls._registry:
            # Return TextSurfaceAugmenter as default fallback
            print(f"⚠️ Unknown variation type '{variation_type}', using TextSurfaceAugmenter as fallback")
            return TextSurfaceAugmenter(n_augments=n_augments)
        
        augmenter_class = cls._registry[variation_type]
        
        # Handle special cases based on augmenter type
        if augmenter_class == Paraphrase:
            # Paraphrase requires api_key
            if api_key:
                return augmenter_class(n_augments=n_augments-1, api_key=api_key)
            else:
                print(f"⚠️ Paraphrase augmenter requires api_key, using TextSurfaceAugmenter as fallback")
                return TextSurfaceAugmenter(n_augments=n_augments)
                
        elif augmenter_class == FewShotAugmenter:
            # FewShotAugmenter might have different parameters
            return augmenter_class(n_augments=n_augments)
            
        elif augmenter_class == EnumeratorAugmenter:
            # EnumeratorAugmenter can take custom enumeration patterns
            enumeration_patterns = kwargs.get('enumeration_patterns', None)
            if enumeration_patterns:
                return augmenter_class(enumeration_patterns=enumeration_patterns, n_augments=n_augments)
            else:
                return augmenter_class(n_augments=n_augments)
            
        else:
            # Standard augmenters (TextSurfaceAugmenter, ContextAugmenter, ShuffleAugmenter)
            return augmenter_class(n_augments=n_augments)

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
        return augmenter_class == Paraphrase

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


# Convenience functions for common use cases
def create_augmenter(variation_type: str, n_augments: int, api_key: Optional[str] = None) -> BaseAxisAugmenter:
    """
    Convenience function to create an augmenter.
    
    Args:
        variation_type: Type of augmenter to create
        n_augments: Number of augmentations to generate
        api_key: API key for augmenters that require it
        
    Returns:
        Configured augmenter instance
    """
    return AugmenterFactory.create(variation_type, n_augments, api_key)


def get_augmenter_types() -> list:
    """
    Get list of available augmenter types.
    
    Returns:
        List of supported variation types
    """
    return AugmenterFactory.get_available_types() 