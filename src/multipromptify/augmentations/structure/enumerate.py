from typing import List, Dict, Any, Union

from multipromptify.augmentations.base import BaseAxisAugmenter
from multipromptify.exceptions import (
    InvalidAugmentationInputError, 
    EnumeratorLengthMismatchError,
    AugmentationConfigurationError
)


class EnumeratorAugmenter(BaseAxisAugmenter):
    """
    Augmenter that adds enumeration (numbering) to specified fields.
    
    This augmenter works with template configuration like:
    'enumerate': {
        'field': 'options',    # Which field to enumerate
        'type': '1234'         # Type of enumeration: '1234', 'ABCD', 'abcd', etc.
    }
    
    The augmenter:
    1. Takes the specified field's data (comma-separated string or list)
    2. Applies enumeration with the specified type
    3. Returns enumerated list with format: "1. Item1 2. Item2 3. Item3"
    """

    # Predefined enumeration types
    ENUMERATION_TYPES = {
        '1234': '1234567890',
        'ABCD': 'ABCDEFGHIJKLMNOPQRSTUVWXYZ',
        'abcd': 'abcdefghijklmnopqrstuvwxyz',
        'hebrew': 'אבגדהוזחטיכלמנסעפצקרשת',
        'greek': 'αβγδεζηθικλμνξοπρστυφχψω',
        'roman': ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X']
    }

    def __init__(self, n_augments=1):
        """Initialize the enumerator augmenter."""
        super().__init__(n_augments=n_augments)
    
    def get_name(self):
        return "Enumerate Field"

    def _get_enumeration_sequence(self, enum_type: str) -> List[str]:
        """Get enumeration sequence based on type."""
        if enum_type in self.ENUMERATION_TYPES:
            sequence = self.ENUMERATION_TYPES[enum_type]
            if isinstance(sequence, str):
                return list(sequence)
            else:
                return [str(item) for item in sequence]
        else:
            # If custom type provided, treat as string
            return list(enum_type)

    def _enumerate_list(self, data_list: List[str], enumeration_sequence: List[str]) -> str:
        """
        Apply enumeration to a list using the provided sequence.
        
        Args:
            data_list: List of items to enumerate
            enumeration_sequence: Sequence to use for enumeration
            
        Returns:
            Enumerated string with format "1. Item1 2. Item2 3. Item3"
        """
        if len(enumeration_sequence) < len(data_list):
            raise EnumeratorLengthMismatchError(
                len(enumeration_sequence), 
                len(data_list), 
                f"type: {enumeration_sequence[:5]}..."
            )
        
        enumerated_items = []
        for i, item in enumerate(data_list):
            enumerated_items.append(f"{enumeration_sequence[i]}. {item}")
        
        return " ".join(enumerated_items)

    def enumerate_field(self, field_data: Any, enum_type: str) -> str:
        """
        Enumerate a field's data with the specified enumeration type.
        
        Args:
            field_data: The field data to enumerate (string or list)
            enum_type: Type of enumeration ('1234', 'ABCD', etc.)
            
        Returns:
            Enumerated string
        """
        # Convert input to list
        if isinstance(field_data, str):
            # Assume comma-separated format
            data_list = [item.strip() for item in field_data.split(',')]
        elif isinstance(field_data, list):
            data_list = [str(item) for item in field_data]
        else:
            # Convert single value to string
            data_list = [str(field_data)]
        
        if len(data_list) == 0:
            return str(field_data)
        
        # Get enumeration sequence
        enumeration_sequence = self._get_enumeration_sequence(enum_type)
        
        # Apply enumeration
        return self._enumerate_list(data_list, enumeration_sequence)

    def augment(self, input_data: str, identification_data: Dict[str, Any] = None) -> List[str]:
        """
        This method is kept for compatibility but enumerate should be used
        through the template configuration system, not as a direct field variation.
        """
        # For direct usage, use default '1234' enumeration
        try:
            result = self.enumerate_field(input_data, '1234')
            return [result]
        except Exception as e:
            print(f"⚠️ Error in enumerate augmentation: {e}")
            return [input_data]


def main():
    """Example usage of EnumeratorAugmenter."""
    
    print("=== EnumeratorAugmenter Usage Examples ===")
    
    augmenter = EnumeratorAugmenter()
    
    # Test different enumeration types
    options = "Venus, Mercury, Earth, Mars"
    print(f"Original options: {options}")
    
    for enum_type in ['1234', 'ABCD', 'abcd', 'hebrew']:
        try:
            result = augmenter.enumerate_field(options, enum_type)
            print(f"Type '{enum_type}': {result}")
        except Exception as e:
            print(f"Type '{enum_type}': Error - {e}")
    
    # Test with list input
    options_list = ["Venus", "Mercury", "Earth", "Mars"]
    result = augmenter.enumerate_field(options_list, 'ABCD')
    print(f"List input: {result}")
    
    # Test error case
    try:
        short_type = "AB"  # Only 2 characters
        result = augmenter.enumerate_field(options, short_type)
        print(f"Short type: {result}")
    except EnumeratorLengthMismatchError as e:
        print(f"Expected error: {e}")


if __name__ == "__main__":
    main() 