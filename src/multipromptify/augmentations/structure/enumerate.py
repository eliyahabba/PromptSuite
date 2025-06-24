import random
from typing import List, Dict, Any

from multipromptify.augmentations.base import BaseAxisAugmenter
from multipromptify.core.exceptions import (
    EnumeratorLengthMismatchError
)
from multipromptify.shared.constants import ListFormattingConstants


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
        'hebrew': 'אבגדהוזחטיכסעפצקרשת',
        'greek': 'αβγδεζηθικλμνξοπρστυφχψω',
        'roman': ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X']
    }

    def __init__(self, n_augments=1, seed=None):
        """Initialize the enumerator augmenter."""
        super().__init__(n_augments=n_augments, seed=seed)
        self._rng = random.Random(self.seed) if self.seed is not None else random.Random()

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

        return ListFormattingConstants.DEFAULT_LIST_SEPARATOR.join(enumerated_items)

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
            # Try different separators: comma first, then newline
            if ',' in field_data:
                data_list = [item.strip() for item in field_data.split(',')]
            elif '\n' in field_data:
                data_list = [item.strip() for item in field_data.split('\n') if item.strip()]
            else:
                # Single item
                data_list = [field_data.strip()]
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
        Generate multiple variations with different enumeration types.
        
        Args:
            input_data: The input data to enumerate
            identification_data: Optional data containing enumeration configuration
            
        Returns:
            List of variations with different enumeration types
        """
        variations = []
        
        # Get enumeration type from identification_data if available
        enum_type = '1234'  # default
        if identification_data and 'enum_type' in identification_data:
            enum_type = identification_data['enum_type']
        
        # If we have n_augments > 1, generate multiple variations with different types
        if self.n_augments > 1:
            # Define different enumeration types to try
            enum_types = ['1234', 'ABCD', 'abcd', 'roman']
            
            # Use deterministic selection to ensure consistency between few-shot and main variations
            # Take the first n_augments types in order for consistency
            selected_types = enum_types[:min(self.n_augments, len(enum_types))]
            
            # Generate variations with different types
            for enum_type in selected_types:
                try:
                    result = self.enumerate_field(input_data, enum_type)
                    if result not in variations:
                        variations.append(result)
                except Exception as e:
                    print(f"⚠️ Error in enumerate augmentation with type {enum_type}: {e}")
                    continue
        else:
            # Single variation with specified type
            try:
                result = self.enumerate_field(input_data, enum_type)
                variations.append(result)
            except Exception as e:
                print(f"⚠️ Error in enumerate augmentation: {e}")
                # Don't return original input_data for enumerate - it should always enumerate
                return []
        
        # For enumerate, we should always return enumerated versions, not the original
        return variations


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
