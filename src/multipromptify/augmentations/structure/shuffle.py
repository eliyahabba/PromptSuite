import random
from typing import List, Dict, Any, Optional

from multipromptify.augmentations.base import BaseAxisAugmenter
from multipromptify.core.exceptions import AugmentationConfigurationError, InvalidAugmentationInputError, \
    ShuffleIndexError
from multipromptify.shared.constants import BaseAugmenterConstants, ListFormattingConstants


class ShuffleAugmenter(BaseAxisAugmenter):
    """
    Augmenter that shuffles list data and updates the gold field accordingly.
    
    This augmenter:
    1. Takes a list as input (must be actual Python list)
    2. Shuffles the order of items
    3. Returns the shuffled list and the new index of the correct answer
    
    Input must be a Python list. If you have string data, convert it to list first.
    """

    def __init__(self, n_augments=BaseAugmenterConstants.DEFAULT_N_AUGMENTS, seed: Optional[int] = None):
        """Initialize the shuffle augmenter."""
        super().__init__(n_augments=n_augments)
        self.seed = seed

    def get_name(self):
        return "Shuffle Variations"

    def augment(self, input_data: str, identification_data: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Generate shuffled variations of the input list.
        
        Args:
            input_data: Must be a comma-separated string that represents a list
            identification_data: Must contain 'gold_field' and 'gold_value' keys
            
        Returns:
            List of dictionaries containing 'shuffled_data' and 'new_gold_index'
        """
        if not identification_data or 'gold_field' not in identification_data or 'gold_value' not in identification_data:
            raise AugmentationConfigurationError("ShuffleAugmenter", ['gold_field', 'gold_value'])

        # Convert comma-separated string to list
        if not isinstance(input_data, str):
            raise InvalidAugmentationInputError("ShuffleAugmenter", "string", type(input_data).__name__)

        # Simple split by comma - input should already be formatted as "item1, item2, item3"
        data_list = [item.strip() for item in input_data.split(ListFormattingConstants.DEFAULT_LIST_SEPARATOR)]

        if len(data_list) <= 1:
            # Can't shuffle a list with 0 or 1 items
            return [{'shuffled_data': input_data, 'new_gold_index': identification_data['gold_value']}]

        # Find the current index of the correct answer
        gold_value = identification_data['gold_value']

        # Parse gold_value as integer index
        try:
            current_gold_index = int(gold_value)
            if current_gold_index < 0 or current_gold_index >= len(data_list):
                raise ShuffleIndexError(current_gold_index, len(data_list))
        except (ValueError, TypeError):
            raise ShuffleIndexError(gold_value, len(data_list))

        variations = []

        # Generate n_augments shuffled variations
        for i in range(self.n_augments):
            # Create a copy of the list to shuffle
            shuffled_list = data_list.copy()

            # Use seed + i to get different shuffles for each variation
            if self.seed is not None:
                random.seed(self.seed + i)  # Different seed for each variation
            else:
                random.seed(i)  # Fallback to original behavior

            random.shuffle(shuffled_list)

            # Find where the original correct answer ended up
            original_correct_item = data_list[current_gold_index]
            new_gold_index = shuffled_list.index(original_correct_item)

            # Convert back to comma-separated string
            shuffled_data = ', '.join(shuffled_list)

            variations.append({
                'shuffled_data': shuffled_data,
                'new_gold_index': str(new_gold_index)
            })

        return variations


def main():
    """Example usage of ShuffleAugmenter."""
    augmenter = ShuffleAugmenter(n_augments=3, seed=42)

    # Example: Comma-separated format (the expected input format)
    options = "Paris, London, Berlin, Madrid"
    identification_data = {
        'gold_field': 'answer',
        'gold_value': '0'  # Paris is the correct answer (index 0)
    }

    print("Original options:", options)
    print("Gold value:", identification_data['gold_value'])

    variations = augmenter.augment(options, identification_data)
    for i, var in enumerate(variations):
        print(f"\nVariation {i + 1}:")
        print("Shuffled:", var['shuffled_data'])
        print("New gold index:", var['new_gold_index'])


if __name__ == "__main__":
    main()
