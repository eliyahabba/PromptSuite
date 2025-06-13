import random
import json
from typing import List, Dict, Any, Tuple

from src.multipromptify.augmentations.base import BaseAxisAugmenter
from src.shared.constants import ShuffleConstants


class ShuffleAugmenter(BaseAxisAugmenter):
    """
    Augmenter that shuffles list-like data and updates the gold field accordingly.
    
    This augmenter:
    1. Takes a list (or list-like string) as input
    2. Shuffles the order of items
    3. Returns the shuffled list and the new index of the correct answer
    
    Supported formats:
    - JSON list: ["item1", "item2", "item3"]
    - Comma-separated: "item1, item2, item3"
    - Newline-separated: "item1\nitem2\nitem3"
    """

    def __init__(self, n_augments=ShuffleConstants.DEFAULT_N_SHUFFLES):
        """Initialize the shuffle augmenter."""
        super().__init__(n_augments=n_augments)
    
    def get_name(self):
        return "Shuffle Variations"

    def augment(self, input_data: str, identification_data: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Generate shuffled variations of the input list.
        
        Args:
            input_data: String representation of a list or list-like data
            identification_data: Must contain 'gold_field' and 'gold_value' keys
            
        Returns:
            List of dictionaries containing 'shuffled_data' and 'new_gold_index'
        """
        if not identification_data or 'gold_field' not in identification_data or 'gold_value' not in identification_data:
            raise ValueError("ShuffleAugmenter requires identification_data with 'gold_field' and 'gold_value' keys")
        
        gold_value = identification_data['gold_value']
        
        # Parse the input data into a list
        try:
            data_list = self._parse_input_to_list(input_data)
        except ValueError as e:
            raise ValueError(f"ShuffleAugmenter can only work with list-like data: {e}")
        
        if len(data_list) <= 1:
            # Can't shuffle a list with 0 or 1 items
            return [{'shuffled_data': input_data, 'new_gold_index': gold_value}]
        
        # Find the current index of the correct answer
        current_gold_index = None
        
        # First, try to parse gold_value as an integer index
        try:
            current_gold_index = int(gold_value)
            if current_gold_index < 0 or current_gold_index >= len(data_list):
                raise ValueError(f"Gold index {current_gold_index} is out of range for list of length {len(data_list)}")
        except (ValueError, TypeError):
            # If it's not a valid index, try to find the value in the list
            try:
                # Look for the gold_value in the parsed list items
                for i, item in enumerate(data_list):
                    # Try exact match first
                    if item.strip() == gold_value.strip():
                        current_gold_index = i
                        break
                    # Try partial match (for multiple choice where gold_value might be just "Paris" but item is "Paris")
                    if gold_value.strip() in item.strip() or item.strip() in gold_value.strip():
                        current_gold_index = i
                        break
                
                if current_gold_index is None:
                    raise ValueError(f"Could not find gold value '{gold_value}' in the list items: {data_list}")
                    
            except Exception as e:
                raise ValueError(f"Gold value '{gold_value}' must be either a valid integer index or a value present in the list: {e}")
        
        variations = []
        
        # Generate n_augments shuffled variations
        for i in range(self.n_augments):
            # Create a copy of the list to shuffle
            shuffled_list = data_list.copy()
            
            # Shuffle the list
            random.seed(i)  # For reproducible results
            random.shuffle(shuffled_list)
            
            # Find where the original correct answer ended up
            original_correct_item = data_list[current_gold_index]
            new_gold_index = shuffled_list.index(original_correct_item)
            
            # Convert back to string format
            shuffled_data = self._list_to_string(shuffled_list, input_data)
            
            variations.append({
                'shuffled_data': shuffled_data,
                'new_gold_index': str(new_gold_index)
            })
        
        return variations
    
    def _parse_input_to_list(self, input_data: str) -> List[str]:
        """
        Parse input string into a list.
        
        Supports multiple formats:
        - JSON list: ["item1", "item2", "item3"]
        - Comma-separated: "item1, item2, item3"
        - Newline-separated: "item1\nitem2\nitem3"
        """
        input_data = input_data.strip()
        
        # Try JSON format first
        try:
            parsed = json.loads(input_data)
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
        except json.JSONDecodeError:
            pass
        
        # Try comma-separated format
        if ',' in input_data:
            return [item.strip() for item in input_data.split(',') if item.strip()]
        
        # Try newline-separated format
        if '\n' in input_data:
            return [item.strip() for item in input_data.split('\n') if item.strip()]
        
        # If none of the above work, raise an error
        raise ValueError(f"Could not parse '{input_data}' as a list. Supported formats: JSON list, comma-separated, or newline-separated.")
    
    def _list_to_string(self, data_list: List[str], original_format: str) -> str:
        """
        Convert list back to string format, preserving the original format style.
        """
        original_format = original_format.strip()
        
        # If original was JSON, return JSON
        try:
            json.loads(original_format)
            return json.dumps(data_list)
        except json.JSONDecodeError:
            pass
        
        # If original was comma-separated, return comma-separated
        if ',' in original_format:
            return ', '.join(data_list)
        
        # If original was newline-separated, return newline-separated
        if '\n' in original_format:
            return '\n'.join(data_list)
        
        # Default: return comma-separated
        return ', '.join(data_list)


def main():
    """Example usage of ShuffleAugmenter."""
    augmenter = ShuffleAugmenter(n_augments=3)
    
    # Example 1: Comma-separated format
    options1 = "Paris, London, Berlin, Madrid"
    identification_data1 = {
        'gold_field': 'answer',
        'gold_value': '0'  # Paris is the correct answer (index 0)
    }
    
    print("Original options:", options1)
    print("Gold value:", identification_data1['gold_value'])
    
    variations1 = augmenter.augment(options1, identification_data1)
    for i, var in enumerate(variations1):
        print(f"\nVariation {i+1}:")
        print("Shuffled:", var['shuffled_data'])
        print("New gold index:", var['new_gold_index'])
    
    # Example 2: JSON list format
    options2 = '["Apple", "Banana", "Cherry", "Date"]'
    identification_data2 = {
        'gold_field': 'correct_fruit',
        'gold_value': '1'  # Banana is correct (index 1)
    }
    
    print("\n\nOriginal options:", options2)
    print("Gold value:", identification_data2['gold_value'])
    
    variations2 = augmenter.augment(options2, identification_data2)
    for i, var in enumerate(variations2):
        print(f"\nVariation {i+1}:")
        print("Shuffled:", var['shuffled_data'])
        print("New gold index:", var['new_gold_index'])
    
    # Example 3: Value-based gold field (not index)
    options3 = "Paris, London, Berlin, Madrid"
    identification_data3 = {
        'gold_field': 'answer',
        'gold_value': 'Paris'  # Paris is the correct answer (by value, not index)
    }
    
    print("\n\nOriginal options:", options3)
    print("Gold value:", identification_data3['gold_value'])
    
    variations3 = augmenter.augment(options3, identification_data3)
    for i, var in enumerate(variations3):
        print(f"\nVariation {i+1}:")
        print("Shuffled:", var['shuffled_data'])
        print("New gold index:", var['new_gold_index'])


if __name__ == "__main__":
    main() 