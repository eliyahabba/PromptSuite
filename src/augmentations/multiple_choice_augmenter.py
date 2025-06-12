import random
from typing import List, Dict, Any

from src.augmentations.base_augmenter import BaseAxisAugmenter
from src.shared.constants import MultipleChoiceConstants


class MultipleChoiceAugmenter(BaseAxisAugmenter):
    """
    Augmenter for multiple choice questions.
    
    Creates variations by:
    1. Changing the enumeration style (A/B/C, 1/2/3, etc.)
    2. Changing the order of answer options
    """

    def __init__(self, n_augments=3):
        """Initialize the multiple choice augmenter."""
        super().__init__(n_augments=n_augments)
        
        # Define available enumeration styles
        self.enumeration_styles = MultipleChoiceConstants.ENUMERATION_STYLES
    
    def get_name(self):
        return "Multiple Choice Variations"

    def augment(self, prompt: str, identification_data: Dict[str, Any] = None) -> List[str]:
        """Generate variations of the multiple choice question."""
        # Start with the original prompt
        variations = [prompt]
        
        # If no identification data is provided, return original prompt
        if not identification_data:
            return variations
        
        # Extract data
        question = identification_data.get("question", "")
        options = identification_data.get("options", [])
        current_markers = identification_data.get("markers", [])
        
        if not question or not options or not current_markers:
            return variations
        
        # Find current style
        current_style_index = -1
        for i, style in enumerate(self.enumeration_styles):
            if current_markers[0] in style:
                current_style_index = i
                break
        
        if current_style_index == -1:
            current_style_index = 0
        
        # 1. Create variations with different enumeration styles
        for i, style in enumerate(self.enumeration_styles):
            if i == current_style_index:
                continue  # Skip current style
                
            # Create variation with new style
            new_prompt = question + "\n\n"
            for j, option in enumerate(options):
                if j < len(style):
                    new_prompt += f"{style[j]} {option}\n"
            
            variations.append(new_prompt.strip())
        
        # 2. Create variations with different order
        current_style = self.enumeration_styles[current_style_index]
        for _ in range(min(2, self.n_augments)):
            # Shuffle options
            shuffled_indices = list(range(len(options)))
            random.shuffle(shuffled_indices)
            
            # Skip if order is unchanged
            if shuffled_indices == list(range(len(options))):
                continue
            
            # Create variation with new order
            new_prompt = question + "\n\n"
            for j, idx in enumerate(shuffled_indices):
                if j < len(current_style) and idx < len(options):
                    new_prompt += f"{current_style[j]} {options[idx]}\n"
            
            variations.append(new_prompt.strip())
        
        # Remove duplicates and limit to n_augments
        variations = list(dict.fromkeys(variations))
        return variations[:self.n_augments]


def main():
    """Example usage of MultipleChoiceAugmenter."""
    # Create the augmenter
    augmenter = MultipleChoiceAugmenter(n_augments=15)
    
    # Example 1: Simple multiple choice question
    example1 = {
        "question": "What is the capital of France?",
        "options": ["Paris", "London", "Berlin", "Madrid"],
        "markers": ["A", "B", "C", "D"]
    }
    
    prompt1 = example1["question"] + "\n\n"
    for i, option in enumerate(example1["options"]):
        prompt1 += f"{example1['markers'][i]} {option}\n"
    
    print("Original prompt 1:")
    print(prompt1)
    print("\nVariations:")
    
    variations1 = augmenter.augment(prompt1, example1)
    for i, var in enumerate(variations1):
        print(f"\nVariation {i+1}:")
        print(var)
    
    # Example 2: Quiz question
    example2 = {
        "question": "Which of the following is NOT a programming language?",
        "options": ["Python", "Java", "HTML", "Banana"],
        "markers": ["1)", "2)", "3)", "4)"]
    }
    
    prompt2 = example2["question"] + "\n\n"
    for i, option in enumerate(example2["options"]):
        prompt2 += f"{example2['markers'][i]} {option}\n"
    
    print("\n\nOriginal prompt 2:")
    print(prompt2)
    print("\nVariations:")
    
    variations2 = augmenter.augment(prompt2, example2)
    for i, var in enumerate(variations2):
        print(f"\nVariation {i+1}:")
        print(var)


if __name__ == "__main__":
    main() 