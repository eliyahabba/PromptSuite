# Augmentor for custom augmentations
# This module provides an augmenter that generates variations of a prompt
from typing import List, Dict, Any, Optional
from multipromptify.augmentations.base import BaseAxisAugmenter
from multipromptify.shared.model_client import get_completion


class OtherAugmenter(BaseAxisAugmenter):
    """
    Augmenter that gets input from the user and generates variations of the prompt
    according to the user's input, using an LLM in the background.
    """

    def __init__(self, n_augments=3, augmentation_title="", augmentation_description="", augmentation_examples="", seed: Optional[int] = None):
        """
        Initialize the context augmenter.

        Args:
            n_augments: Number of variations to generate
            augmentation_title: Title of the augmentation
            augmentation_description: Description of the augmentation
            seed: Random seed for reproducibility
        """
        super().__init__(n_augments=n_augments, seed=seed)
        self.augmentation_title = augmentation_title
        self.augmentation_description = augmentation_description
        self.augmentation_examples = augmentation_examples
        self.meta_prompt = ""

    def get_name(self):
        return "Other Variations " + self.augmentation_title

    def _create_meta_prompt(self, augmentation_title: str, augmentation_description: str) -> str:
        """
        Create a meta-prompt to ask the language model to add context.
        This function will be revoked by the augment method, at the first call.

        Args:
            augmentation_title: Title of the augmentation
            augmentation_description: Description of the augmentation

        Returns:
            A meta-prompt for the language model
        """

        res =  f"""
        You are a specialized text augmentation system. Your task is to apply the specified augmentation technique to the input text and return ONLY the augmented result as a Python string.
        Augmentation Details:
        
        Title: {augmentation_title}
        Description: {augmentation_description}
        
        Instructions:
        
        Carefully analyze the augmentation title and description to understand the exact modification required
        Apply this specific augmentation technique to the input text
        Ensure the augmented text:
        
        Maintains the original meaning, structure, and flow
        Does NOT reveal or change any answers to problems in the text
        Does NOT alter the difficulty level or expected solution
        Preserves special characters, formatting, and structure when appropriate
        Feels natural and coherent
        
        
        For texts containing questions or problems:
        
        Preserve the original intent and solvability
        Maintain the same answer/solution as the original
        Do not add hints that would make the problem easier
        
        
        Return ONLY the augmented text as a Python list of strings, with no explanations, notes, or additional commentary.
        """

        if self.augmentation_examples:
            res += f"""
            Here are some examples of the augmentation:
            {self.augmentation_examples}
            """

        return res

    def augment(self, input_text: str, identification_data: Dict[str, Any] = None) -> List[str]:
        """
        Generate variations of the prompt according to the user's input.

        Args:
            input_text: The original prompt text
            identification_data: Data from the identifier (not used in this augmenter)

        Returns:
            List of variations with added context
        """
        if not self.meta_prompt:
            self.meta_prompt = self._create_meta_prompt(self.augmentation_title, self.augmentation_description)
        variations = [input_text]  # Start with the original prompt

        # Generate n_augments-1 variations (since we already have the original)
        for _ in range(self.n_augments - 1):
            # Generate the variation
            new_variation = self._generate_variation(input_text)
            if new_variation and new_variation != input_text:
                variations.append(new_variation)

        return variations

    def _generate_variation(self, text: str) -> str:
        """
        Generate a single variation by adding context.

        Args:
            text: The original text

        Returns:
            A new variation of the text
        """
        # Call language model to generate the variation
        try:
            temp = self.meta_prompt + f"Input Text: {text} \nReturn only the augmented result as a Python string."
            result = get_completion(temp)
            # Check if the result is valid (not empty and not the same as the original prompt and the original prompt is in the result)
            if result and result != text:
                return result
            else:
                return text
        except Exception as e:
            return text


if __name__ == "__main__":
    # Create the augmenter
    augmenter = OtherAugmenter(n_augments=3,
                               augmentation_title="Capitalization Variations",
                               augmentation_description="Generate variations of the prompt by changing the capitalization of words in the text.")

    # Example 1: Simple question
    prompt1 = "What is the capital of France?"
    variations1 = augmenter.augment(prompt1)

    print(f"Original prompt: {prompt1}")
    print(f"\nGenerated {len(variations1)} variations:")
    for i, variation in enumerate(variations1):
        print(f"\nVariation {i + 1}:")
        print(variation)
        print("-" * 50)
