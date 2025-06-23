import random
from typing import List, Dict, Any, Optional
from multipromptify.augmentations.base import BaseAxisAugmenter
from multipromptify.shared.model_client import get_completion


class ContextAugmenter(BaseAxisAugmenter):
    """
    Augmenter that adds irrelevant context before or after the main prompt.
    This doesn't change the meaning of the task but makes the prompt longer.
    """

    def __init__(self, n_augments=3, seed: Optional[int] = None):
        """
        Initialize the context augmenter.

        Args:
            n_augments: Number of variations to generate
            seed: Random seed for reproducibility
        """
        super().__init__(n_augments=n_augments, seed=seed)
        
    def get_name(self):
        return "Context Variations"

    def augment(self, prompt: str, identification_data: Dict[str, Any] = None) -> List[str]:
        """
        Generate variations of the prompt by adding irrelevant context.

        Args:
            prompt: The original prompt text
            identification_data: Data from the identifier (not used in this augmenter)

        Returns:
            List of variations with added context
        """
        variations = [prompt]  # Start with the original prompt
        
        # Generate n_augments-1 variations (since we already have the original)
        for _ in range(self.n_augments - 1):
            # Randomly decide whether to add context before, after, or both
            variation_type = random.choice(["before", "after", "both"])
            
            # Generate the variation
            new_variation = self._generate_variation(prompt, variation_type)
            if new_variation and new_variation != prompt:
                variations.append(new_variation)
        
        return variations

    def _generate_variation(self, prompt: str, variation_type: str) -> str:
        """
        Generate a single variation by adding context.

        Args:
            prompt: The original prompt
            variation_type: Where to add context ("before", "after", or "both")

        Returns:
            A new variation of the prompt
        """
        # Create a meta-prompt to ask the language model to add irrelevant context
        meta_prompt = self._create_meta_prompt(prompt, variation_type)
        
        # Call language model to generate the variation
        try:
            result = get_completion(meta_prompt)
            # Check if the result is valid (not empty and not the same as the original prompt and the original prompt is in the result)
            if result and result != prompt and prompt in result:
                return result
            else:
                return prompt
        except Exception as e:
            return prompt

    def _create_meta_prompt(self, prompt: str, variation_type: str) -> str:
        """
        Create a meta-prompt to ask the language model to add context.

        Args:
            prompt: The original prompt
            variation_type: Where to add context

        Returns:
            A meta-prompt for the language model
        """
        if variation_type == "before":
            return f"""
            Your task is to add context BEFORE the following prompt.

            IMPORTANT GUIDELINES:
            1. The added context SHOULD be thematically related to the original prompt, maintaining a coherent flow.
            2. The added context must NOT contain any hint or answer to the original prompt.
            3. The added context should NOT change the meaning or expected answer of the original prompt.
            4. The context should provide additional background or related information that feels natural.
            5. You MUST preserve the original prompt EXACTLY as given, including any spelling or grammatical errors. Do not change a single character in the original prompt.

            Original prompt:
            "{prompt}"

            Return ONLY the modified prompt with added context before it. Do not include any explanations.
            """
        elif variation_type == "after":
            return f"""
            Your task is to add context AFTER the following prompt.

            IMPORTANT GUIDELINES:
            1. The added context SHOULD be thematically related to the original prompt, maintaining a coherent flow.
            2. The added context must NOT contain any hint or answer to the original prompt.
            3. The added context should NOT change the meaning or expected answer of the original prompt.
            4. The context should provide additional background or related information that feels natural.
            5. You MUST preserve the original prompt EXACTLY as given, including any spelling or grammatical errors. Do not change a single character in the original prompt.

            Original prompt:
            "{prompt}"

            Return ONLY the modified prompt with added context after it. Do not include any explanations.
            """
        else:  # both
            return f"""
            Your task is to add context BOTH BEFORE AND AFTER the following prompt.

            IMPORTANT GUIDELINES:
            1. The added context SHOULD be thematically related to the original prompt, maintaining a coherent flow.
            2. The added context must NOT contain any hint or answer to the original prompt.
            3. The added context should NOT change the meaning or expected answer of the original prompt.
            4. The context should provide additional background or related information that feels natural.
            5. You MUST preserve the original prompt EXACTLY as given, including any spelling or grammatical errors. Do not change a single character in the original prompt.

            Original prompt:
            "{prompt}"

            Return ONLY the modified prompt with added context before and after it. Do not include any explanations.
            """
if __name__ == "__main__":
    # Create the augmenter
    augmenter = ContextAugmenter(n_augments=3)
    
    # Example 1: Simple question
    prompt1 = "What is the capital of France?"
    variations1 = augmenter.augment(prompt1)
    
    print(f"Original prompt: {prompt1}")
    print(f"\nGenerated {len(variations1)} variations:")
    for i, variation in enumerate(variations1):
        print(f"\nVariation {i+1}:")
        print(variation)
        print("-" * 50)
    
    # Example 2: More complex prompt
    prompt2 = "Explain the process of photosynthesis in plants."
    variations2 = augmenter.augment(prompt2)

    print(f"\nOriginal prompt: {prompt2}")
    print(f"\nGenerated {len(variations2)} variations:")
    for i, variation in enumerate(variations2):
        print(f"\nVariation {i+1}:")
        print(variation)
        print("-" * 50)

