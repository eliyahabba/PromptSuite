"""
Augmentation pipeline that combines multiple augmentation methods.
"""
import random
from typing import List, Optional, Dict, Any

from multipromptify.augmentations.base import BaseAxisAugmenter
from multipromptify.augmentations.text.context import ContextAugmenter
from multipromptify.augmentations.structure.fewshot import FewShotAugmenter

from multipromptify.augmentations.structure.shuffle import ShuffleAugmenter
from multipromptify.augmentations.text.paraphrase import Paraphrase
from multipromptify.augmentations.text.surface import TextSurfaceAugmenter
from multipromptify.shared.constants import AugmentationPipelineConstants, BaseAugmenterConstants


class AugmentationPipeline:
    """
    A pipeline that applies multiple augmentation methods sequentially.
    Each augmenter in the pipeline processes all variations produced by the previous augmenter.
    """

    def __init__(self, augmenters: Optional[List[BaseAxisAugmenter]] = None, 
                 max_variations: int = AugmentationPipelineConstants.DEFAULT_MAX_VARIATIONS):
        """
        Initialize the augmentation pipeline.

        Args:
            augmenters: List of augmenters to apply in sequence. If None, a default set will be used.
            max_variations: Maximum number of variations to generate in total.
        """
        self.max_variations = max_variations

        # Use provided augmenters or create default ones
        if augmenters is not None:
            self.augmenters = augmenters
        else:
            self.augmenters = [
                TextSurfaceAugmenter(n_augments=BaseAugmenterConstants.DEFAULT_N_AUGMENTS),
                ContextAugmenter(n_augments=2)
            ]

    def apply_augmenter(self, augmenter: BaseAxisAugmenter, text: str, identification_data: Dict[str, Any] = None) -> \
    List[str]:
        """
        Apply a single augmenter to a text.

        Args:
            augmenter: The augmenter to apply
            text: The text to augment
            identification_data: Optional identification data for augmenters that need it

        Returns:
            List of augmented texts
        """
        # Handle different augmenter interfaces
        if isinstance(augmenter, Paraphrase):
            return augmenter.augment(text)
        elif isinstance(augmenter, ShuffleAugmenter) and identification_data:
            # ShuffleAugmenter returns List[Dict[str, Any]], extract the shuffled_data strings
            shuffle_results = augmenter.augment(text, identification_data)
            return [result['shuffled_data'] for result in shuffle_results]
        elif isinstance(augmenter, FewShotAugmenter):
            # If we have example pairs in identification_data, use them
            if identification_data:
                return augmenter.augment(text, identification_data)
            # Otherwise return the original text
            return [text]
        elif isinstance(augmenter, ContextAugmenter):
            # ContextAugmenter has a standard interface
            return augmenter.augment(text)
        elif isinstance(augmenter, TextSurfaceAugmenter):
            # TextSurfaceAugmenter has a standard interface
            return augmenter.augment(text)
        elif hasattr(augmenter, 'augment'):
            # Standard augmenter interface
            try:
                result = augmenter.augment(text, identification_data)
                # Handle potential dict return from augmenters like ShuffleAugmenter
                if isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict):
                    # If it's a list of dicts, extract the text field
                    if 'shuffled_data' in result[0]:
                        return [item['shuffled_data'] for item in result]
                    else:
                        # Try to find a text field in the dict
                        text_keys = ['text', 'content', 'data', 'output']
                        for key in text_keys:
                            if key in result[0]:
                                return [item[key] for item in result]
                        # If no text field found, convert dict to string
                        return [str(item) for item in result]
                return result
            except TypeError:
                # Try without identification_data if it fails
                try:
                    result = augmenter.augment(text)
                    # Handle potential dict return
                    if isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict):
                        if 'shuffled_data' in result[0]:
                            return [item['shuffled_data'] for item in result]
                        else:
                            text_keys = ['text', 'content', 'data', 'output']
                            for key in text_keys:
                                if key in result[0]:
                                    return [item[key] for item in result]
                            return [str(item) for item in result]
                    return result
                except:
                    # If all else fails, return the original text
                    return [text]
        else:
            # If the augmenter doesn't have an augment method, return the original text
            return [text]

    def augment(self, text: str, special_data: Dict[str, Any] = None) -> List[str]:
        """
        Apply all augmenters to the text and return all variations.
        
        Args:
            text: The base text to augment.
            special_data: Any special data needed by augmenters.
            
        Returns:
            List of augmented texts.
        """
        all_variations = [text]  # Start with the original text

        print("\nAugmentation pipeline:")

        # Apply each augmenter in sequence
        for i, augmenter in enumerate(self.augmenters):
            print(f"Applying augmenter {i+1}/{len(self.augmenters)}: {augmenter.__class__.__name__}")
            print(f"Input variations: {len(all_variations)}")
            print(f"  Step {i + 1}: Applying {augmenter.__class__.__name__}")
            new_variations = []

            # Apply the current augmenter to each variation produced so far
            for variation in all_variations:
                # Skip empty variations
                augmented = self.apply_augmenter(augmenter, variation, special_data)
                new_variations.extend(augmented)
            print(f"    Generated {len(new_variations)} variations")

            # Update the list of variations for the next augmenter
            # Check if we've reached the maximum number of variations
            if len(new_variations) >= self.max_variations:
                print(f"  Reached maximum of {self.max_variations} variations")
                all_variations = random.sample(new_variations, self.max_variations)
                break
            all_variations = new_variations
            print(f"  After step  {i + 1}: {len(all_variations)} total variations")


        print(f"Final: Generated {len(all_variations)} total variations")
        return all_variations


def run_basic_augmentation_example():
    """
    Run a basic example with text surface, context, and paraphrase augmenters.
    """
    print("\n--- Basic Augmentation Example ---")

    # Create individual augmenters
    text_surface_augmenter = TextSurfaceAugmenter(n_augments=3)
    context_augmenter = ContextAugmenter(n_augments=2)
    paraphrase_augmenter = Paraphrase(n_augments=2)

    # Create pipeline with explicit augmenters
    pipeline = AugmentationPipeline(
        augmenters=[paraphrase_augmenter, context_augmenter, text_surface_augmenter],
        max_variations=20
    )

    # Sample text
    original_text = "Please explain the process of photosynthesis in plants."

    # Apply the augmentation pipeline
    augmented_texts = pipeline.augment(original_text)

    # Print the results
    print(f"\nOriginal text: {original_text}")
    print(f"\nGenerated {len(augmented_texts)} variations:")

    for i, text in enumerate(augmented_texts):
        print(f"\n{i + 1}. {text}")


def run_shuffle_example():
    """
    Run an example with shuffle augmentation.
    """
    print("\n\n--- Shuffle Example ---")

    shuffle_augmenter = ShuffleAugmenter(n_augments=3)
    shuffle_text = "The quick brown fox jumps over the lazy dog."

    # Identification data for the shuffle question
    shuffle_data = {
        "question": "The quick brown fox jumps over the lazy dog.",
        "options": ["The quick brown fox jumps over the lazy dog.", "The lazy dog jumps over the quick brown fox."],
        "markers": ["A", "B"]
    }

    # Create a pipeline with just the shuffle augmenter
    shuffle_pipeline = AugmentationPipeline(augmenters=[shuffle_augmenter], max_variations=10)
    shuffle_variations = shuffle_pipeline.augment(shuffle_text, shuffle_data)

    print(f"\nOriginal text: {shuffle_text}")
    print(f"\nGenerated {len(shuffle_variations)} variations:")

    for i, text in enumerate(shuffle_variations):
        print(f"\n{i + 1}. {text}")


def run_fewshot_combined_example():
    """
    Run an example that combines few-shot formatting with other augmenters.
    """
    print("\n\n--- Few-Shot Combined Example ---")

    # Create sample data for few-shot augmentation
    import pandas as pd
    sample_data = pd.DataFrame({
        "input": [
            "What is the capital of France?",
            "What is the largest planet in our solar system?",
            "Who wrote Romeo and Juliet?"
        ],
        "output": [
            "Paris",
            "Jupiter",
            "William Shakespeare"
        ]
    })

    # Create a question that will be augmented with few-shot examples
    question = "What is the boiling point of water?"

    # Create example pairs for few-shot learning as tuples (input, output)
    example_pairs = [
        ("What is the capital of France?", "Paris"),
        ("What is the largest planet in our solar system?", "Jupiter"),
        ("Who wrote Romeo and Juliet?", "William Shakespeare")
    ]

    # Create identification data with the example pairs and dataset
    fewshot_data = {
        "example_pairs": example_pairs,
        "dataset": sample_data
    }

    # Create a few-shot augmenter for standalone testing
    fewshot_augmenter = FewShotAugmenter(num_examples=2, n_augments=3)

    # Test the augmenter directly first
    direct_results = fewshot_augmenter.augment_with_examples(question, example_pairs)

    print(f"\nOriginal question: {question}")
    print(f"\nFew-shot examples:")
    for input_q, output_a in example_pairs:
        print(f"Q: {input_q}")
        print(f"A: {output_a}")
    print("-" * 50)

    print(f"\nDirect few-shot results ({len(direct_results)} variations):")
    for i, text in enumerate(direct_results):
        print(f"\n{i + 1}. {text}")
        print("-" * 50)

    # Create a pipeline that includes the few-shot augmenter along with other augmenters
    combined_pipeline = AugmentationPipeline(
        augmenters=[
            FewShotAugmenter(num_examples=2, n_augments=2),
            ContextAugmenter(n_augments=2),
            TextSurfaceAugmenter(n_augments=2)
        ],
        max_variations=10
    )

    # Apply the combined pipeline
    combined_results = combined_pipeline.augment(question, fewshot_data)

    print(f"\nGenerated {len(combined_results)} variations with few-shot + other augmenters:")

    for i, text in enumerate(combined_results):
        print(f"\n{i + 1}. {text}")
        print("-" * 50)




if __name__ == "__main__":
    # Run all examples
    # run_basic_augmentation_example()
    # run_shuffle_example()
    run_fewshot_combined_example()
