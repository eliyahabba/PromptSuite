"""
Augmentation pipeline that combines multiple augmentation methods.
"""
import random
from typing import List, Optional, Dict, Any

from src.augmentations.base_augmenter import BaseAxisAugmenter
from src.augmentations.context_augmenter import ContextAugmenter
from src.augmentations.fewshot_augmenter import FewShotAugmenter
from src.augmentations.multidoc_augmenter import MultiDocAugmenter
from src.augmentations.multiple_choice_augmenter import MultipleChoiceAugmenter
from src.augmentations.paraphrase_instruct import Paraphrase
from src.augmentations.text_surface_augmenter import TextSurfaceAugmenter
from src.shared.constants import AugmentationPipelineConstants, BaseAugmenterConstants


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
        elif isinstance(augmenter, MultipleChoiceAugmenter) and identification_data:
            return augmenter.augment(text, identification_data)
        elif isinstance(augmenter, FewShotAugmenter):
            # If we have example pairs in identification_data, use them
            if identification_data:
                return augmenter.augment(text, identification_data)
            # Otherwise return the original text
            return [text]
        elif isinstance(augmenter, MultiDocAugmenter):
            # MultiDocAugmenter works with lists of documents
            if identification_data and "docs" in identification_data:
                # Get the documents from identification_data
                docs = identification_data["docs"]
                # Get the concatenation type if provided
                concat_type = identification_data.get("concat_type", "single_doc")
                # Generate permutations
                permutations = augmenter.permute_docs_order(docs, n_permutations=augmenter.n_augments)
                # Concatenate each permutation
                return [augmenter.concatenate_docs(perm, concat_type) for perm in permutations]
            # If no documents are provided, return the original text
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
                return augmenter.augment(text, identification_data)
            except TypeError:
                # Try without identification_data if it fails
                try:
                    return augmenter.augment(text)
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


def run_multiple_choice_example():
    """
    Run an example with multiple choice augmentation.
    """
    print("\n\n--- Multiple Choice Example ---")

    mc_augmenter = MultipleChoiceAugmenter(n_augments=3)
    mc_text = "What is the capital of France?\n\nA) Paris\nB) London\nC) Berlin\nD) Madrid"

    # Identification data for the multiple choice question
    mc_data = {
        "question": "What is the capital of France?",
        "options": ["Paris", "London", "Berlin", "Madrid"],
        "markers": ["A", "B", "C", "D"]
    }

    # Create a pipeline with just the multiple choice augmenter
    mc_pipeline = AugmentationPipeline(augmenters=[mc_augmenter], max_variations=10)
    mc_variations = mc_pipeline.augment(mc_text, mc_data)

    print(f"\nOriginal text: {mc_text}")
    print(f"\nGenerated {len(mc_variations)} variations:")

    for i, text in enumerate(mc_variations):
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


def run_multidoc_combined_example():
    """
    Run an example that combines multi-document augmentation with other augmenters.
    """
    print("\n\n--- Multi-Document Combined Example ---")

    # Create sample documents
    docs = [
        "The sun is the star at the center of the Solar System.",
        "Earth is the third planet from the Sun and the only astronomical object known to harbor life.",
        "The Moon is Earth's only natural satellite."
    ]

    # Create a pipeline that combines multi-document with other augmenters
    combined_pipeline = AugmentationPipeline(
        augmenters=[
            MultiDocAugmenter(n_augments=2),
            Paraphrase(n_augments=2),
            TextSurfaceAugmenter(n_augments=2)
        ],
        max_variations=10
    )

    # Create identification data with the documents
    multidoc_data = {
        "docs": docs,
        "concat_type": "2_newlines"
    }

    # Display the original documents
    print(f"\nOriginal documents:")
    for i, doc in enumerate(docs):
        print(f"{i + 1}. {doc}")
    print("-" * 50)

    # Apply the combined pipeline
    combined_results = combined_pipeline.augment("", multidoc_data)  # Empty string as the base text

    print(f"\nGenerated {len(combined_results)} variations with multi-doc + other augmenters:")

    for i, text in enumerate(combined_results):
        print(f"\n{i + 1}. {text[:200]}...")  # Show first 200 chars
        print("-" * 50)


if __name__ == "__main__":
    # Run all examples
    # run_basic_augmentation_example()
    # run_multiple_choice_example()
    run_fewshot_combined_example()
    # run_multidoc_combined_example()
