"""
Text Noise Augmenter: Robustness testing with noise injection.

This augmenter introduces various types of noise to test model robustness
to noisy input, including typos, character swaps, punctuation changes, etc.
"""

import itertools
import random
import re
from typing import List

import numpy as np

from multipromptify.augmentations.base import BaseAxisAugmenter
from multipromptify.shared.constants import NoiseAugmenterConstants
from multipromptify.augmentations.utils import random_composed_augmentations, protect_placeholders, restore_placeholders


class TextNoiseAugmenter(BaseAxisAugmenter):
    """
    Augmenter for robustness testing with noise injection.
    Introduces various types of noise to test model robustness to noisy input.
    """

    def __init__(self, n_augments=3, seed=None):
        """
        Initialize the text noise augmenter.

        Args:
            n_augments: Number of variations to generate (also used for max combinations)
            seed: Random seed for reproducibility
        """
        super().__init__(n_augments=n_augments, seed=seed)
        self._rng = random.Random(self.seed)

    def _add_white_spaces_to_single_text(self, value, placeholder_map=None):
        """
        Add white spaces to the input text.
        If placeholder_map is provided, placeholders are already protected.

        Args:
            value: The input text to augment.
            placeholder_map: Optional mapping of placeholder tokens to restore

        Returns:
            Augmented text with added white spaces.
        """
        words = re.split(r"(\s+)", value)
        new_value = ""

        for word in words:
            if word.isspace():
                for j in range(self._rng.randint(
                        NoiseAugmenterConstants.MIN_WHITESPACE_COUNT,
                        NoiseAugmenterConstants.MAX_WHITESPACE_COUNT)):
                    new_value += NoiseAugmenterConstants.WHITE_SPACE_OPTIONS[self._rng.randint(
                        NoiseAugmenterConstants.MIN_WHITESPACE_INDEX,
                        NoiseAugmenterConstants.MAX_WHITESPACE_INDEX)]
            else:
                new_value += word
        
        # Restore placeholders if provided
        if placeholder_map:
            new_value = restore_placeholders(new_value, placeholder_map)
        
        return new_value

    def add_white_spaces(self, inputs, max_outputs=NoiseAugmenterConstants.DEFAULT_MAX_OUTPUTS):
        """
        Add white spaces to input text(s).
        Placeholders in format {field_name} are protected during augmentation.

        Args:
            inputs: Either a single text string or a list of input texts to augment.
            max_outputs: Maximum number of augmented outputs per input.

        Returns:
            If inputs is a string: List of augmented texts.
            If inputs is a list: List of lists of augmented texts.
        """
        # Handle single text input
        if isinstance(inputs, str):
            # Protect placeholders
            protected_text, placeholder_map = protect_placeholders(inputs)
            
            augmented_input = []
            for i in range(max_outputs):
                augmented_text = self._add_white_spaces_to_single_text(protected_text, placeholder_map)
                augmented_input.append(augmented_text)
            return augmented_input

        # Handle list of texts
        augmented_texts = []
        for input_text in inputs:
            # Protect placeholders for each text
            protected_text, placeholder_map = protect_placeholders(input_text)
            
            augmented_input = []
            for i in range(max_outputs):
                # Apply augmentation
                cur_augmented_texts = self._add_white_spaces_to_single_text(protected_text, placeholder_map)
                augmented_input.append(cur_augmented_texts)
            augmented_texts.append(augmented_input)
        return augmented_texts

    def butter_finger(self, text, prob=NoiseAugmenterConstants.DEFAULT_TYPO_PROB, keyboard="querty", seed=0,
                      max_outputs=NoiseAugmenterConstants.DEFAULT_MAX_OUTPUTS):
        """
        Introduce typos in the text by simulating butter fingers on a keyboard.
        Placeholders in format {field_name} are protected during augmentation.

        Args:
            text: Input text to augment.
            prob: Probability of introducing a typo for each character.
            keyboard: Keyboard layout to use.
            seed: Random seed for reproducibility.
            max_outputs: Maximum number of augmented outputs.

        Returns:
            List of texts with typos.
        """
        # Protect placeholders
        protected_text, placeholder_map = protect_placeholders(text)
        
        rng = random.Random(self.seed + seed)
        key_approx = NoiseAugmenterConstants.QUERTY_KEYBOARD if keyboard == "querty" else {}

        if not key_approx:
            print("Keyboard not supported.")
            return [text]

        prob_of_typo = int(prob * 100)
        perturbed_texts = []
        for _ in itertools.repeat(None, max_outputs):
            butter_text = ""
            for letter in protected_text:
                lcletter = letter.lower()
                if lcletter not in key_approx.keys():
                    new_letter = lcletter
                else:
                    if rng.choice(range(0, 100)) <= prob_of_typo:
                        new_letter = rng.choice(key_approx[lcletter])
                    else:
                        new_letter = lcletter
                # go back to original case
                if not lcletter == letter:
                    new_letter = new_letter.upper()
                butter_text += new_letter
            
            # Restore placeholders
            restored_text = restore_placeholders(butter_text, placeholder_map)
            perturbed_texts.append(restored_text)
        return perturbed_texts

    def change_char_case(self, text, prob=NoiseAugmenterConstants.DEFAULT_CASE_CHANGE_PROB, seed=0,
                         max_outputs=NoiseAugmenterConstants.DEFAULT_MAX_OUTPUTS):
        """
        Change the case of characters in the text.
        Placeholders in format {field_name} are protected during augmentation.

        Args:
            text: Input text to augment.
            prob: Probability of changing the case of each character.
            seed: Random seed for reproducibility.
            max_outputs: Maximum number of augmented outputs.

        Returns:
            List of texts with modified character cases.
        """
        # Protect placeholders
        protected_text, placeholder_map = protect_placeholders(text)
        
        rng = np.random.default_rng(self.seed + seed)
        results = []
        for _ in range(max_outputs):
            result = []
            for c in protected_text:
                if c.isupper() and rng.random() < prob:
                    result.append(c.lower())
                elif c.islower() and rng.random() < prob:
                    result.append(c.upper())
                else:
                    result.append(c)
            result = "".join(result)
            
            # Restore placeholders
            restored_text = restore_placeholders(result, placeholder_map)
            results.append(restored_text)
        return results

    def swap_characters(self, text, prob=NoiseAugmenterConstants.DEFAULT_TYPO_PROB, seed=0,
                        max_outputs=NoiseAugmenterConstants.DEFAULT_MAX_OUTPUTS):
        """
        Swaps characters in text, with probability prob for any given pair.
        Ex: 'apple' -> 'aplpe'
        Placeholders in format {field_name} are protected during augmentation.
        
        Args:
            text: Text to transform
            prob: Probability of any two characters swapping. Default: 0.05
            seed: Random seed
            max_outputs: Maximum number of augmented outputs.
            (taken from the NL-Augmenter project)
        """
        # Protect placeholders
        protected_text, placeholder_map = protect_placeholders(text)
        
        results = []
        for _ in range(max_outputs):
            max_seed = 2 ** 32
            # seed with hash so each text of same length gets different treatment.
            np.random.seed((self.seed + seed + sum([ord(c) for c in protected_text])) % max_seed)
            # number of possible characters to swap.
            num_pairs = len(protected_text) - 1
            # if no pairs, do nothing
            if num_pairs < 1:
                return [text]  # Return original text as list
            # get indices to swap.
            indices_to_swap = np.argwhere(
                np.random.rand(num_pairs) < prob
            ).reshape(-1)
            # shuffle swapping order, may matter if there are adjacent swaps.
            np.random.shuffle(indices_to_swap)
            # convert to list.
            text_list = list(protected_text)
            # swap.
            for index in indices_to_swap:
                text_list[index], text_list[index + 1] = text_list[index + 1], text_list[index]
            # convert to string.
            swapped_text = "".join(text_list)
            
            # Restore placeholders
            restored_text = restore_placeholders(swapped_text, placeholder_map)
            results.append(restored_text)
        return results

    def switch_punctuation(self, text, prob=NoiseAugmenterConstants.DEFAULT_TYPO_PROB, seed=0,
                           max_outputs=NoiseAugmenterConstants.DEFAULT_MAX_OUTPUTS):
        """
        Switches punctuation in text with a probability of prob.
        Placeholders in format {field_name} are protected during augmentation.
        
        Args:
            text: Text to transform
            prob: Probability of any two characters switching. Default: 0.05
            seed: Random seed
            max_outputs: Maximum number of augmented outputs.
        """
        # Protect placeholders
        protected_text, placeholder_map = protect_placeholders(text)
        
        results = []
        for _ in range(max_outputs):
            np.random.seed(self.seed + seed)
            text_chars = list(protected_text)
            for i in range(len(text_chars)):
                if text_chars[i] in NoiseAugmenterConstants.PUNCTUATION_MARKS and np.random.rand() < prob:
                    # Randomly select a different punctuation mark to switch with
                    new_punctuation = np.random.choice([p for p in NoiseAugmenterConstants.PUNCTUATION_MARKS
                                                        if p != text_chars[i]])
                    text_chars[i] = new_punctuation
            
            # Restore placeholders
            modified_text = "".join(text_chars)
            restored_text = restore_placeholders(modified_text, placeholder_map)
            results.append(restored_text)
        return results

    def augment(self, text: str, techniques: List[str] = None) -> List[str]:
        """
        Apply text noise transformations to generate variations.
        Placeholders in format {field_name} are protected during augmentation.

        Args:
            text: The text to augment
            techniques: List of techniques to apply in sequence. If None, a default sequence will be used.
                Options: "typos", "capitalization", "spacing", "swap_characters", "punctuation"

        Returns:
            List of augmented texts including the original text
        """
        # Protect placeholders before augmentation
        protected_text, placeholder_map = protect_placeholders(text)
        
        # Default sequence if none provided
        if techniques is None:
            techniques = ["typos", "capitalization", "spacing", "swap_characters", "punctuation"]
        
        # Map technique names to functions
        technique_map = {
            "typos": lambda t: self.butter_finger(t, prob=0.05, max_outputs=1),
            "capitalization": lambda t: self.change_char_case(t, prob=0.15, max_outputs=1),
            "spacing": lambda t: self.add_white_spaces(t, max_outputs=1),
            "swap_characters": lambda t: self.swap_characters(t, max_outputs=1),
            "punctuation": lambda t: self.switch_punctuation(t, max_outputs=1),
        }
        transformations = [technique_map[name] for name in techniques if name in technique_map]
        
        variations = random_composed_augmentations(
            protected_text,
            transformations,
            self.n_augments,
            self._rng
        )
        # Restore placeholders in all variations
        restored_variations = [restore_placeholders(var, placeholder_map) for var in variations]
        return restored_variations


if __name__ == "__main__":
    # Create the augmenter
    augmenter = TextNoiseAugmenter(n_augments=5)

    # Example 1: Simple text with default sequence
    text1 = "This is a simple example of text noise augmentation."
    variations1 = augmenter.augment(text1)

    print(f"Original text: {text1}")
    print(f"\nGenerated {len(variations1)} variations with default sequence:")
    for i, variation in enumerate(variations1):
        if variation == text1:
            print(f"\nVariation {i+1} (Original):")
        else:
            print(f"\nVariation {i+1}:")
        print(variation)
        print("-" * 50)

    # Example 2: Custom sequence
    text2 = "What is the capital of France? Paris is the correct answer."
    variations2 = augmenter.augment(text2, techniques=["spacing", "typos"])

    print(f"\nOriginal text: {text2}")
    print(f"\nGenerated {len(variations2)} variations with custom sequence (spacing → typos):")
    for i, variation in enumerate(variations2):
        if variation == text2:
            print(f"\nVariation {i+1} (Original):")
        else:
            print(f"\nVariation {i+1}:")
        print(variation)
        print("-" * 50)

    # Example 3: Individual transformations
    print("\nIndividual transformations:")
    print(f"Original: {text1}")
    print(f"With typos: {augmenter.butter_finger(text1, prob=0.1, max_outputs=1)[0]}")
    print(f"With capitalization changes: {augmenter.change_char_case(text1, prob=0.15, max_outputs=1)[0]}")
    print(f"With spacing changes: {augmenter.add_white_spaces(text1, max_outputs=1)[0]}")
    print(f"With character swaps: {augmenter.swap_characters(text1, prob=0.08, max_outputs=1)[0]}")

    # Example 4: Placeholder protection test
    print("\nPlaceholder protection test:")
    prompt_format = "Question: {question} Answer: {answer}"
    print(f"Original prompt template: {prompt_format}")
    
    # Test with noise variations - placeholders should remain intact
    variations3 = augmenter.augment(prompt_format, techniques=["typos", "capitalization"])
    print(f"\nGenerated {len(variations3)} variations with placeholder protection:")
    for i, variation in enumerate(variations3):
        if variation == prompt_format:
            print(f"\nVariation {i+1} (Original):")
        else:
            print(f"\nVariation {i+1}:")
        print(variation)
        print("-" * 50) 