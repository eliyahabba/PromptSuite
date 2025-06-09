# Non-semantic changes / structural changes (UNI TEXT)
import itertools
import random
import re
from typing import List

import numpy as np

from src.axis_augmentation.base_augmenter import BaseAxisAugmenter
from src.utils.constants import TextSurfaceAugmenterConstants


class TextSurfaceAugmenter(BaseAxisAugmenter):
    """
    Augmenter that creates variations of prompts using non-LLM techniques.
    This includes simple transformations like adding typos, changing capitalization, etc.
    """

    def __init__(self, n_augments=3):
        """
        Initialize the non-LLM augmenter.

        Args:
            n_augments: Number of variations to generate
        """
        super().__init__(n_augments=n_augments)

    def _add_white_spaces_to_single_text(self, value):
        """
        Add white spaces to the input text.

        Args:
            value: The input text to augment.

        Returns:
            Augmented text with added white spaces.
        """
        words = re.split(r"(\s+)", value)
        new_value = ""

        for word in words:
            if word.isspace():
                for j in range(random.randint(
                        TextSurfaceAugmenterConstants.MIN_WHITESPACE_COUNT,
                        TextSurfaceAugmenterConstants.MAX_WHITESPACE_COUNT)):
                    new_value += TextSurfaceAugmenterConstants.WHITE_SPACE_OPTIONS[random.randint(
                        TextSurfaceAugmenterConstants.MIN_WHITESPACE_INDEX,
                        TextSurfaceAugmenterConstants.MAX_WHITESPACE_INDEX)]
            else:
                new_value += word
        return new_value

    def add_white_spaces(self, inputs, max_outputs=TextSurfaceAugmenterConstants.DEFAULT_MAX_OUTPUTS):
        """
        Add white spaces to input text(s).

        Args:
            inputs: Either a single text string or a list of input texts to augment.
            max_outputs: Maximum number of augmented outputs per input.

        Returns:
            If inputs is a string: List of augmented texts.
            If inputs is a list: List of lists of augmented texts.
        """
        # Handle single text input
        if isinstance(inputs, str):
            augmented_input = []
            for i in range(max_outputs):
                augmented_text = self._add_white_spaces_to_single_text(inputs)
                augmented_input.append(augmented_text)
            return augmented_input

        # Handle list of texts
        augmented_texts = []
        for input_text in inputs:
            augmented_input = []
            for i in range(max_outputs):
                # Apply augmentation
                cur_augmented_texts = self._add_white_spaces_to_single_text(input_text)
                augmented_input.append(cur_augmented_texts)
            augmented_texts.append(augmented_input)
        return augmented_texts

    def butter_finger(self, text, prob=TextSurfaceAugmenterConstants.DEFAULT_TYPO_PROB, keyboard="querty", seed=0,
                      max_outputs=TextSurfaceAugmenterConstants.DEFAULT_MAX_OUTPUTS):
        """
        Introduce typos in the text by simulating butter fingers on a keyboard.

        Args:
            text: Input text to augment.
            prob: Probability of introducing a typo for each character.
            keyboard: Keyboard layout to use.
            seed: Random seed for reproducibility.
            max_outputs: Maximum number of augmented outputs.

        Returns:
            List of texts with typos.
        """
        random.seed(seed)
        key_approx = TextSurfaceAugmenterConstants.QUERTY_KEYBOARD if keyboard == "querty" else {}

        if not key_approx:
            print("Keyboard not supported.")
            return [text]

        prob_of_typo = int(prob * 100)
        perturbed_texts = []
        for _ in itertools.repeat(None, max_outputs):
            butter_text = ""
            for letter in text:
                lcletter = letter.lower()
                if lcletter not in key_approx.keys():
                    new_letter = lcletter
                else:
                    if random.choice(range(0, 100)) <= prob_of_typo:
                        new_letter = random.choice(key_approx[lcletter])
                    else:
                        new_letter = lcletter
                # go back to original case
                if not lcletter == letter:
                    new_letter = new_letter.upper()
                butter_text += new_letter
            perturbed_texts.append(butter_text)
        return perturbed_texts

    def change_char_case(self, text, prob=TextSurfaceAugmenterConstants.DEFAULT_CASE_CHANGE_PROB, seed=0,
                         max_outputs=TextSurfaceAugmenterConstants.DEFAULT_MAX_OUTPUTS):
        """
        Change the case of characters in the text.

        Args:
            text: Input text to augment.
            prob: Probability of changing the case of each character.
            seed: Random seed for reproducibility.
            max_outputs: Maximum number of augmented outputs.

        Returns:
            List of texts with modified character cases.
        """
        random.seed(seed)
        results = []
        for _ in range(max_outputs):
            result = []
            for c in text:
                if c.isupper() and random.random() < prob:
                    result.append(c.lower())
                elif c.islower() and random.random() < prob:
                    result.append(c.upper())
                else:
                    result.append(c)
            result = "".join(result)
            results.append(result)
        return results


    def swap_characters(self, text, prob=TextSurfaceAugmenterConstants.DEFAULT_TYPO_PROB, seed=0,
                        max_outputs=TextSurfaceAugmenterConstants.DEFAULT_MAX_OUTPUTS):
        """
        Swaps characters in text, with probability prob for ang given pair.
        Ex: 'apple' -> 'aplpe'
        Arguments:
            text (string): text to transform
            prob (float): probability of any two characters swapping. Default: 0.05
            seed (int): random seed
            max_outputs: Maximum number of augmented outputs.
            (taken from the NL-Augmenter project)
        """
        results = []
        for _ in range(max_outputs):
            max_seed = 2 ** 32
            # seed with hash so each text of same length gets different treatment.
            np.random.seed((seed + sum([ord(c) for c in text])) % max_seed)
            # np.random.seed((seed) % max_seed).
            # number of possible characters to swap.
            num_pairs = len(text) - 1
            # if no pairs, do nothing
            if num_pairs < 1:
                return text
            # get indices to swap.
            indices_to_swap = np.argwhere(
                np.random.rand(num_pairs) < prob
            ).reshape(-1)
            # shuffle swapping order, may matter if there are adjacent swaps.
            np.random.shuffle(indices_to_swap)
            # convert to list.
            text = list(text)
            # swap.
            for index in indices_to_swap:
                text[index], text[index + 1] = text[index + 1], text[index]
            # convert to string.
            text = "".join(text)
            results.append(text)
        return results

    def switch_punctuation(self, text, prob=TextSurfaceAugmenterConstants.DEFAULT_TYPO_PROB, seed=0, max_outputs=TextSurfaceAugmenterConstants.DEFAULT_MAX_OUTPUTS):
        """
        Switches punctuation in text with a probability of prob.
        Arguments:
            text (string): text to transform
            prob (float): probability of any two characters switching. Default: 0.05
            seed (int): random seed
            max_outputs: Maximum number of augmented outputs.
        """
        results = []
        for _ in range(max_outputs):
            np.random.seed(seed)
            text_chars = list(text)
            for i in range(len(text_chars)):
                if text_chars[i] in TextSurfaceAugmenterConstants.PUNCTUATION_MARKS and np.random.rand() < prob:
                    # Randomly select a different punctuation mark to switch with
                    new_punctuation = np.random.choice([p for p in TextSurfaceAugmenterConstants.PUNCTUATION_MARKS
                                                        if p != text_chars[i]])
                    text_chars[i] = new_punctuation
            results.append("".join(text_chars))
        return results

    def augment(self, text: str, techniques: List[str] = None) -> List[str]:
        """
        Apply text surface transformations to generate variations.

        Args:
            text: The text to augment
            techniques: List of techniques to apply in sequence. If None, a default sequence will be used.
                Options: "typos", "capitalization", "spacing", "swap_characters", "punctuation"

        Returns:
            List of augmented texts including the original text
        """
        # Default sequence if none provided
        if techniques is None:
            techniques = ["typos", "capitalization", "spacing", "swap_characters", "punctuation"]

        # Start with the original text
        variations = [text]

        # Apply each technique in sequence
        for technique in techniques:
            new_variations = []

            # Always keep the original variations
            new_variations.extend(variations)

            # For each existing variation, apply the current technique
            for variation in variations:
                if technique == "typos":
                    # Add typo variations
                    typo_results = self.butter_finger(variation, prob=0.1, max_outputs=2)
                    new_variations.extend(typo_results)
                elif technique == "capitalization":
                    # Add case variations
                    case_results = self.change_char_case(variation, prob=0.15, max_outputs=2)
                    new_variations.extend(case_results)
                elif technique == "spacing":
                    # Add spacing variations
                    spacing_results = self.add_white_spaces(variation, max_outputs=2)
                    new_variations.extend(spacing_results)
                elif technique == "swap_characters":
                    # Add character swap variations
                    swap_results = self.swap_characters(variation, max_outputs=2)
                    new_variations.extend(swap_results)
                elif technique == "punctuation":
                    # Add punctuation variations
                    punctuation_results = self.switch_punctuation(variation, max_outputs=2)
                    new_variations.extend(punctuation_results)

            # Update variations for the next technique
            variations = new_variations

            # If we already have enough variations, we can stop
            if len(variations) >= self.n_augments:
                break

        # Remove duplicates while preserving order
        unique_variations = []
        for var in variations:
            if var not in unique_variations:
                unique_variations.append(var)

        # Ensure we return the requested number of variations
        if len(unique_variations) > self.n_augments:
            # Keep the original text and sample from the rest
            original = unique_variations[0]
            rest = unique_variations[1:]
            sampled = random.sample(rest, min(self.n_augments - 1, len(rest)))
            return [original] + sampled

        return unique_variations


if __name__ == "__main__":
    # Create the augmenter
    augmenter = TextSurfaceAugmenter(n_augments=5)

    # Example 1: Simple text with default sequence
    text1 = "This is a simple example of text surface augmentation."
    text1_1 = "This, is a simple example: Text surface augmentation."
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
    print(f"With punctuation changes: {augmenter.switch_punctuation(text1_1, prob=0.9, max_outputs=1)[0]}")
