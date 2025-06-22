# Non-semantic changes / structural changes (UNI TEXT)
import itertools
import random
import re
from typing import List

import numpy as np

from multipromptify.augmentations.base import BaseAxisAugmenter
from multipromptify.shared.constants import TextSurfaceAugmenterConstants
from multipromptify.core.template_keys import REWORDING, PROMPT_FORMAT


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
                for j in range(random.randint(
                        TextSurfaceAugmenterConstants.MIN_WHITESPACE_COUNT,
                        TextSurfaceAugmenterConstants.MAX_WHITESPACE_COUNT)):
                    new_value += TextSurfaceAugmenterConstants.WHITE_SPACE_OPTIONS[random.randint(
                        TextSurfaceAugmenterConstants.MIN_WHITESPACE_INDEX,
                        TextSurfaceAugmenterConstants.MAX_WHITESPACE_INDEX)]
            else:
                new_value += word
        
        # Restore placeholders if provided
        if placeholder_map:
            new_value = self._restore_placeholders(new_value, placeholder_map)
        
        return new_value

    def add_white_spaces(self, inputs, max_outputs=TextSurfaceAugmenterConstants.DEFAULT_MAX_OUTPUTS):
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
            protected_text, placeholder_map = self._protect_placeholders(inputs)
            
            augmented_input = []
            for i in range(max_outputs):
                augmented_text = self._add_white_spaces_to_single_text(protected_text, placeholder_map)
                augmented_input.append(augmented_text)
            return augmented_input

        # Handle list of texts
        augmented_texts = []
        for input_text in inputs:
            # Protect placeholders for each text
            protected_text, placeholder_map = self._protect_placeholders(input_text)
            
            augmented_input = []
            for i in range(max_outputs):
                # Apply augmentation
                cur_augmented_texts = self._add_white_spaces_to_single_text(protected_text, placeholder_map)
                augmented_input.append(cur_augmented_texts)
            augmented_texts.append(augmented_input)
        return augmented_texts

    def butter_finger(self, text, prob=TextSurfaceAugmenterConstants.DEFAULT_TYPO_PROB, keyboard="querty", seed=0,
                      max_outputs=TextSurfaceAugmenterConstants.DEFAULT_MAX_OUTPUTS):
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
        protected_text, placeholder_map = self._protect_placeholders(text)
        
        random.seed(seed)
        key_approx = TextSurfaceAugmenterConstants.QUERTY_KEYBOARD if keyboard == "querty" else {}

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
                    if random.choice(range(0, 100)) <= prob_of_typo:
                        new_letter = random.choice(key_approx[lcletter])
                    else:
                        new_letter = lcletter
                # go back to original case
                if not lcletter == letter:
                    new_letter = new_letter.upper()
                butter_text += new_letter
            
            # Restore placeholders
            restored_text = self._restore_placeholders(butter_text, placeholder_map)
            perturbed_texts.append(restored_text)
        return perturbed_texts

    def change_char_case(self, text, prob=TextSurfaceAugmenterConstants.DEFAULT_CASE_CHANGE_PROB, seed=0,
                         max_outputs=TextSurfaceAugmenterConstants.DEFAULT_MAX_OUTPUTS):
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
        protected_text, placeholder_map = self._protect_placeholders(text)
        
        random.seed(seed)
        results = []
        for _ in range(max_outputs):
            result = []
            for c in protected_text:
                if c.isupper() and random.random() < prob:
                    result.append(c.lower())
                elif c.islower() and random.random() < prob:
                    result.append(c.upper())
                else:
                    result.append(c)
            result = "".join(result)
            
            # Restore placeholders
            restored_text = self._restore_placeholders(result, placeholder_map)
            results.append(restored_text)
        return results


    def swap_characters(self, text, prob=TextSurfaceAugmenterConstants.DEFAULT_TYPO_PROB, seed=0,
                        max_outputs=TextSurfaceAugmenterConstants.DEFAULT_MAX_OUTPUTS):
        """
        Swaps characters in text, with probability prob for ang given pair.
        Ex: 'apple' -> 'aplpe'
        Placeholders in format {field_name} are protected during augmentation.
        Arguments:
            text (string): text to transform
            prob (float): probability of any two characters swapping. Default: 0.05
            seed (int): random seed
            max_outputs: Maximum number of augmented outputs.
            (taken from the NL-Augmenter project)
        """
        # Protect placeholders
        protected_text, placeholder_map = self._protect_placeholders(text)
        
        results = []
        for _ in range(max_outputs):
            max_seed = 2 ** 32
            # seed with hash so each text of same length gets different treatment.
            np.random.seed((seed + sum([ord(c) for c in protected_text])) % max_seed)
            # np.random.seed((seed) % max_seed).
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
            restored_text = self._restore_placeholders(swapped_text, placeholder_map)
            results.append(restored_text)
        return results

    def switch_punctuation(self, text, prob=TextSurfaceAugmenterConstants.DEFAULT_TYPO_PROB, seed=0, max_outputs=TextSurfaceAugmenterConstants.DEFAULT_MAX_OUTPUTS):
        """
        Switches punctuation in text with a probability of prob.
        Placeholders in format {field_name} are protected during augmentation.
        Arguments:
            text (string): text to transform
            prob (float): probability of any two characters switching. Default: 0.05
            seed (int): random seed
            max_outputs: Maximum number of augmented outputs.
        """
        # Protect placeholders
        protected_text, placeholder_map = self._protect_placeholders(text)
        
        results = []
        for _ in range(max_outputs):
            np.random.seed(seed)
            text_chars = list(protected_text)
            for i in range(len(text_chars)):
                if text_chars[i] in TextSurfaceAugmenterConstants.PUNCTUATION_MARKS and np.random.rand() < prob:
                    # Randomly select a different punctuation mark to switch with
                    new_punctuation = np.random.choice([p for p in TextSurfaceAugmenterConstants.PUNCTUATION_MARKS
                                                        if p != text_chars[i]])
                    text_chars[i] = new_punctuation
            
            # Restore placeholders
            modified_text = "".join(text_chars)
            restored_text = self._restore_placeholders(modified_text, placeholder_map)
            results.append(restored_text)
        return results

    def _protect_placeholders(self, text: str) -> tuple[str, dict]:
        """
        Replace placeholders with temporary tokens to protect them during augmentation.
        
        Args:
            text: Text that may contain placeholders like {field_name}
            
        Returns:
            Tuple of (protected_text, placeholder_map)
        """
        import re
        
        # Find all placeholders in format {field_name}
        placeholders = re.findall(r'\{[^}]+\}', text)
        placeholder_map = {}
        protected_text = text
        
        # Replace each placeholder with a simple number token that's unlikely to be corrupted
        for i, placeholder in enumerate(placeholders):
            # Use a simple numeric token to minimize corruption
            token = f"9999{i}9999"
            placeholder_map[token] = placeholder
            protected_text = protected_text.replace(placeholder, token)
        
        return protected_text, placeholder_map
    
    def _restore_placeholders(self, text: str, placeholder_map: dict) -> str:
        """
        Restore original placeholders from temporary tokens.
        
        Args:
            text: Text with temporary tokens
            placeholder_map: Mapping of tokens to original placeholders
            
        Returns:
            Text with original placeholders restored
        """
        restored_text = text
        for token, placeholder in placeholder_map.items():
            restored_text = restored_text.replace(token, placeholder)
        return restored_text

    def augment(self, text: str, techniques: List[str] = None) -> List[str]:
        """
        Apply text surface transformations to generate variations.
        Placeholders in format {field_name} are protected during augmentation.

        Args:
            text: The text to augment
            techniques: List of techniques to apply in sequence. If None, a default sequence will be used.
                Options: "typos", "capitalization", "spacing", "swap_characters", "punctuation"

        Returns:
            List of augmented texts including the original text
        """
        # Protect placeholders before augmentation
        protected_text, placeholder_map = self._protect_placeholders(text)
        
        # Default sequence if none provided
        if techniques is None:
            techniques = ["typos", "capitalization", "spacing", "swap_characters", "punctuation"]

        # Start with the original protected text
        variations = [protected_text]

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

        # Restore placeholders in all variations
        restored_variations = []
        for var in unique_variations:
            restored_var = self._restore_placeholders(var, placeholder_map)
            restored_variations.append(restored_var)

        # Ensure we return the requested number of variations
        if len(restored_variations) > self.n_augments:
            # Keep the original text and sample from the rest
            original = restored_variations[0]
            rest = restored_variations[1:]
            sampled = random.sample(rest, min(self.n_augments - 1, len(rest)))
            return [original] + sampled

        return restored_variations


if __name__ == "__main__":
    # Create the augmenter
    augmenter = TextSurfaceAugmenter(n_augments=5)

    # Example 1: Simple text with default sequence
    text1 = "This is a simple example of text surface augmentation."
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
    prompt_format = "Answer the following question"
    print(f"Original instruction template: {prompt_format}")
    
    # Test with surface variations - placeholders should remain intact
    variations3 = augmenter.augment(prompt_format, techniques=["typos", "capitalization"])
    print(f"\nGenerated {len(variations3)} variations with placeholder protection:")
    for i, variation in enumerate(variations3):
        if variation == prompt_format:
            print(f"\nVariation {i+1} (Original):")
        else:
            print(f"\nVariation {i+1}:")
        print(variation)
        print("-" * 50)
