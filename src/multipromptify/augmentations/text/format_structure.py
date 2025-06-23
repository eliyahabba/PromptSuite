"""
Format Structure Augmenter: Semantic-preserving format variations.

This augmenter follows the FORMATSPREAD paper approach to create format variations
that preserve semantic meaning while changing the structural presentation of prompts.
"""

import random
import re
from typing import List

from multipromptify.augmentations.base import BaseAxisAugmenter
from multipromptify.augmentations.utils import random_composed_augmentations, protect_placeholders, restore_placeholders


class FormatStructureAugmenter(BaseAxisAugmenter):
    """
    Augmenter for format structure variations following FORMATSPREAD paper approach.
    Preserves semantic meaning while changing format structure.
    """
    
    # Separators between descriptor and content (S1 in paper)
    SEPARATORS = [':', '::', ':::', ' - ', ' -- ', ': ', ' : ', '\n', '\t', ' || ', ', ']
    
    # Connectors between fields (C in paper)
    FIELD_CONNECTORS = [' ', '\n', '\n\n', '\t', ' || ', ' -- ', '  ', '; ', '\n\t']
    
    # Casing functions for descriptors only
    CASING_FUNCTIONS = {
        'uppercase': str.upper,
        'lowercase': str.lower,
        'title': str.title,
        'capitalize': str.capitalize
    }
    
    def __init__(self, n_augments=5, seed=None):
        super().__init__(n_augments=n_augments, seed=seed)
        self._rng = random.Random(self.seed)
    
    def change_separators(self, text: str) -> List[str]:
        """
        Change separators between descriptors and their values.
        
        Examples:
            "Question: {text} Answer: {answer}" → "Question :: {text} Answer :: {answer}"
            "Input: {x} Output: {y}" → "Input - {x} Output - {y}"
            "Passage: {text}" → "Passage\t{text}"
        """
        variations = [text]
        
        # Find all patterns of "Word: " (descriptor followed by separator)
        pattern = r'(\b[A-Za-z]+)(:\s*)'
        
        # Randomly select separators instead of using the first ones
        selected_separators = self._rng.sample(self.SEPARATORS, min(len(self.SEPARATORS), self.n_augments-1))
        
        for separator in selected_separators:
            new_text = re.sub(pattern, lambda m: m.group(1) + separator, text)
            if new_text != text and new_text not in variations:
                variations.append(new_text)
        
        return variations[:self.n_augments]
    
    def change_field_connectors(self, text: str) -> List[str]:
        """
        Change connectors between different fields.
        
        Examples:
            "Question: {q} Answer: {a}" → "Question: {q}\nAnswer: {a}"
            "Input: {x} Output: {y}" → "Input: {x} || Output: {y}"
            "Context: {c} Question: {q}" → "Context: {c}\n\nQuestion: {q}"
        """
        variations = [text]
        
        # Pattern to find field boundaries: "} Word:"
        pattern = r'(\})\s+([A-Z][a-z]+\s*:)'
        
        # Randomly select connectors instead of using the first ones
        selected_connectors = self._rng.sample(self.FIELD_CONNECTORS, min(len(self.FIELD_CONNECTORS), self.n_augments-1))
        
        for connector in selected_connectors:
            new_text = re.sub(pattern, rf'\1{connector}\2', text)
            if new_text != text and new_text not in variations:
                variations.append(new_text)
        
        return variations[:self.n_augments]
    
    def apply_descriptor_casing(self, text: str) -> List[str]:
        """
        Apply casing changes only to descriptors (field names).
        
        Examples:
            "Question: {text} Answer: {answer}" → "QUESTION: {text} ANSWER: {answer}"
            "Input: {x} Output: {y}" → "input: {x} output: {y}"
            "Passage: {p} Question: {q}" → "Passage: {p} Question: {q}" (title case)
        """
        variations = [text]
        
        # Find all descriptors (words before colons)
        pattern = r'\b([A-Za-z]+)(\s*:)'
        
        # Randomly select casing functions instead of using the first ones
        selected_casings = self._rng.sample(list(self.CASING_FUNCTIONS.items()), 
                                       min(len(self.CASING_FUNCTIONS), self.n_augments-1))
        
        for case_name, case_func in selected_casings:
            def replace_func(match):
                descriptor = match.group(1)
                separator = match.group(2)
                return case_func(descriptor) + separator
            
            new_text = re.sub(pattern, replace_func, text)
            if new_text != text and new_text not in variations:
                variations.append(new_text)
        
        return variations[:self.n_augments]
    
    def remove_separators(self, text: str) -> List[str]:
        """
        Remove separators entirely for more compact format.
        
        Examples:
            "Question: {text}" → "Question {text}"
            "Input: {x} Output: {y}" → "Input {x} Output {y}"
        """
        variations = [text]
        
        # Remove colons and following spaces
        pattern = r'(\b[A-Za-z]+):\s*'
        new_text = re.sub(pattern, r'\1 ', text)
        
        if new_text != text:
            variations.append(new_text)
        
        return variations
    
    def augment(self, text: str) -> List[str]:
        """
        Generate format variations preserving semantic meaning.
        
        Examples:
            Input: "Question: {text} Answer: {answer}"
            Output variations:
                - "Question :: {text} Answer :: {answer}"
                - "QUESTION: {text} ANSWER: {answer}"
                - "Question: {text}\nAnswer: {answer}"
                - "question: {text} answer: {answer}"
                - "Question {text} Answer {answer}"
        """
        # Protect placeholders before augmentation
        protected_text, placeholder_map = protect_placeholders(text)
        
        # Use the shared utility for random composed augmentations
        transformations = [
            self.change_separators,
            self.apply_descriptor_casing,
            self.change_field_connectors,
            self.remove_separators
        ]
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
    # Test the FormatStructureAugmenter
    augmenter = FormatStructureAugmenter(n_augments=5)
    
    # Test with a simple prompt format
    test_text = "Question: {question} Answer: {answer}"
    print(f"Original: {test_text}")
    
    variations = augmenter.augment(test_text)
    print(f"\nGenerated {len(variations)} format structure variations:")
    for i, var in enumerate(variations):
        print(f"{i+1}. {var}")
    
    # Test with more complex format
    complex_text = "Context: {context} Question: {question} Options: {options} Answer: {answer}"
    print(f"\n\nOriginal: {complex_text}")
    
    variations2 = augmenter.augment(complex_text)
    print(f"\nGenerated {len(variations2)} format structure variations:")
    for i, var in enumerate(variations2):
        print(f"{i+1}. {var}") 