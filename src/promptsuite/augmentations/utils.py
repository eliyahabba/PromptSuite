import random
import re
from typing import Callable, List, Set, Tuple, Dict


def protect_placeholders(text: str) -> Tuple[str, Dict[str, str]]:
    """
    Replace placeholders with temporary tokens to protect them during augmentation.
    
    Args:
        text: Text that may contain placeholders like {field_name}
        
    Returns:
        Tuple of (protected_text, placeholder_map)
    """
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


def restore_placeholders(text: str, placeholder_map: Dict[str, str]) -> str:
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


def random_composed_augmentations(
        text: str,
        transformations: List[Callable[[str], List[str]]],
        n_augments: int,
        rng: random.Random
) -> List[str]:
    """
    Generate n_augments variations by composing random subsets of transformations.
    Each variation is created by applying a random subset (1..N) of the transformations in random order.
    Args:
        text: The input text to augment
        transformations: List of transformation functions (each returns a list of variations)
        n_augments: Number of variations to generate
        rng: random.Random instance for reproducibility
    Returns:
        List of n_augments unique variations (including the original text)
    """
    variations: Set[str] = set()
    variations.add(text)
    attempts = 0
    max_attempts = n_augments * 5
    while len(variations) < n_augments and attempts < max_attempts:
        k = rng.randint(1, len(transformations))
        chosen = rng.sample(transformations, k)
        rng.shuffle(chosen)
        var = text
        for t in chosen:
            result = t(var)
            if isinstance(result, list) and result:
                var = result[-1]
            else:
                var = result
        variations.add(var)
        attempts += 1
    return list(variations)[:n_augments]
