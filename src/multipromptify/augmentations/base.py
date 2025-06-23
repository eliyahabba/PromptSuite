from multipromptify.shared.constants import BaseAugmenterConstants, GenerationDefaults

class BaseAxisAugmenter:
    """
    Base class for all axis augmenters.
    
    Axis augmenters generate variations of a prompt along a specific dimension
    without changing the meaning of the prompt.
    """

    def __init__(self, n_augments=BaseAugmenterConstants.DEFAULT_N_AUGMENTS, seed=None):
        """
        Initialize the augmenter.
        
        Args:
            n_augments: Number of variations to generate (default from constants)
            seed: Random seed for reproducibility (default from GenerationDefaults)
        """
        self.n_augments = n_augments
        self.seed = seed if seed is not None else GenerationDefaults.RANDOM_SEED

    def get_name(self):
        """Get the name of this augmenter."""
        return self.__class__.__name__

    # def augment(self, prompt: str, identification_data: Dict[str, Any] = None) -> List[str]:
    #     """
    #     Generate variations of the prompt based on identification data.
    #
    #     Args:
    #         prompt: The original prompt text
    #         identification_data: Data from the identifier
    #
    #     Returns:
    #         List of variations of this axis
    #     """
    #     # Default implementation returns the original prompt
    #     return [prompt]
