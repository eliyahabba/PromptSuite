"""Constants for the Multi-Prompt Evaluation Tool."""


# Default number of variations to generate per axis
class GenerationInterfaceConstants:
    # Generation limits
    MIN_VARIATIONS = 1
    MAX_VARIATIONS = 1000
    MIN_VARIATIONS_PER_FIELD = 1
    MAX_VARIATIONS_PER_FIELD = 10


# Base augmenter constants
class BaseAugmenterConstants:
    # Default number of augmentations to generate
    DEFAULT_N_AUGMENTS = 3


# Constants for ShuffleAugmenter
class ShuffleConstants:
    # Default number of shuffle variations to generate
    DEFAULT_N_SHUFFLES = 3

    # Supported list formats for parsing
    SUPPORTED_FORMATS = [
        "json",  # ["item1", "item2", "item3"]
        "multiple_choice",  # "A) item1 B) item2 C) item3"
        "comma_separated",  # "item1, item2, item3"
        "newline_separated"  # "item1\nitem2\nitem3"
    ]


# Constants for FewShotAugmenter
class FewShotConstants:
    # Format strings for examples
    EXAMPLE_FORMAT = "Input: {}\nOutput: {}"
    INPUT_FORMAT = "Input: {}\nOutput:"

    # Separator between examples
    EXAMPLE_SEPARATOR = "\n\n"

    # Default random seed for sampling
    DEFAULT_RANDOM_SEED = 42

    # Default number of examples to include
    DEFAULT_NUM_EXAMPLES = 1


# Constants for NonLLMAugmenter
class NoiseAugmenterConstants:
    # White space options
    WHITE_SPACE_OPTIONS = ["\n", "\t", " ", ""]

    # Keyboard layout for butter finger
    QUERTY_KEYBOARD = {
        "q": "qwasedzx",
        "w": "wqesadrfcx",
        "e": "ewrsfdqazxcvgt",
        "r": "retdgfwsxcvbnju",
        "t": "tryfhgedcvbnju",
        "y": "ytugjhrfvbnji",
        "u": "uyihkjtgbnmlo",
        "i": "iuojlkyhnmlp",
        "o": "oipklujm",
        "p": "plo['ik",
        "a": "aqszwxwdce",
        "s": "swxadrfv",
        "d": "decsfaqgbv",
        "f": "fdgrvwsxyhn",
        "g": "gtbfhedcyjn",
        "h": "hyngjfrvkim",
        "j": "jhknugtblom",
        "k": "kjlinyhn",
        "l": "lokmpujn",
        "z": "zaxsvde",
        "x": "xzcsdbvfrewq",
        "c": "cxvdfzswergb",
        "v": "vcfbgxdertyn",
        "b": "bvnghcftyun",
        "n": "nbmhjvgtuik",
        "m": "mnkjloik",
        " ": " "
    }

    PUNCTUATION_MARKS = [".", ",", "!", "?", ";", ":", "-", "_"]

    # Default probabilities
    DEFAULT_TYPO_PROB = 0.05
    DEFAULT_CASE_CHANGE_PROB = 0.1

    # Default max outputs
    DEFAULT_MAX_OUTPUTS = 1

    # Random ranges for white space generation
    MIN_WHITESPACE_COUNT = 1
    MAX_WHITESPACE_COUNT = 3

    # Random index range for white space options
    MIN_WHITESPACE_INDEX = 0
    MAX_WHITESPACE_INDEX = 2

    # Transformation techniques
    TRANSFORMATION_TECHNIQUES = ["typos", "capitalization", "punctuation", "spacing"]


class GenerationDefaults:
    """Centralized defaults for generation parameters across API, CLI, and UI."""
    MAX_VARIATIONS_PER_ROW = None  # None means no limit on variations
    MAX_ROWS = None  # None means use all rows
    VARIATIONS_PER_FIELD = 3
    MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
    API_PLATFORM = "TogetherAI"
    RANDOM_SEED = 42


# Few-shot dynamic default (used in template builder UI)
FEW_SHOT_DYNAMIC_DEFAULT = lambda available_rows: min(2, max(0, available_rows - 1)) if available_rows > 1 else 0

# List formatting constants
class ListFormattingConstants:
    """Constants for formatting lists in prompts."""
    # Default separator for list items when displaying in prompts
    DEFAULT_LIST_SEPARATOR = "\n"
