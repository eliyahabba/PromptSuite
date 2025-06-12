"""Constants for the Multi-Prompt Evaluation Tool."""
from pathlib import Path

# Model configuration
DEFAULT_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"

# Default number of variations to generate per axis
DEFAULT_VARIATIONS_PER_AXIS = 3

# Minimum and maximum number of variations per axis
MIN_VARIATIONS_PER_AXIS = 1
MAX_VARIATIONS_PER_AXIS = 10

# Base augmenter constants
class BaseAugmenterConstants:
    # Default number of augmentations to generate
    DEFAULT_N_AUGMENTS = 3

# Augmentation pipeline constants
class AugmentationPipelineConstants:
    # Default maximum number of variations to generate
    DEFAULT_MAX_VARIATIONS = 100

# Constants for MultipleChoiceAugmenter
class MultipleChoiceConstants:
    # Enumeration styles for multiple choice options
    ENUMERATION_STYLES = [
        ["A", "B", "C", "D"],  # uppercase letters
        ["a", "b", "c", "d"],  # lowercase letters
        ["1", "2", "3", "4"],  # numbers
        ["A)", "B)", "C)", "D)"],  # uppercase with bracket
        ["a)", "b)", "c)", "d)"],  # lowercase with bracket
        ["1)", "2)", "3)", "4)"],  # numbers with bracket
    ]

# Constants for MultiDocAugmenter
class MultiDocConstants:
    # Concatenation types
    SINGLE_DOC = "single_doc"
    DOUBLE_NEWLINES = "2_newlines"
    TITLES = "titles"
    DASHES = "dashes"
    
    # Default separator for dashes
    DEFAULT_SEPARATOR_LENGTH = 20
    
    # Document title format
    DOC_TITLE_FORMAT = "Document {}: "
    
    # Default values for augmentation
    DEFAULT_N_NEW_DOCS = 3
    DEFAULT_N_PERMUTATIONS = 3

# Constants for FewShotAugmenter
class FewShotConstants:
    # Format strings for examples
    EXAMPLE_FORMAT = "Input: {}\nOutput: {}"
    QUESTION_FORMAT = "Input: {}\nOutput:"
    
    # Separator between examples
    EXAMPLE_SEPARATOR = "\n\n"
    
    # Default random seed for sampling
    DEFAULT_RANDOM_SEED = 42
    
    # Default number of examples to include
    DEFAULT_NUM_EXAMPLES = 1

# Constants for NonLLMAugmenter
class TextSurfaceAugmenterConstants:
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

# Directory where data files are located
DATA_DIR_NAME = "data"
DATA_DIR = Path(__file__).resolve().parents[2] / DATA_DIR_NAME
# Default input file for annotations
DEFAULT_ANNOTATIONS_INPUT_FILE = f"{DATA_DIR}/annotations_input.json"

# Default output file for augmented variations
DEFAULT_AUGMENTED_VARIATIONS_OUTPUT_FILE = f"{DATA_DIR}/augmented_variations_output.json"

