# Constants for MultiPromptify pipeline
VARIATIONS_PER_ROW = 10
MAX_ROWS_PER_DATASET = 10

# Model configuration constants
DEFAULT_MAX_TOKENS = 1024
DEFAULT_PLATFORM = "TogetherAI"

# Platform options
PLATFORMS = {
    "TogetherAI": "TogetherAI",
    "OpenAI": "OpenAI"
}

# Model names by platform
MODELS = {
    "TogetherAI": {
        "default": "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        "llama_3_3_70b": "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
    },
    "OpenAI": {
        "default": "gpt-4o-mini",
        "gpt_4o_mini": "gpt-4o-mini",
    }
}

# Short model names for file naming
MODEL_SHORT_NAMES = {
    "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free": "llama_3_3_70b",
    "gpt-4o-mini": "gpt_4o_mini",
}