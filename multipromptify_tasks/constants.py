# Task Generation Constants (for creating prompt variations)
TASK_DEFAULT_VARIATIONS_PER_FIELD = 15
TASK_DEFAULT_MAX_VARIATIONS_PER_ROW = 25
TASK_DEFAULT_MAX_ROWS = 50
TASK_DEFAULT_RANDOM_SEED = 42
TASK_DEFAULT_PLATFORM = "TogetherAI"
TASK_DEFAULT_MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
# TASK_DEFAULT_PLATFORM = "OpenAI"
# TASK_DEFAULT_MODEL_NAME = "gpt-4o-mini"

# Language Model Running Constants (for running models on variations)
LM_DEFAULT_MAX_TOKENS = 1024
LM_DEFAULT_PLATFORM = "OpenAI"
LM_DEFAULT_MODEL_NAME = "gpt-4o-mini"
LM_DEFAULT_TEMPERATURE = 0.0
LM_DEFAULT_PARALLEL_WORKERS = 6  # Number of parallel workers for model calls (1 = sequential)

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

# Backward compatibility aliases (deprecated - use specific TASK_ or LM_ prefixed constants)
DEFAULT_VARIATIONS_PER_FIELD = TASK_DEFAULT_VARIATIONS_PER_FIELD
DEFAULT_MAX_VARIATIONS_PER_ROW = TASK_DEFAULT_MAX_VARIATIONS_PER_ROW
DEFAULT_MAX_ROWS = TASK_DEFAULT_MAX_ROWS
DEFAULT_RANDOM_SEED = TASK_DEFAULT_RANDOM_SEED
DEFAULT_PLATFORM = TASK_DEFAULT_PLATFORM
DEFAULT_MODEL_NAME = TASK_DEFAULT_MODEL_NAME
DEFAULT_MAX_TOKENS = LM_DEFAULT_MAX_TOKENS
DEFAULT_PARALLEL_WORKERS = LM_DEFAULT_PARALLEL_WORKERS