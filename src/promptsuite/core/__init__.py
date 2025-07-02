"""
PromptSuiteEngine: A tool that creates multi-prompt datasets from single-prompt datasets using templates.
"""

__version__ = "3.0.0"

# Import main classes for easier access
from .engine import PromptSuiteEngine
from .api import PromptSuite
from .template_parser import TemplateParser

# Import exceptions for better error handling
from .exceptions import (
    PromptSuiteEngineError,
    TemplateError,
    InvalidTemplateError,
    MissingInstructionTemplateError,
    TemplateValidationError,
    DataError,
    DataNotLoadedError,
    FileNotFoundError,
    DataParsingError,
    UnsupportedFileFormatError,
    FewShotError,
    FewShotGoldFieldMissingError,
    FewShotDataInsufficientError,
    FewShotConfigurationError,
    ConfigurationError,
    InvalidConfigurationError,
    UnknownConfigurationError,
    APIError,
    APIKeyMissingError,
    DatasetLoadError,
    GenerationError,
    ExportError,
    NoResultsToExportError,
    UnsupportedExportFormatError,
    ExportWriteError,
    AugmentationError,
    ShuffleIndexError,
    ErrorCollector
)

from promptsuite.core.template_keys import (
    PROMPT_FORMAT, PROMPT_FORMAT_VARIATIONS, QUESTION_KEY, GOLD_KEY, FEW_SHOT_KEY, OPTIONS_KEY, CONTEXT_KEY, PROBLEM_KEY,
    PARAPHRASE_WITH_LLM, CONTEXT_VARIATION, SHUFFLE_VARIATION, MULTIDOC_VARIATION, ENUMERATE_VARIATION
)

__all__ = [
    "PromptSuiteEngine", 
    "PromptSuite", 
    "TemplateParser",
    # Exceptions
    "PromptSuiteEngineError",
    "TemplateError",
    "InvalidTemplateError", 
    "MissingInstructionTemplateError",
    "TemplateValidationError",
    "DataError",
    "DataNotLoadedError",
    "FileNotFoundError",
    "DataParsingError",
    "UnsupportedFileFormatError", 
    "FewShotError",
    "FewShotGoldFieldMissingError",
    "FewShotDataInsufficientError",
    "FewShotConfigurationError",
    "ConfigurationError",
    "InvalidConfigurationError",
    "UnknownConfigurationError",
    "APIError",
    "APIKeyMissingError",
    "DatasetLoadError",
    "GenerationError",
    "ExportError", 
    "NoResultsToExportError",
    "UnsupportedExportFormatError",
    "ExportWriteError",
    "AugmentationError",
    "ShuffleIndexError",
    "ErrorCollector"
]