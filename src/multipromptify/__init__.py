"""
MultiPromptify: A tool that creates multi-prompt datasets from single-prompt datasets using templates.
"""

__version__ = "2.0.2"

# Import main classes for easier access
from .engine import MultiPromptify
from .api import MultiPromptifier
from .template_parser import TemplateParser

# Import exceptions for better error handling
from .exceptions import (
    MultiPromptifyError,
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

__all__ = [
    "MultiPromptify", 
    "MultiPromptifier", 
    "TemplateParser",
    # Exceptions
    "MultiPromptifyError",
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