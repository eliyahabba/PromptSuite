"""
MultiPromptify Custom Exceptions

This module defines a hierarchical structure of custom exceptions for the MultiPromptify library.
Each exception provides:
- Clear error messages
- Context about the problem
- Suggested solutions
- Error codes for programmatic handling
"""

from typing import List, Dict, Any


class MultiPromptifyError(Exception):
    """Base exception for all MultiPromptify errors."""

    def __init__(self, message: str, error_code: str = None, context: Dict[str, Any] = None,
                 suggestion: str = None):
        """
        Initialize a MultiPromptify error.
        
        Args:
            message: Human-readable error message
            error_code: Unique error code for programmatic handling
            context: Additional context about the error
            suggestion: Suggested solution or next steps
        """
        super().__init__(message)
        self.error_code = error_code or self.__class__.__name__.upper()
        self.context = context or {}
        self.suggestion = suggestion

    def __str__(self):
        """Format the error message with context and suggestion."""
        message = super().__str__()

        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            message += f" [Context: {context_str}]"

        if self.suggestion:
            message += f" [Suggestion: {self.suggestion}]"

        message += f" [Error Code: {self.error_code}]"
        return message


# Template-related exceptions
class TemplateError(MultiPromptifyError):
    """Base class for template-related errors."""
    pass


class InvalidTemplateError(TemplateError):
    """Raised when template format is invalid."""

    def __init__(self, errors: List[str], template: Dict[str, Any] = None):
        context = {"template_keys": list(template.keys()) if template else None}
        suggestion = "Please check your template format. Required fields: prompt_format_template. Optional: gold, few_shot, variation fields"

        if len(errors) == 1:
            message = f"Invalid template: {errors[0]}"
        else:
            message = f"Invalid template with {len(errors)} errors: {'; '.join(errors)}"

        super().__init__(message, "TEMPLATE_INVALID", context, suggestion)
        self.errors = errors


class MissingTemplateError(TemplateError):
    """Raised when required template is not provided."""

    def __init__(self):
        message = "No template configuration provided"
        suggestion = "Use set_template() to provide a template dictionary before generating variations"
        super().__init__(message, "TEMPLATE_MISSING", suggestion=suggestion)


class InvalidTemplateFieldError(TemplateError):
    """Raised when a template field has invalid configuration."""

    def __init__(self, field_name: str, field_value: Any, expected_type: str = None):
        context = {"field_name": field_name, "field_value": str(field_value), "field_type": type(field_value).__name__}
        suggestion = f"Field '{field_name}' should be {expected_type or 'a valid configuration'}. Check the documentation for proper format"
        message = f"Invalid configuration for template field '{field_name}'"
        super().__init__(message, "TEMPLATE_FIELD_INVALID", context, suggestion)


class MissingInstructionTemplateError(TemplateError):
    """Raised when prompt_format_template is required but not provided."""

    def __init__(self, has_prompt_format_variations: bool = False):
        message = "prompt_format_template is required"
        context = {"has_prompt_format_variations": has_prompt_format_variations}

        if has_prompt_format_variations:
            suggestion = "Add 'prompt_format_template' to your template when using prompt_format variations. Example: \"prompt_format_template\": \"Process: {input}\\nOutput: {output}\""
        else:
            suggestion = "Add 'prompt_format_template' to your template. Example: \"prompt_format_template\": \"Process: {input}\\nOutput: {output}\""

        super().__init__(message, "INSTRUCTION_TEMPLATE_MISSING", context, suggestion)


class TemplateValidationError(TemplateError):
    """Raised when template validation fails."""

    def __init__(self, field_name: str, validation_message: str):
        context = {"field_name": field_name}
        message = f"Template validation failed for field '{field_name}': {validation_message}"
        suggestion = "Check the template field configuration and ensure it meets the requirements"
        super().__init__(message, "TEMPLATE_VALIDATION_ERROR", context, suggestion)


# Data-related exceptions
class DataError(MultiPromptifyError):
    """Base class for data-related errors."""
    pass


class DataNotLoadedError(DataError):
    """Raised when no data has been loaded."""

    def __init__(self):
        message = "No data loaded"
        suggestion = "Use load_dataset(), load_csv(), load_json(), or load_dataframe() to load data first"
        super().__init__(message, "DATA_NOT_LOADED", suggestion=suggestion)


class InvalidDataFormatError(DataError):
    """Raised when data format is invalid."""

    def __init__(self, expected_format: str, actual_format: str, filepath: str = None):
        context = {"expected_format": expected_format, "actual_format": actual_format}
        if filepath:
            context["filepath"] = filepath

        message = f"Invalid data format: expected {expected_format}, got {actual_format}"
        suggestion = "Ensure your data is in the correct format. For DataFrames, use load_dataframe(). For files, check file extension and content"
        super().__init__(message, "DATA_FORMAT_INVALID", context, suggestion)


class FileNotFoundError(DataError):
    """Raised when a data file cannot be found."""

    def __init__(self, filepath: str, file_type: str = "file"):
        context = {"filepath": filepath, "file_type": file_type}
        message = f"{file_type.capitalize()} not found: {filepath}"
        suggestion = f"Check that the {file_type} path is correct and the file exists"
        super().__init__(message, "FILE_NOT_FOUND", context, suggestion)


class DataParsingError(DataError):
    """Raised when data cannot be parsed."""

    def __init__(self, filepath: str, file_type: str, original_error: str):
        context = {"filepath": filepath, "file_type": file_type, "original_error": original_error}
        message = f"Failed to parse {file_type} file '{filepath}'"
        suggestion = f"Check that the {file_type} file is valid and not corrupted. Original error: {original_error}"
        super().__init__(message, "DATA_PARSING_ERROR", context, suggestion)


class UnsupportedFileFormatError(DataError):
    """Raised when file format is not supported."""

    def __init__(self, filepath: str, supported_formats: List[str] = None):
        context = {"filepath": filepath}
        if supported_formats:
            context["supported_formats"] = supported_formats
            suggestion = f"Use one of these supported formats: {', '.join(supported_formats)}"
        else:
            suggestion = "Currently supported formats: .csv, .json"

        message = f"Unsupported file format: {filepath}"
        super().__init__(message, "FILE_FORMAT_UNSUPPORTED", context, suggestion)


class InsufficientDataError(DataError):
    """Raised when there's not enough data for the requested operation."""

    def __init__(self, required: int, available: int, operation: str = "operation"):
        context = {"required": required, "available": available, "operation": operation}
        message = f"Insufficient data for {operation}: need {required}, have {available}"
        suggestion = f"Provide more data or reduce the requirements for {operation}"
        super().__init__(message, "DATA_INSUFFICIENT", context, suggestion)


class GoldFieldExtractionError(DataError):
    """Raised when extracting the gold field value fails (e.g., invalid expression or missing key)."""

    def __init__(self, gold_field: str, row: dict, original_error: str):
        context = {"gold_field": gold_field, "row_keys": list(row.keys())}
        message = f"Failed to extract gold field '{gold_field}' from row. Error: {original_error}"
        suggestion = ("Check that your gold field expression is valid for your data. "
                      "For nested fields, use Python expressions like answers['text'][0]. ")
        super().__init__(message, "GOLD_EXTRACTION_ERROR", context, suggestion)


# Few-shot related exceptions  
class FewShotError(MultiPromptifyError):
    """Base class for few-shot related errors."""
    pass


class FewShotConfigurationError(FewShotError):
    """Raised when few-shot configuration is invalid."""

    def __init__(self, config_key: str, config_value: Any, valid_values: List[str] = None):
        context = {"config_key": config_key, "config_value": str(config_value)}
        if valid_values:
            context["valid_values"] = valid_values
            suggestion = f"Set {config_key} to one of: {', '.join(valid_values)}"
        else:
            suggestion = f"Check the documentation for valid {config_key} values"

        message = f"Invalid few-shot configuration: {config_key}={config_value}"
        super().__init__(message, "FEWSHOT_CONFIG_INVALID", context, suggestion)


class FewShotGoldFieldMissingError(FewShotError):
    """Raised when gold field is required for few-shot but not specified."""

    def __init__(self):
        message = "Gold field is required when using few-shot examples"
        suggestion = "Specify the 'gold' field in your template to indicate which column contains correct outputs. Example: \"gold\": \"output\""
        super().__init__(message, "FEWSHOT_GOLD_MISSING", suggestion=suggestion)


class FewShotDataInsufficientError(FewShotError):
    """Raised when there's not enough data for few-shot examples."""

    def __init__(self, requested: int, available: int, split: str = "all"):
        context = {"requested": requested, "available": available, "split": split}
        message = f"Not enough data for few-shot examples: requested {requested}, available {available} (split: {split})"
        suggestion = "Reduce few-shot count or provide more data. Check if split configuration is correct"
        super().__init__(message, "FEWSHOT_DATA_INSUFFICIENT", context, suggestion)


# Validation-related exceptions
class ValidationError(MultiPromptifyError):
    """Base class for validation errors."""
    pass


class FieldValidationError(ValidationError):
    """Raised when field validation fails."""

    def __init__(self, field_name: str, field_value: Any, validation_rule: str):
        context = {"field_name": field_name, "field_value": str(field_value), "validation_rule": validation_rule}
        message = f"Field '{field_name}' failed validation: {validation_rule}"
        suggestion = f"Ensure field '{field_name}' meets the requirement: {validation_rule}"
        super().__init__(message, "FIELD_VALIDATION_ERROR", context, suggestion)


class MissingRequiredFieldError(ValidationError):
    """Raised when a required field is missing."""

    def __init__(self, field_name: str, available_fields: List[str] = None):
        context = {"field_name": field_name}
        if available_fields:
            context["available_fields"] = available_fields

        message = f"Required field '{field_name}' is missing"
        if available_fields:
            suggestion = f"Add the '{field_name}' field to your data. Available fields: {', '.join(available_fields)}"
        else:
            suggestion = f"Add the '{field_name}' field to your data"
        super().__init__(message, "REQUIRED_FIELD_MISSING", context, suggestion)


# API and Configuration exceptions
class ConfigurationError(MultiPromptifyError):
    """Base class for configuration errors."""
    pass


class InvalidConfigurationError(ConfigurationError):
    """Raised when configuration parameter is invalid."""

    def __init__(self, parameter: str, value: Any, valid_values: List[str] = None):
        context = {"parameter": parameter, "value": str(value)}
        if valid_values:
            context["valid_values"] = valid_values
            suggestion = f"Set {parameter} to one of: {', '.join(valid_values)}"
        else:
            suggestion = f"Check documentation for valid {parameter} values"

        message = f"Invalid configuration parameter '{parameter}': {value}"
        super().__init__(message, "CONFIG_INVALID", context, suggestion)


class UnknownConfigurationError(ConfigurationError):
    """Raised when unknown configuration parameter is provided."""

    def __init__(self, parameter: str, valid_parameters: List[str] = None):
        context = {"parameter": parameter}
        if valid_parameters:
            context["valid_parameters"] = valid_parameters
            suggestion = f"Use one of these valid parameters: {', '.join(valid_parameters)}"
        else:
            suggestion = "Check documentation for valid configuration parameters"

        message = f"Unknown configuration parameter: {parameter}"
        super().__init__(message, "CONFIG_UNKNOWN", context, suggestion)


class APIError(MultiPromptifyError):
    """Base class for API-related errors."""
    pass


class APIKeyMissingError(APIError):
    """Raised when API key is required but not provided."""

    def __init__(self, platform: str):
        context = {"platform": platform}
        env_var = "TOGETHER_API_KEY" if platform == "TogetherAI" else "OPENAI_API_KEY"
        message = f"API key required for {platform} but not found"
        suggestion = f"Set API key with configure(api_key='your_key') or environment variable {env_var}"
        super().__init__(message, "API_KEY_MISSING", context, suggestion)


class DatasetLoadError(APIError):
    """Raised when dataset cannot be loaded from API."""

    def __init__(self, dataset_name: str, original_error: str):
        context = {"dataset_name": dataset_name, "original_error": original_error}
        message = f"Failed to load dataset '{dataset_name}'"
        suggestion = "Check dataset name, internet connection, and that the dataset exists on the platform"
        super().__init__(message, "DATASET_LOAD_ERROR", context, suggestion)


# Generation and Processing exceptions  
class GenerationError(MultiPromptifyError):
    """Base class for generation errors."""
    pass


class VariationGenerationError(GenerationError):
    """Raised when variation generation fails."""

    def __init__(self, field_name: str, variation_type: str, original_error: str):
        context = {"field_name": field_name, "variation_type": variation_type, "original_error": original_error}
        message = f"Failed to generate {variation_type} variation for field '{field_name}'"
        suggestion = "Check field data quality and API connectivity if using paraphrase variations"
        super().__init__(message, "VARIATION_GENERATION_ERROR", context, suggestion)


class ExportError(MultiPromptifyError):
    """Base class for export errors."""
    pass


class NoResultsToExportError(ExportError):
    """Raised when trying to export but no results exist."""

    def __init__(self):
        message = "No results to export"
        suggestion = "Run generate() first to create variations before exporting"
        super().__init__(message, "NO_RESULTS_TO_EXPORT", suggestion=suggestion)


class UnsupportedFormatError(ExportError):
    """Raised when export format is not supported."""

    def __init__(self, format_name: str, supported_formats: List[str] = None):
        context = {"format": format_name}
        if supported_formats:
            context["supported_formats"] = supported_formats
            suggestion = f"Use one of these supported formats: {', '.join(supported_formats)}"
        else:
            suggestion = "Check documentation for supported export formats"

        message = f"Unsupported export format: {format_name}"
        super().__init__(message, "FORMAT_UNSUPPORTED", context, suggestion)


# Alias for UnsupportedFormatError for more specific naming
UnsupportedExportFormatError = UnsupportedFormatError


class ExportWriteError(ExportError):
    """Raised when export fails due to write issues."""

    def __init__(self, filepath: str, original_error: str):
        context = {"filepath": filepath, "original_error": original_error}
        message = f"Failed to write export file: {filepath}"
        suggestion = "Check file permissions, disk space, and that the directory exists"
        super().__init__(message, "EXPORT_WRITE_ERROR", context, suggestion)


# Augmentation-specific exceptions
class AugmentationError(MultiPromptifyError):
    """Base class for augmentation errors."""
    pass


class InvalidAugmentationInputError(AugmentationError):
    """Raised when augmentation input is invalid."""

    def __init__(self, augmentation_name: str, expected_type: str, actual_type: str):
        context = {"augmentation": augmentation_name, "expected_type": expected_type, "actual_type": actual_type}
        message = f"{augmentation_name} expects {expected_type} input, got {actual_type}"
        suggestion = f"Ensure input data for {augmentation_name} is of type {expected_type}"
        super().__init__(message, "AUGMENTATION_INPUT_INVALID", context, suggestion)


class AugmentationConfigurationError(AugmentationError):
    """Raised when augmentation configuration is invalid."""

    def __init__(self, augmentation_name: str, missing_keys: List[str]):
        context = {"augmentation": augmentation_name, "missing_keys": missing_keys}
        message = f"{augmentation_name} missing required configuration: {', '.join(missing_keys)}"
        suggestion = f"Provide required configuration keys for {augmentation_name}: {', '.join(missing_keys)}"
        super().__init__(message, "AUGMENTATION_CONFIG_INVALID", context, suggestion)


class ShuffleIndexError(AugmentationError):
    """Raised when shuffle operation has index issues."""

    def __init__(self, index: int, list_length: int):
        context = {"index": index, "list_length": list_length}
        message = f"Shuffle index {index} out of range for list of length {list_length}"
        suggestion = "Ensure gold_value is a valid index within the data range"
        super().__init__(message, "SHUFFLE_INDEX_ERROR", context, suggestion)


class EnumeratorLengthMismatchError(AugmentationError):
    """Raised when enumeration sequence is shorter than the list to enumerate."""

    def __init__(self, sequence_length: int, list_length: int, sequence_type: str):
        context = {"sequence_length": sequence_length, "list_length": list_length, "sequence_type": sequence_type}
        message = f"Enumeration sequence ({sequence_type}) has {sequence_length} items but list has {list_length} items"
        suggestion = f"Provide an enumeration sequence with at least {list_length} items to match the list length"
        super().__init__(message, "ENUMERATOR_LENGTH_MISMATCH", context, suggestion)


# Utility functions for error collection
class ErrorCollector:
    """Utility class for collecting multiple errors before raising."""

    def __init__(self):
        self.errors: List[MultiPromptifyError] = []

    def add_error(self, error: MultiPromptifyError):
        """Add an error to the collection."""
        self.errors.append(error)

    def has_errors(self) -> bool:
        """Check if any errors have been collected."""
        return len(self.errors) > 0

    def get_error_messages(self) -> List[str]:
        """Get list of error messages."""
        return [str(error) for error in self.errors]

    def raise_if_errors(self, operation: str = "operation"):
        """Raise a combined error if any errors were collected."""
        if self.has_errors():
            if len(self.errors) == 1:
                raise self.errors[0]
            else:
                messages = [str(error) for error in self.errors]
                combined_message = f"Multiple errors during {operation}: " + "; ".join(messages)
                raise MultiPromptifyError(combined_message, "MULTIPLE_ERRORS",
                                          {"error_count": len(self.errors)},
                                          "Fix each error individually")
