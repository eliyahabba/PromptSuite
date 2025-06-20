"""
Formatting utilities for MultiPromptify.
"""

from typing import Any
from multipromptify.core.exceptions import GoldFieldExtractionError


def format_field_value(value: Any) -> str:
    """
    Format a field value for display in prompts in a user-friendly way.
    
    Simple rules:
    - Lists/tuples: convert to comma-separated string
    - Everything else: convert to string as-is
    
    Args:
        value: The value to format
        
    Returns:
        User-friendly string representation
    """
    if value is None:
        return ""
    
    # Handle Python lists and tuples - convert to comma-separated
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value)
    
    # Everything else - just convert to string
    return str(value)


def format_field_values_dict(values: dict) -> dict:
    """
    Format all values in a dictionary using format_field_value.
    
    Args:
        values: Dictionary of field values
        
    Returns:
        Dictionary with formatted values
    """
    return {key: format_field_value(value) for key, value in values.items()}


def extract_gold_value(row, gold_field):
    """
    Extract the gold value from a row, supporting both simple fields and Python expressions.
    - If gold_field is a simple column name, returns row[gold_field]
    - If gold_field is an expression (e.g., answers['text'][0]), evaluates it with row as context
    Raises GoldFieldExtractionError if extraction fails.
    """
    if isinstance(gold_field, str) and any(c in gold_field for c in ".[[]'):"):
        try:
            return eval(gold_field, {}, row)
        except Exception as e:
            raise GoldFieldExtractionError(gold_field, row, str(e))
    else:
        try:
            return row[gold_field]
        except Exception as e:
            raise GoldFieldExtractionError(gold_field, row, str(e)) 