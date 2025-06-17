"""
Formatting utilities for MultiPromptify.
"""

from typing import Any


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