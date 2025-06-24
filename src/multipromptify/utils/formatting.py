"""
Formatting utilities for MultiPromptify.
"""

from typing import Any, Optional
import pandas as pd
from multipromptify.core.exceptions import GoldFieldExtractionError
from multipromptify.shared.constants import ListFormattingConstants


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
    
    # Handle Python lists and tuples - convert using default separator
    if isinstance(value, (list, tuple)):
        return ListFormattingConstants.DEFAULT_LIST_SEPARATOR.join(str(item) for item in value)
    
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


def convert_index_to_value(row: pd.Series, gold_field: str, gold_type: str, options_field: str = None) -> str:
    """
    Convert gold index to actual value from options field.
    
    This utility function consolidates the logic for converting an index-based gold field
    to its corresponding value from an options field.
    
    Args:
        row: pandas Series containing the data row
        gold_field: Name of the gold field column
        gold_type: Type of gold field ('value' or 'index')
        options_field: Name of the options field column (required for index type)
        
    Returns:
        String representation of the gold value (converted from index if needed)
    """
    if not gold_field or gold_field not in row.index:
        return format_field_value(row.get(gold_field, ''))

    gold_value = row[gold_field]

    # If gold_type is 'value', return as is
    if gold_type == 'value':
        return format_field_value(gold_value)
    
    # If gold_type is 'index', try to extract from options
    if gold_type == 'index' and options_field and options_field in row.index:
        try:
            options_data = row[options_field]
            
            # Handle both list and string formats
            if isinstance(options_data, (list, tuple)):
                options_list = [str(item).strip() for item in options_data]
            else:
                # Parse options as comma-separated string (existing logic)
                options_text = str(options_data)
                options_list = [item.strip() for item in options_text.split(',')]

            index = int(gold_value)
            if 0 <= index < len(options_list):
                # Return the actual option text, cleaned up
                return options_list[index].strip()

        except (ValueError, IndexError):
            pass

    # Fallback: return the gold value as string
    return format_field_value(gold_value) 