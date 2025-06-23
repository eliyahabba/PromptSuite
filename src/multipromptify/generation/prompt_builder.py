"""
Prompt Builder: Handles building prompts from templates and filling placeholders.
"""

from typing import Dict

import pandas as pd

from multipromptify.utils.formatting import format_field_value


class PromptBuilder:
    """
    Handles building prompts from templates and filling placeholders with data.
    """

    def fill_template_placeholders(self, template: str, values: Dict[str, str]) -> str:
        """Fill template placeholders with values."""
        if not template:
            return ""

        result = template
        for field_name, field_value in values.items():
            placeholder = f'{{{field_name}}}'
            if placeholder in result:
                result = result.replace(placeholder, format_field_value(field_value))

        return result

    def create_main_input(self, prompt_format_variant: str, row: pd.Series, gold_field: str = None) -> str:
        """Create main input by filling prompt_format with row data (excluding outputs)."""

        row_values = {}
        for col in row.index:
            # Assume clean data - skip gold field, process all others
            if gold_field and col == gold_field:
                continue  # Skip the gold output field for the main input
            else:
                row_values[col] = format_field_value(row[col])

        # Fill template and remove the gold field placeholder completely
        input_text = self.fill_template_placeholders(prompt_format_variant, row_values)

        # Remove any remaining gold field placeholder
        if gold_field:
            input_text = input_text.replace(f'{{{gold_field}}}', '')

        return input_text.strip()
