"""
Prompt Builder: Handles building prompts from templates and filling placeholders.
"""

from typing import Dict
import pandas as pd


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
                result = result.replace(placeholder, str(field_value))

        return result

    def create_main_input(self, instruction_variant: str, row: pd.Series, gold_field: str = None) -> str:
        """Create main input by filling instruction with row data (excluding outputs)."""

        row_values = {}
        for col in row.index:
            # Handle pandas array comparison issue
            try:
                is_not_na = pd.notna(row[col])
                if hasattr(is_not_na, '__len__') and len(is_not_na) > 1:
                    # For arrays/lists, check if any element is not na
                    is_not_na = is_not_na.any() if hasattr(is_not_na, 'any') else True
            except (ValueError, TypeError):
                # Fallback: assume not na if we can't check
                is_not_na = row[col] is not None

            if is_not_na:
                # Skip the gold output field for the main input
                if gold_field and col == gold_field:
                    continue
                else:
                    row_values[col] = str(row[col])

        # Fill template and remove the gold field placeholder completely
        input_text = self.fill_template_placeholders(instruction_variant, row_values)

        # Remove any remaining gold field placeholder
        if gold_field:
            input_text = input_text.replace(f'{{{gold_field}}}', '')

        return input_text.strip()

    def validate_gold_field_requirement(self, instruction_template: str, gold_field: str, few_shot_fields: list):
        """Validate that gold field is provided when needed for few-shot examples."""

        # Check if few-shot is configured (needs to separate input from output)
        if few_shot_fields and len(few_shot_fields) > 0 and not gold_field:
            raise ValueError(
                "Gold field is required when using few-shot examples. "
                "Please specify the 'gold' field in your template to indicate which column contains the correct outputs/labels. "
                "Example: \"gold\": \"output\" or \"gold\": \"label\""
            ) 