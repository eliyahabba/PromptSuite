from typing import Dict, List, Any
import pandas as pd

from multipromptify.augmentations.base import BaseAxisAugmenter
from multipromptify.utils.formatting import format_field_value


class FewShotAugmenter(BaseAxisAugmenter):
    """
This augmenter handles few-shot examples for NLP tasks.
    It works with the engine to generate structured few-shot examples.
    """

    def __init__(self, n_augments: int = 1):
        """
        Initialize the few-shot augmenter.
        
        Args:
            n_augments: Number of variations to generate (not used in current implementation)
        """
        super().__init__(n_augments=n_augments)

    def get_name(self):
            return "Few-Shot Examples"

    def augment(self, prompt: str, identification_data: Dict[str, Any] = None) -> List[Dict[str, str]]:
        """
        Generate few-shot variations of the prompt for engine use.
        
        Args:
            prompt: The instruction template to use for few-shot examples
            identification_data: Dictionary containing:
                - few_shot_field: TemplateField object with few-shot configuration
                - data: DataFrame with the dataset
                - current_row_idx: Index of current row to exclude
                - gold_field: Name of the gold field column
                - gold_type: Type of gold field ('value' or 'index')
                - options_field: Name of options field (for index-based gold)
            
        Returns:
            List of dictionaries with 'input' and 'output' keys for few-shot examples
        """
        if not identification_data or 'few_shot_field' not in identification_data:
            return []
        
        # Validate gold field requirement
        few_shot_field = identification_data.get('few_shot_field')
        gold_field = identification_data.get('gold_field')
        few_shot_fields = [few_shot_field] if few_shot_field else []
        
        self._validate_gold_field_requirement(prompt, gold_field, few_shot_fields)
        
        # Engine mode - use structured generation
        structured_examples = self.generate_few_shot_examples_structured(
            identification_data.get('few_shot_field'),
            prompt,
            identification_data.get('data'),
            identification_data.get('current_row_idx', 0),
            identification_data.get('gold_field'),
            identification_data.get('gold_type', 'value'),
            identification_data.get('options_field')
        )
        # Return structured examples directly (not formatted strings)
        return structured_examples

    def _validate_gold_field_requirement(self, instruction_template: str, gold_field: str, few_shot_fields: list):
        """Validate that gold field is provided when needed for separating inputs from outputs."""
        needs_gold_field = False
        
        # Check if few-shot is configured (needs to separate input from output)
        if few_shot_fields and len(few_shot_fields) > 0:
            needs_gold_field = True
        
        # Check if instruction template has the gold field placeholder
        if instruction_template and gold_field:
            gold_placeholder = f'{{{gold_field}}}'
            if gold_placeholder in instruction_template:
                needs_gold_field = True
        
        if needs_gold_field and not gold_field:
            raise ValueError(
                "Gold field is required when using few-shot examples. "
                "Please specify the 'gold' field in your template to indicate which column contains the correct answers/labels. "
                "Example: \"gold\": \"answer\" or \"gold\": \"label\""
            )

    def generate_few_shot_examples_structured(self, few_shot_field, instruction_variant: str, data: pd.DataFrame,
                                            current_row_idx: int, gold_field: str = None, gold_type: str = 'value',
                                            options_field: str = None) -> List[Dict[str, str]]:
        """Generate few-shot examples using the configured parameters with structured output."""

        if not few_shot_field:
            return []

        count = few_shot_field.few_shot_count or 2
        few_shot_format = few_shot_field.few_shot_format or "rotating"
        split = few_shot_field.few_shot_split or "all"

        # Get available data for few-shot examples
        if split == "train":
            # Use only training data
            available_data = data[data.get('split', 'train') == 'train']
        elif split == "test":
            # Use only test data
            available_data = data[data.get('split', 'train') == 'test']
        else:
            # Use all data
            available_data = data

        # Remove current row to avoid data leakage
        available_data = available_data.drop(current_row_idx, errors='ignore')

        if len(available_data) < count:
            # Raise error with appropriate explanation instead of returning empty list
            raise ValueError(
                f"Not enough data for few-shot examples. "
                f"Requested {count} examples but only {len(available_data)} available after filtering. "
                f"Consider reducing the few-shot count or providing more data."
            )

        # Sample examples
        if few_shot_format == "fixed":
            # Use the same first N examples for all questions
            sampled_data = available_data.head(count)
        else:
            # Randomly sample different examples for each question
            sampled_data = available_data.sample(n=count, random_state=current_row_idx)

        examples = []
        
        for _, example_row in sampled_data.iterrows():
            # Create row values for input template (excluding gold field)
            input_values = {}
            output_value = ""
            
            for col in example_row.index:
                # Assume clean data - process all columns
                if gold_field and col == gold_field:
                    # Extract the output value separately for the output field
                    output_value = self._extract_answer_from_options(
                        example_row, gold_field, gold_type, options_field
                    )
                else:
                    # Add to input values (everything except gold field)
                    input_values[col] = format_field_value(example_row[col])

            # Fill template for input (without gold field placeholder)
            input_template = instruction_variant
            # Remove gold field placeholder from input template
            if gold_field:
                gold_placeholder = f'{{{gold_field}}}'
                input_template = input_template.replace(gold_placeholder, '').strip()
            
            input_text = self._fill_template_placeholders(input_template, input_values)

            if input_text:
                examples.append({
                    "input": input_text.strip(),
                    "output": output_value.strip() if output_value else ""
                })
        
        return examples

    def _extract_answer_from_options(self, row: pd.Series, gold_field: str, gold_type: str,
                                   options_field: str = None) -> str:
        """Extract the actual answer text from options based on the gold field."""

        if not gold_field or gold_field not in row.index:
            return format_field_value(row.get(gold_field, ''))

        gold_value = row[gold_field]

        # If gold_type is 'value', return as is
        if gold_type == 'value':
            return format_field_value(gold_value)
    
        # If gold_type is 'index', try to extract from options
        if gold_type == 'index' and options_field and options_field in row.index:
            try:
                # Import here to avoid circular imports
                from multipromptify.augmentations.structure.shuffle import ShuffleAugmenter
                shuffle_augmenter = ShuffleAugmenter()

                options_text = str(row[options_field])
                options_list = shuffle_augmenter._parse_input_to_list(options_text)

                index = int(gold_value)
                if 0 <= index < len(options_list):
                    # Return the actual option text, cleaned up
                    return options_list[index].strip()

            except (ValueError, IndexError, Exception):
                pass

        # Fallback: return the gold value as string
        return format_field_value(gold_value)

    def _fill_template_placeholders(self, template: str, values: Dict[str, str]) -> str:
        """Fill template placeholders with values."""
        if not template:
            return ""

        result = template
        for field_name, field_value in values.items():
            placeholder = f'{{{field_name}}}'
            if placeholder in result:
                result = result.replace(placeholder, format_field_value(field_value))

        return result

    def format_few_shot_as_string(self, few_shot_examples: List[Dict[str, str]]) -> str:
        """Format few-shot examples as string."""
        if not few_shot_examples:
            return ""

        formatted_examples = []
        for example in few_shot_examples:
            # Combine input and output for the traditional prompt format
            formatted_example = f"{example['input']}\n{example['output']}"
            formatted_examples.append(formatted_example)

        return "\n\n".join(formatted_examples)


if __name__ == "__main__":
    print("FewShotAugmenter is designed to work with the MultiPromptify engine.")
    print("It requires few_shot_field configuration and structured data.")
    print("For standalone usage examples, please refer to the engine documentation.")
