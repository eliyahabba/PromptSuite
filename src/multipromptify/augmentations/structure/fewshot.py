from typing import Dict, List, Any, Optional

import pandas as pd

from multipromptify.augmentations.base import BaseAxisAugmenter
from multipromptify.core.exceptions import FewShotGoldFieldMissingError, FewShotDataInsufficientError
from multipromptify.utils.formatting import format_field_value


class FewShotAugmenter(BaseAxisAugmenter):
    """
This augmenter handles few-shot examples for NLP tasks.
    It works with the engine to generate structured few-shot examples.
    """

    def __init__(self, n_augments: int = 1, seed: Optional[int] = None):
        """
        Initialize the few-shot augmenter.
        
        Args:
            n_augments: Number of variations to generate (not used in current implementation)
            seed: Random seed for reproducibility
        """
        super().__init__(n_augments=n_augments, seed=seed)

    def get_name(self):
        return "Few-Shot Examples"

    def augment(self, prompt: str, identification_data: Dict[str, Any] = None) -> List[Dict[str, str]]:
        """
        Generate few-shot variations of the prompt for engine use.
        
        Args:
            prompt: The prompt format template to use for few-shot examples
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
            identification_data.get('options_field'),
            identification_data.get('enumerate_configs')
        )
        # Return structured examples directly (not formatted strings)
        return structured_examples

    def _validate_gold_field_requirement(self, prompt_format_template: str, gold_field: str, few_shot_fields: list):
        """Validate that gold field is provided when needed for separating inputs from outputs."""
        needs_gold_field = False

        # Check if few-shot is configured (needs to separate input from output)
        if few_shot_fields and len(few_shot_fields) > 0:
            needs_gold_field = True

        # Check if prompt format template has the gold field placeholder
        if prompt_format_template and gold_field:
            gold_placeholder = f'{{{gold_field}}}'
            if gold_placeholder in prompt_format_template:
                needs_gold_field = True

        if needs_gold_field and not gold_field:
            raise FewShotGoldFieldMissingError()

    def generate_few_shot_examples_structured(self, few_shot_field, prompt_format_variant: str, data: pd.DataFrame,
                                              current_row_idx: int, gold_field: str = None, gold_type: str = 'value',
                                              options_field: str = None, enumerate_configs: Dict[str, dict] = None) -> List[Dict[str, str]]:
        """Generate few-shot examples using the configured parameters with structured output.
        
        Few-shot formats:
        - "shared_first_n": Always use the first N examples from the available data (deterministic, shared for all rows)
        - "shared_random_n": Always use the same N random examples (with fixed seed, shared for all rows)
        - "random_per_row": Randomly sample different examples for each row (using row index as seed)
        """

        if not few_shot_field:
            return []

        count = few_shot_field.few_shot_count or 2
        few_shot_format = few_shot_field.few_shot_format or "shared_first_n"
        split = few_shot_field.few_shot_split or "all"

        # Get available data for few-shot examples based on split configuration
        if split == "train":
            available_data = data[data.get('split', 'train') == 'train']
        elif split == "test":
            available_data = data[data.get('split', 'train') == 'test']
        else:
            available_data = data

        # Remove current row to avoid data leakage (regardless of its split)
        available_data = available_data.drop(current_row_idx, errors='ignore')

        if len(available_data) < count:
            raise FewShotDataInsufficientError(count, len(available_data), split)

        # Sample examples based on format
        if few_shot_format == "shared_first_n":
            sampled_data = available_data.head(count)
        elif few_shot_format == "shared_random_n":
            sampled_data = available_data.sample(n=count, random_state=42)
        elif few_shot_format == "random_per_row":
            sampled_data = available_data.sample(n=count, random_state=current_row_idx)
        else:
            print(f"⚠️ Unknown few-shot format '{few_shot_format}', using 'shared_first_n'")
            sampled_data = available_data.head(count)

        examples = []
        for _, example_row in sampled_data.iterrows():
            input_values = {}
            output_value = ""
            for col in example_row.index:
                if gold_field and col == gold_field:
                    from multipromptify.utils.formatting import convert_index_to_value
                    output_value = convert_index_to_value(
                        example_row, gold_field, gold_type, options_field
                    )
                    # If the options field is enumerated and gold_type is 'index', 
                    # we need to format the output to match the enumerated format
                    if (gold_type == 'index' and options_field and 
                        enumerate_configs and options_field in enumerate_configs):
                        enum_config = enumerate_configs[options_field]
                        enum_type = enum_config.get('type', '1234')
                        try:
                            gold_index = int(example_row[gold_field])
                            from multipromptify.augmentations.structure.enumerate import EnumeratorAugmenter
                            enumerator = EnumeratorAugmenter()
                            
                            # Handle both list and string formats for options
                            options_data = example_row[options_field]
                            if isinstance(options_data, (list, tuple)):
                                options_list = [str(item).strip() for item in options_data]
                            else:
                                # Create enumerated format: "1. option1, 2. option2, ..." then extract the right one
                                options_text = str(options_data)
                                options_list = [item.strip() for item in options_text.split(',')]
                            
                            if 0 <= gold_index < len(options_list):
                                # Format as enumerated item: "2. option_text"
                                if enum_type == '1234':
                                    output_value = f"{gold_index + 1}. {options_list[gold_index].strip()}"
                                elif enum_type == 'ABCD':
                                    letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
                                    if gold_index < len(letters):
                                        output_value = f"{letters[gold_index]}. {options_list[gold_index].strip()}"
                                elif enum_type == 'abcd':
                                    letters = 'abcdefghijklmnopqrstuvwxyz'
                                    if gold_index < len(letters):
                                        output_value = f"{letters[gold_index]}. {options_list[gold_index].strip()}"
                                elif enum_type == 'roman':
                                    romans = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X']
                                    if gold_index < len(romans):
                                        output_value = f"{romans[gold_index]}. {options_list[gold_index].strip()}"
                                # Add more enum types as needed
                        except (ValueError, IndexError) as e:
                            print(f"⚠️ Error formatting enumerated gold value: {e}")
                else:
                    field_value = format_field_value(example_row[col])
                    # Apply enumeration if configured
                    if enumerate_configs and col in enumerate_configs:
                        enum_config = enumerate_configs[col]
                        enum_type = enum_config.get('type', '1234')
                        try:
                            from multipromptify.augmentations.structure.enumerate import EnumeratorAugmenter
                            enumerator = EnumeratorAugmenter()
                            field_value = enumerator.enumerate_field(field_value, enum_type)
                        except Exception as e:
                            print(f"⚠️ Error enumerating field '{col}' in few-shot example: {e}")
                    input_values[col] = field_value
            input_template = prompt_format_variant
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
