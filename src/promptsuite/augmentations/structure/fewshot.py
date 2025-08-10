from typing import Dict, List, Any, Optional

import pandas as pd

from promptsuite.augmentations.base import BaseAxisAugmenter
from promptsuite.core.exceptions import FewShotGoldFieldMissingError, FewShotDataInsufficientError
from promptsuite.utils.formatting import format_field_value


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
            identification_data.get('enumerate_configs'),
            identification_data
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
                                              options_field: str = None, enumerate_configs: Dict[str, dict] = None,
                                              identification_data: Dict[str, Any] = None) -> List[Dict[str, str]]:
        """Generate few-shot examples using the configured parameters with structured output.
        
        Few-shot formats:
        - "same_examples__no_variations": Same examples for all rows, no variations (single variation per row)
        - "same_examples__synchronized_order_variations": Same examples for all rows, synchronized order variations
        - "different_examples__same_shuffling_order_across_rows": Different examples per row, same shuffling order across rows
        - "different_examples__different_order_per_variation": Different examples and different order per variation
        """

        if not few_shot_field:
            return []

        count = few_shot_field.few_shot_count or 2
        few_shot_format = few_shot_field.few_shot_format or "same_examples__no_variations"
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

        # Apply category filtering if configured
        filter_by = getattr(few_shot_field, 'few_shot_filter_by', None)
        fallback_strategy = getattr(few_shot_field, 'few_shot_fallback_strategy', 'global')
        
        if filter_by:
            current_row = data.loc[current_row_idx]
            available_data = self._filter_examples_by_category(
                available_data, current_row, filter_by, count, fallback_strategy
            )

        if len(available_data) < count:
            if filter_by and fallback_strategy == 'strict':
                current_category = data.loc[current_row_idx][filter_by] if current_row_idx in data.index else "Unknown"
                raise FewShotDataInsufficientError(
                    count, len(available_data), split,
                    filter_by=filter_by, filter_value=current_category
                )
            else:
                raise FewShotDataInsufficientError(count, len(available_data), split)

        # Sample examples based on format
        if few_shot_format == "same_examples__no_variations":
            # Same examples for all rows, no variations - use first N examples
            sampled_data = available_data.head(count)
        elif few_shot_format == "same_examples__synchronized_order_variations":
            # Same examples for all rows, synchronized order variations
            # Use order_seed from identification_data if available for variations
            order_seed = identification_data.get('order_seed', current_row_idx) if identification_data else current_row_idx
            sampled_data = available_data.head(count).sample(frac=1.0, random_state=order_seed)
        elif few_shot_format == "different_examples__same_shuffling_order_across_rows":
            # Different examples per row, same shuffling order across rows
            # Use row-specific seed for example selection, but consistent shuffling
            selection_seed = current_row_idx
            sampled_data = available_data.sample(n=count, random_state=selection_seed)
            # Apply consistent shuffling if order_seed is provided
            if identification_data and 'order_seed' in identification_data:
                order_seed = identification_data.get('order_seed')
                sampled_data = sampled_data.sample(frac=1.0, random_state=order_seed)
        elif few_shot_format == "different_examples__different_order_per_variation":
            # Different examples and different order per variation
            # Use selection_seed from identification_data if available for variations
            selection_seed = identification_data.get('selection_seed', current_row_idx) if identification_data else current_row_idx
            sampled_data = available_data.sample(n=count, random_state=selection_seed)
        else:
            print(f"⚠️ Unknown few-shot format '{few_shot_format}', using 'same_examples__no_variations'")
            sampled_data = available_data.head(count)

        examples = []
        for _, example_row in sampled_data.iterrows():
            input_values = {}
            output_value = ""
            for col in example_row.index:
                if gold_field and col == gold_field:
                    from promptsuite.utils.formatting import convert_index_to_value
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
                            from promptsuite.augmentations.structure.enumerate import EnumeratorAugmenter
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
                                if enum_type == 'numbers':
                                    output_value = f"{gold_index + 1}. {options_list[gold_index].strip()}"
                                elif enum_type == 'capitals':
                                    letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
                                    if gold_index < len(letters):
                                        output_value = f"{letters[gold_index]}. {options_list[gold_index].strip()}"
                                elif enum_type == 'lowercase':
                                    letters = 'abcdefghijklmnopqrstuvwxyz'
                                    if gold_index < len(letters):
                                        output_value = f"{letters[gold_index]}. {options_list[gold_index].strip()}"
                                elif enum_type == 'roman':
                                    romans = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI', 'XII', 'XIII', 'XIV', 'XV', 'XVI', 'XVII', 'XVIII', 'XIX', 'XX', 'XXI', 'XXII', 'XXIII', 'XXIV', 'XXV', 'XXVI', 'XXVII', 'XXVIII', 'XXIX', 'XXX', 'XXXI', 'XXXII', 'XXXIII', 'XXXIV', 'XXXV', 'XXXVI', 'XXXVII', 'XXXVIII', 'XXXIX', 'XL']
                                    if gold_index < len(romans):
                                        output_value = f"{romans[gold_index]}. {options_list[gold_index].strip()}"
                                elif enum_type == 'greek':
                                    greek = 'αβγδεζηθικλμνξοπρστυφχψω'
                                    if gold_index < len(greek):
                                        output_value = f"{greek[gold_index]}. {options_list[gold_index].strip()}"
                                # Add more enum types as needed
                        except (ValueError, IndexError) as e:
                            print(f"⚠️ Error formatting enumerated gold value: {e}")
                else:
                    # Keep original field value for enumeration processing
                    original_field_value = example_row[col]
                    
                    # Apply enumeration if configured (before formatting)
                    if enumerate_configs and col in enumerate_configs:
                        enum_config = enumerate_configs[col]
                        enum_type = enum_config.get('type', '1234')
                        try:
                            from promptsuite.augmentations.structure.enumerate import EnumeratorAugmenter
                            enumerator = EnumeratorAugmenter()
                            # Pass the original value (could be list or string) directly to enumerate
                            field_value = enumerator.enumerate_field(original_field_value, enum_type)
                        except Exception as e:
                            print(f"⚠️ Error enumerating field '{col}' in few-shot example: {e}")
                            # Fallback to formatted original value
                            field_value = format_field_value(original_field_value)
                    else:
                        # No enumeration needed, just format the value
                        field_value = format_field_value(original_field_value)
                    
                    input_values[col] = field_value
            input_template = prompt_format_variant
            if gold_field:
                gold_placeholder = f'{{{gold_field}}}'
                input_template = input_template.replace(gold_placeholder, '').strip()
            input_text = self._fill_template_placeholders(input_template, input_values)
            if input_text:
                examples.append({
                    "input": input_text,
                    "output": output_value if output_value else ""
                })
        return examples

    def _filter_examples_by_category(
        self, 
        data: pd.DataFrame, 
        current_row: pd.Series,
        filter_column: str,
        count: int,
        fallback_strategy: str
    ) -> pd.DataFrame:
        """Filter few-shot examples by category/metadata."""
        
        if filter_column not in data.columns:
            print(f"⚠️ Filter column '{filter_column}' not found in data, using all available data")
            return data
        
        if filter_column not in current_row.index:
            print(f"⚠️ Filter column '{filter_column}' not found in current row, using all available data")
            return data
        
        current_category = current_row[filter_column]
        
        # Filter by category
        category_data = data[data[filter_column] == current_category]
        
        if len(category_data) >= count:
            return category_data
        
        # Handle fallback strategies
        if fallback_strategy == "global":
            remaining_needed = count - len(category_data)
            other_data = data[data[filter_column] != current_category]
            
            if len(other_data) > 0:
                # Sample the remaining needed examples from other categories
                sample_size = min(remaining_needed, len(other_data))
                other_sampled = other_data.sample(n=sample_size, random_state=42)
                return pd.concat([category_data, other_sampled])
            else:
                return category_data
        
        elif fallback_strategy == "strict":
            # Return only what we have from the category, even if it's less than count
            return category_data
        
        return category_data

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
    print("FewShotAugmenter is designed to work with the PromptSuiteEngine engine.")
    print("It requires few_shot_field configuration and structured data.")
    print("For standalone usage examples, please refer to the engine documentation.")
