"""
MultiPromptify: A library for generating multi-prompt datasets from single-prompt datasets.
"""

import json
from typing import Dict, List, Any

import pandas as pd

from src.multipromptify.augmentations.structure.shuffle import ShuffleAugmenter
from src.multipromptify.augmentations.text.context import ContextAugmenter
from src.multipromptify.augmentations.text.paraphrase import Paraphrase
from src.multipromptify.augmentations.text.surface import TextSurfaceAugmenter
from src.multipromptify.template_parser import TemplateParser


class MultiPromptify:
    """
    Main class for generating prompt variations based on dictionary templates.
    
    Template format:
    {
        "instruction_template": "Answer the following question: {question}\nAnswer: {answer}",
        "instruction": ["paraphrase", "surface"],
        "gold": "answer",  # Name of the column containing the correct answer/label
        "few_shot": {
            "count": 2,
            "format": "fixed",  # or "rotating"
            "split": "train"    # or "test" or "all"
        },
        "question": ["surface"]
    }
    """

    VARIATION_TYPE_TO_AUGMENTER = {
        "paraphrase": Paraphrase,
        "surface": TextSurfaceAugmenter,
        "context": ContextAugmenter,
        "shuffle": ShuffleAugmenter,
    }

    def __init__(self, max_variations: int = 100):
        """Initialize MultiPromptify with maximum variations limit."""
        self.max_variations = max_variations
        self.template_parser = TemplateParser()

    def generate_variations(
            self,
            template: dict,
            data: pd.DataFrame,
            variations_per_field: int = 3,
            api_key: str = None,
            **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Generate prompt variations based on dictionary template and data.
        
        Args:
            template: Dictionary template with field configurations
            data: DataFrame with the data
            variations_per_field: Number of variations per field
            api_key: API key for services that require it
        
        Returns:
            List of generated variations
        """
        # Validate template
        is_valid, errors = self.template_parser.validate_template(template)
        if not is_valid:
            raise ValueError(f"Invalid template: {', '.join(errors)}")

        # Load data if needed
        if isinstance(data, str):
            data = self._load_data(data)

        # Parse template
        fields = self.template_parser.parse(template)
        variation_fields = self.template_parser.get_variation_fields()
        few_shot_fields = self.template_parser.get_few_shot_fields()

        # Get gold field (answer column name) from template
        gold_config = template.get('gold', None)

        # Handle both old format (string) and new format (dict)
        if isinstance(gold_config, str):
            # Backward compatibility: old format
            gold_field = gold_config
            gold_type = 'value'
            options_field = None
        elif isinstance(gold_config, dict):
            # New format
            gold_field = gold_config.get('field')
            gold_type = gold_config.get('type', 'value')
            options_field = gold_config.get('options_field')
        else:
            gold_field = None
            gold_type = 'value'
            options_field = None

        # Get instruction template from user - required
        instruction_template = self.template_parser.get_instruction_template()
        if not instruction_template:
            raise ValueError(
                "Instruction template is required. Please specify 'instruction_template' in your template. "
                "Example: \"instruction_template\": \"Answer the question: {question}\\nAnswer: {answer}\""
            )

        # Validate gold field requirement
        self._validate_gold_field_requirement(instruction_template, gold_field, few_shot_fields)

        all_variations = []

        # For each data row
        for row_idx, row in data.iterrows():
            if len(all_variations) >= self.max_variations:
                break

            # Generate variations for all fields
            field_variations = self._generate_all_field_variations(
                instruction_template, variation_fields, row, variations_per_field, api_key,
                gold_field, gold_type, options_field
            )

            # Instead of generating few_shot_examples here, pass few_shot_fields[0] and data to the next function
            row_variations = self._create_row_variations(
                field_variations, row, gold_field, few_shot_fields[0] if few_shot_fields else None, template, row_idx,
                gold_type, data, options_field
            )

            all_variations.extend(row_variations)

            if len(all_variations) >= self.max_variations:
                break

        return all_variations[:self.max_variations]

    def _load_data(self, data_path: str) -> pd.DataFrame:
        """Load data from file path."""
        if data_path.endswith('.csv'):
            return pd.read_csv(data_path)
        elif data_path.endswith('.json'):
            with open(data_path, 'r') as f:
                json_data = json.load(f)
            return pd.DataFrame(json_data)
        else:
            raise ValueError(f"Unsupported file format: {data_path}")

    def _generate_instruction_variations(
            self,
            instruction_template: str,
            variation_fields: Dict[str, List[str]],
            variations_per_field: int,
            api_key: str
    ) -> List[str]:
        """Generate variations of the instruction template."""

        if 'instruction' not in variation_fields or not variation_fields['instruction']:
            return [instruction_template]

        variation_types = variation_fields['instruction']
        all_variations = []

        # Generate variations for each type
        for variation_type in variation_types:
            try:
                augmenter_class = self.VARIATION_TYPE_TO_AUGMENTER.get(
                    variation_type, TextSurfaceAugmenter
                )

                if augmenter_class == Paraphrase and api_key:
                    augmenter = augmenter_class(n_augments=variations_per_field, api_key=api_key)
                else:
                    augmenter = augmenter_class(n_augments=variations_per_field)

                variations = augmenter.augment(instruction_template)

                if variations and isinstance(variations, list):
                    # Handle potential dict return from certain augmenters
                    string_variations = []
                    for var in variations:
                        if isinstance(var, dict):
                            # Extract text data from dict
                            if 'shuffled_data' in var:
                                string_variations.append(var['shuffled_data'])
                            elif 'data' in var:
                                string_variations.append(var['data'])
                            elif 'text' in var:
                                string_variations.append(var['text'])
                            else:
                                string_variations.append(str(var))
                        else:
                            # Standard string return
                            string_variations.append(var)
                    
                    all_variations.extend(string_variations[:variations_per_field])

            except Exception as e:
                print(f"⚠️ Error generating {variation_type} variations: {e}")
                continue

        # Remove duplicates while preserving order
        unique_variations = []
        seen = set()
        for var in all_variations:
            if var not in seen:
                unique_variations.append(var)
                seen.add(var)

        # Ensure original is included first
        if instruction_template not in unique_variations:
            unique_variations.insert(0, instruction_template)

        return unique_variations[:variations_per_field + 1]

    def _validate_gold_field_requirement(self, instruction_template: str, gold_field: str, few_shot_fields: list):
        """Validate that gold field is provided when needed for few-shot examples."""

        # Check if few-shot is configured (needs to separate question from answer)
        if few_shot_fields and len(few_shot_fields) > 0 and not gold_field:
            raise ValueError(
                "Gold field is required when using few-shot examples. "
                "Please specify the 'gold' field in your template to indicate which column contains the correct answers/labels. "
                "Example: \"gold\": \"answer\" or \"gold\": \"label\""
            )

    def _generate_few_shot_examples_structured(self, few_shot_field, instruction_variant: str, data: pd.DataFrame,
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
            # Not enough data for few-shot examples
            return []

        # Sample examples
        if few_shot_format == "fixed":
            # Use the same first N examples for all questions
            sampled_data = available_data.head(count)
        else:
            # Randomly sample different examples for each question
            sampled_data = available_data.sample(n=count, random_state=current_row_idx)

        examples = []

        for _, example_row in sampled_data.iterrows():
            # Create row values for few-shot examples (including extracting real answer)
            row_values = {}
            for col in example_row.index:
                # Handle pandas array comparison issue
                try:
                    is_not_na = pd.notna(example_row[col])
                    if hasattr(is_not_na, '__len__') and len(is_not_na) > 1:
                        # For arrays/lists, check if any element is not na
                        is_not_na = is_not_na.any() if hasattr(is_not_na, 'any') else True
                except (ValueError, TypeError):
                    # Fallback: assume not na if we can't check
                    is_not_na = example_row[col] is not None
                
                if is_not_na:
                    if gold_field and col == gold_field:
                        # For few-shot examples, extract the actual answer value from options
                        answer_value = self._extract_answer_from_options(
                            example_row, gold_field, gold_type, options_field
                        )
                        row_values[col] = answer_value
                    else:
                        row_values[col] = str(example_row[col])

            # Fill template with all placeholders (including extracted answer)
            question = self._fill_template_placeholders(instruction_variant, row_values)

            if question:
                examples.append({
                    "question": question.strip(),
                    "answer": ""
                })

        return examples

    def _extract_answer_from_options(self, row: pd.Series, gold_field: str, gold_type: str,
                                     options_field: str = None) -> str:
        """Extract the actual answer text from options based on the gold field."""

        if not gold_field or gold_field not in row.index:
            return str(row.get(gold_field, ''))

        gold_value = row[gold_field]

        # If gold_type is 'value', return as is
        if gold_type == 'value':
            return str(gold_value)

        # If gold_type is 'index', try to extract from options
        if gold_type == 'index' and options_field and options_field in row.index:
            try:
                from .augmentations.structure.shuffle import ShuffleAugmenter
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
        return str(gold_value)

    def _format_few_shot_as_string(self, few_shot_examples: List[Dict[str, str]]) -> str:
        """Format few-shot examples as string."""
        if not few_shot_examples:
            return ""

        formatted_examples = []
        for example in few_shot_examples:
            # Combine question and answer for the traditional prompt format
            formatted_example = f"{example['question']}\n{example['answer']}"
            formatted_examples.append(formatted_example)

        return "\n\n".join(formatted_examples)

    def _generate_few_shot_examples(self, few_shot_field, instruction_variant: str, data: pd.DataFrame,
                                    current_row_idx: int, gold_field: str = None) -> str:
        """Generate few-shot examples using the configured parameters. (Legacy method for backward compatibility)"""

        # Use the new structured method and convert to string
        structured_examples = self._generate_few_shot_examples_structured(few_shot_field, instruction_variant, data,
                                                                          current_row_idx, gold_field)
        return self._format_few_shot_as_string(structured_examples)

    def _create_main_question(self, instruction_variant: str, row: pd.Series, gold_field: str = None) -> str:
        """Create main question by filling instruction with row data (excluding answers)."""

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
                # Skip the gold answer field for the main question
                if gold_field and col == gold_field:
                    continue
                else:
                    row_values[col] = str(row[col])

        # Fill template and remove the gold field placeholder completely
        question = self._fill_template_placeholders(instruction_variant, row_values)

        # Remove any remaining gold field placeholder
        if gold_field:
            question = question.replace(f'{{{gold_field}}}', '')

        return question.strip()

    def _fill_template_placeholders(self, template: str, values: Dict[str, str]) -> str:
        """Fill template placeholders with values."""
        if not template:
            return ""

        result = template
        for field_name, field_value in values.items():
            placeholder = f'{{{field_name}}}'
            if placeholder in result:
                result = result.replace(placeholder, str(field_value))

        return result

    def get_stats(self, variations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about generated variations."""
        if not variations:
            return {}

        row_counts = {}
        for var in variations:
            row_idx = var.get('original_row_index', 0)
            row_counts[row_idx] = row_counts.get(row_idx, 0) + 1

        # Get field info from template config
        template_config = variations[0].get('template_config', {})
        field_count = len([k for k in template_config.keys() if k not in ['few_shot', 'instruction_template']])
        has_few_shot = 'few_shot' in template_config
        has_custom_instruction = 'instruction_template' in template_config

        return {
            'total_variations': len(variations),
            'original_rows': len(row_counts),
            'avg_variations_per_row': sum(row_counts.values()) / len(row_counts) if row_counts else 0,
            'template_fields': field_count,
            'has_few_shot': has_few_shot,
            'has_custom_instruction': has_custom_instruction,
            'min_variations_per_row': min(row_counts.values()) if row_counts else 0,
            'max_variations_per_row': max(row_counts.values()) if row_counts else 0,
        }

    def parse_template(self, template: dict) -> Dict[str, List[str]]:
        """Parse template to extract fields and their variation types."""
        self.template_parser.parse(template)
        return self.template_parser.get_variation_fields()

    def save_variations(self, variations: List[Dict[str, Any]], output_path: str, format: str = "json"):
        """Save variations to file."""
        if format == "json":
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(variations, f, indent=2, ensure_ascii=False)

        elif format == "csv":
            flattened = []
            for var in variations:
                flat_var = {
                    'prompt': var['prompt'],
                    'original_row_index': var.get('original_row_index', ''),
                    'variation_count': var.get('variation_count', ''),
                }
                for key, value in var.get('field_values', {}).items():
                    flat_var[f'field_{key}'] = value
                flattened.append(flat_var)

            df = pd.DataFrame(flattened)
            df.to_csv(output_path, index=False, encoding='utf-8')

        elif format == "txt":
            with open(output_path, 'w', encoding='utf-8') as f:
                for i, var in enumerate(variations):
                    f.write(f"=== Variation {i + 1} ===\n")
                    f.write(var['prompt'])
                    f.write("\n\n")

        else:
            raise ValueError(f"Unsupported format: {format}")

    def _generate_all_field_variations(
            self,
            instruction_template: str,
            variation_fields: Dict[str, List[str]],
            row: pd.Series,
            variations_per_field: int,
            api_key: str,
            gold_field: str = None,
            gold_type: str = 'value',
            options_field: str = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Generate variations for all fields that have variation types specified."""

        field_variations = {}

        # Generate instruction variations
        if 'instruction' in variation_fields and variation_fields['instruction']:
            instruction_vars = self._generate_instruction_variations(
                instruction_template, variation_fields, variations_per_field, api_key
            )
            # Convert to new format
            field_variations['instruction'] = [{'data': var, 'gold_update': None} for var in instruction_vars]
        else:
            field_variations['instruction'] = [{'data': instruction_template, 'gold_update': None}]

        # Generate variations for other fields
        for field_name, variation_types in variation_fields.items():
            if field_name == 'instruction':
                continue  # Already handled above

            # Handle pandas array comparison issue
            try:
                is_not_na = pd.notna(row[field_name]) if field_name in row.index else False
                if hasattr(is_not_na, '__len__') and len(is_not_na) > 1:
                    # For arrays/lists, check if any element is not na
                    is_not_na = is_not_na.any() if hasattr(is_not_na, 'any') else True
            except (ValueError, TypeError):
                # Fallback: assume not na if we can't check
                is_not_na = field_name in row.index and row[field_name] is not None
            
            if field_name in row.index and is_not_na:
                field_value = str(row[field_name])
                field_variations[field_name] = self._generate_field_variations(
                    field_name, field_value, variation_types, variations_per_field, api_key, row, gold_field, gold_type,
                    options_field
                )
            else:
                # If field not in data, use empty variations
                field_variations[field_name] = [{'data': '', 'gold_update': None}]

        return field_variations

    def _generate_field_variations(
            self,
            field_name: str,
            field_value: str,
            variation_types: List[str],
            variations_per_field: int,
            api_key: str,
            row: pd.Series = None,
            gold_field: str = None,
            gold_type: str = 'value',
            options_field: str = None
    ) -> List[Dict[str, Any]]:
        """Generate variations for a specific field."""

        all_variations = [{'data': field_value, 'gold_update': None}]  # Start with original

        for variation_type in variation_types:
            try:
                augmenter_class = self.VARIATION_TYPE_TO_AUGMENTER.get(
                    variation_type, TextSurfaceAugmenter
                )

                if augmenter_class == Paraphrase and api_key:
                    augmenter = augmenter_class(n_augments=variations_per_field, api_key=api_key)
                else:
                    augmenter = augmenter_class(n_augments=variations_per_field)

                # Special handling for shuffle augmenter
                if variation_type == 'shuffle':
                    if not gold_field or not row.index.tolist() or gold_field not in row.index:
                        print(f"⚠️ Shuffle augmenter requires gold field '{gold_field}' to be present in data")
                        continue

                    # Prepare identification data based on gold type
                    if gold_type == 'index':
                        # For index-based gold, pass the index directly
                        try:
                            gold_index = int(row[gold_field])
                            identification_data = {
                                'gold_field': gold_field,
                                'gold_value': str(gold_index)
                            }
                        except (ValueError, TypeError):
                            print(
                                f"⚠️ Gold field '{gold_field}' must contain valid integer indices for shuffle operation")
                            continue
                    else:
                        # For value-based gold, pass the value and let augmenter find the index
                        identification_data = {
                            'gold_field': gold_field,
                            'gold_value': str(row[gold_field])
                        }

                    variations = augmenter.augment(field_value, identification_data)

                    if variations and isinstance(variations, list):
                        for var in variations:
                            if isinstance(var, dict) and 'shuffled_data' in var and 'new_gold_index' in var:
                                # For index-based gold, update with new index
                                # For value-based gold, convert index back to value if needed
                                if gold_type == 'index':
                                    gold_update_value = var['new_gold_index']
                                else:
                                    # For value-based, we might need to extract the actual value
                                    # from the shuffled options, but for now keep the index
                                    gold_update_value = var['new_gold_index']

                                variation_data = {
                                    'data': var['shuffled_data'],
                                    'gold_update': {gold_field: gold_update_value}
                                }
                                if variation_data not in all_variations:
                                    all_variations.append(variation_data)
                else:
                    # Regular augmenters
                    variations = augmenter.augment(field_value)

                    if variations and isinstance(variations, list):
                        # Add new variations (excluding original if already present)
                        for var in variations:
                            # Handle potential dict return from certain augmenters
                            if isinstance(var, dict):
                                # Extract text data from dict
                                if 'shuffled_data' in var:
                                    text_data = var['shuffled_data']
                                elif 'data' in var:
                                    text_data = var['data']
                                elif 'text' in var:
                                    text_data = var['text']
                                else:
                                    text_data = str(var)
                                variation_data = {'data': text_data, 'gold_update': None}
                            else:
                                # Standard string return
                                variation_data = {'data': var, 'gold_update': None}
                            
                            if variation_data not in all_variations:
                                all_variations.append(variation_data)

            except Exception as e:
                print(f"⚠️ Error generating {variation_type} variations for field {field_name}: {e}")
                continue

        # Remove duplicates while preserving order and limit to variations_per_field + 1 (original)
        unique_variations = []
        seen = set()
        for var in all_variations:
            var_key = (var['data'], str(var['gold_update']))
            if var_key not in seen:
                unique_variations.append(var)
                seen.add(var_key)

        return unique_variations[:variations_per_field + 1]

    def _create_row_variations(
            self,
            field_variations: Dict[str, List[Dict[str, Any]]],
            row: pd.Series,
            gold_field: str,
            few_shot_field,  # instead of few_shot_examples
            template: dict,
            row_idx: int,
            gold_type: str = 'value',
            data: pd.DataFrame = None,  # also pass the data
            options_field: str = None
    ) -> List[Dict[str, Any]]:
        import itertools
        variations = []
        varying_fields = list(field_variations.keys())
        if varying_fields:
            variation_combinations = list(itertools.product(*[field_variations[field] for field in varying_fields]))
            for combination in variation_combinations:
                if len(variations) >= self.max_variations:
                    break
                field_values = dict(zip(varying_fields, combination))
                instruction_variant = field_values.get('instruction', field_variations.get('instruction', [
                    {'data': '', 'gold_update': None}])[0])['data']
                row_values = {}
                gold_updates = {}
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
                        if col in field_values:
                            field_data = field_values[col]
                            row_values[col] = field_data['data']
                            if field_data['gold_update']:
                                gold_updates.update(field_data['gold_update'])
                        elif gold_field and col == gold_field:
                            continue
                        else:
                            row_values[col] = str(row[col])
                # Generate few-shot examples for each instruction_variant
                few_shot_examples = []
                if few_shot_field and data is not None:
                    few_shot_examples = self._generate_few_shot_examples_structured(
                        few_shot_field, instruction_variant, data, row_idx,
                        gold_field, gold_type, options_field
                    )
                main_question = self._fill_template_placeholders(instruction_variant, row_values)
                if gold_field:
                    main_question = main_question.replace(f'{{{gold_field}}}', '')
                main_question = main_question.strip()
                conversation_messages = []
                for example in few_shot_examples:
                    conversation_messages.append({
                        "role": "user",
                        "content": example["question"]
                    })
                    conversation_messages.append({
                        "role": "assistant",
                        "content": example["answer"]
                    })
                if main_question:
                    conversation_messages.append({
                        "role": "user",
                        "content": main_question
                    })
                prompt_parts = []
                if few_shot_examples:
                    few_shot_content = self._format_few_shot_as_string(few_shot_examples)
                    prompt_parts.append(few_shot_content)
                if main_question:
                    prompt_parts.append(main_question)
                final_prompt = '\n\n'.join(prompt_parts)
                output_field_values = {}
                for field_name, field_data in field_values.items():
                    output_field_values[field_name] = field_data['data']
                variations.append({
                    'prompt': final_prompt,
                    'conversation': conversation_messages,
                    'original_row_index': row_idx,
                    'variation_count': len(variations) + 1,
                    'template_config': template,
                    'field_values': output_field_values,
                    'gold_updates': gold_updates if gold_updates else None,
                })
        return variations
