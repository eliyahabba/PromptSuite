"""
MultiPromptify: A library for generating multi-prompt datasets from single-prompt datasets.
"""

import pandas as pd
import json
from typing import Dict, List, Any, Union, Optional

from .template_parser import TemplateParser
from src.axis_augmentation.augmentation_pipeline import AugmentationPipeline
from src.axis_augmentation.context_augmenter import ContextAugmenter
from src.axis_augmentation.fewshot_augmenter import FewShotAugmenter
from src.axis_augmentation.multidoc_augmenter import MultiDocAugmenter
from src.axis_augmentation.multiple_choice_augmenter import MultipleChoiceAugmenter
from src.axis_augmentation.paraphrase_instruct import Paraphrase
from src.axis_augmentation.text_surface_augmenter import TextSurfaceAugmenter


class MultiPromptify:
    """
    Main class for generating prompt variations based on templates.
    """
    
    # Mapping between template variation types and axis augmenters
    VARIATION_TYPE_TO_AUGMENTER = {
        # Original mapping for backwards compatibility
        "paraphrase": Paraphrase,  # Paraphrase variations
        "non-semantic": TextSurfaceAugmenter,  # Non-semantic / structural changes
        "lexical": Paraphrase,  # Word choice variations (using paraphrase)
        "syntactic": Paraphrase,  # Sentence structure variations (using paraphrase)
        "surface": TextSurfaceAugmenter,  # Surface-level formatting variations
        "few-shot": FewShotAugmenter,  # Few-shot examples
        "context": ContextAugmenter,  # Context variations
        "multiple-choice": MultipleChoiceAugmenter,  # Multiple choice variations
        "multidoc": MultiDocAugmenter,  # Multiple document variations
        
        # Extended mapping to match dimension names from simple_augmenter.py
        "Paraphrases": Paraphrase,
        "Non-semantic / structural changes": TextSurfaceAugmenter,
        "Which few-shot examples": FewShotAugmenter,
        "How many few-shot examples": FewShotAugmenter,
        "Add irrelevant context": ContextAugmenter,
        "Order of provided documents": MultiDocAugmenter,
        "Enumeration (letters, numbers, etc)": MultipleChoiceAugmenter,
        "Order of answers": MultipleChoiceAugmenter,
    }
    
    def __init__(self, max_variations: int = 100):
        """
        Initialize MultiPromptify.
        
        Args:
            max_variations: Maximum number of variations to generate
        """
        self.max_variations = max_variations
        self.template_parser = TemplateParser()
    
    def generate_variations(
        self,
        template: str,
        data: Union[pd.DataFrame, str, dict],
        instruction: str = None,
        variations_per_field: int = 3,
        api_key: str = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Generate prompt variations based on template and data.
        
        Args:
            template: Template string with variation annotations
            data: Input data (DataFrame, CSV file path, or dict)
            instruction: Static instruction text
            variations_per_field: Number of variations per field
            api_key: API key for services that need it (like paraphrase)
            **kwargs: Additional arguments
            
        Returns:
            List of variation dictionaries
        """
        # Load data if it's a file path
        if isinstance(data, str):
            if data.endswith('.csv'):
                data = pd.read_csv(data)
            elif data.endswith('.json'):
                with open(data, 'r') as f:
                    json_data = json.load(f)
                data = pd.DataFrame(json_data)
        elif isinstance(data, dict):
            data = pd.DataFrame(data)
        
        # Parse template to get fields and their variation types
        try:
            fields = self.template_parser.parse(template)
            variation_fields = self.template_parser.get_variation_fields()
            required_columns = self.template_parser.get_required_columns()
            few_shot_fields = self.template_parser.get_few_shot_fields()
            
            # Also parse instruction for variations if provided
            instruction_variation_fields = {}
            if instruction:
                instruction_fields = self.template_parser.parse(instruction)
                instruction_variation_fields = {
                    field.name: field.variation_type 
                    for field in instruction_fields 
                    if field.variation_type is not None
                }
                # Add instruction variation fields to main variation fields
                variation_fields.update(instruction_variation_fields)
                
                # Add required columns from instruction
                instruction_required = {
                    field.name for field in instruction_fields 
                    if not field.is_literal and field.name not in {'instruction', 'few_shot'}
                }
                required_columns.update(instruction_required)
                
        except Exception as e:
            raise ValueError(f"Template parsing error: {str(e)}")
        
        # Create clean template without variation types for formatting
        clean_template = self._create_clean_template(template)
        
        # Validate that required columns exist in data
        missing_columns = required_columns - set(data.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns in data: {missing_columns}")
        
        all_variations = []
        
        # Generate variations for each row in the data
        for row_idx, row in data.iterrows():
            # Create base prompt without variations first
            base_values = {}
            
            # Add instruction if provided
            if instruction:
                base_values['instruction'] = instruction
            
            # Handle few-shot examples from template syntax
            if few_shot_fields:
                few_shot_examples = self._generate_few_shot_examples(
                    few_shot_fields[0], data, row_idx, clean_template, instruction
                )
                if few_shot_examples:
                    base_values['few_shot'] = few_shot_examples
            
            # Add data fields
            for col in data.columns:
                if col in row and pd.notna(row[col]):
                    base_values[col] = str(row[col])
                else:
                    base_values[col] = ""
            
            # If no variation fields, just add the base prompt
            if not variation_fields:
                # Generate base prompt using clean template
                base_prompt = clean_template.format(**base_values)
                all_variations.append({
                    'prompt': base_prompt,
                    'original_row_index': row_idx,
                    'variation_count': 1,
                    'field_values': base_values.copy(),
                    'original_values': base_values.copy(),
                    'template': template
                })
                continue
            
            # Generate variations for fields that have variation types
            row_variations = self._generate_row_variations(
                clean_template,
                base_values,
                variation_fields,
                variations_per_field,
                row_idx,
                api_key
            )
            
            all_variations.extend(row_variations)
            
            # Stop if we've reached max variations
            if len(all_variations) >= self.max_variations:
                all_variations = all_variations[:self.max_variations]
                break
        
        return all_variations
    
    def _create_clean_template(self, template: str) -> str:
        """
        Create a clean template without variation type specifications for formatting.
        
        Args:
            template: Original template with variation annotations like {field:type} and few-shot syntax
            
        Returns:
            Clean template with just {field} placeholders
        """
        import re
        
        # First, handle few-shot syntax: convert {few_shot:N}, {few_shot:[N]}, etc. to {few_shot}
        few_shot_pattern = re.compile(r'\{few_shot:(?:(?:train|test)?(?:[\[\(])\d+(?:[\]\)])?|(?:train|test)?\d+)\}')
        clean_template = few_shot_pattern.sub('{few_shot}', template)
        
        # Then, replace regular {field:variation_type} with {field}
        clean_template = re.sub(r'\{([^:}]+):[^}]+\}', r'{\1}', clean_template)
        
        return clean_template
    
    def _generate_row_variations(
        self, 
        clean_template: str, 
        base_values: Dict[str, str], 
        variation_fields: Dict[str, str],
        variations_per_field: int,
        row_idx: int,
        api_key: str = None
    ) -> List[Dict[str, Any]]:
        """
        Generate variations for a single row.
        
        Args:
            clean_template: Template string without variation annotations (just {field} placeholders)
            base_values: Dictionary with field values (e.g., {'instruction': 'Please...', 'question': 'What is...', 'answer': 'B'})
            variation_fields: Dictionary mapping field names to variation types (e.g., {'instruction': 'semantic', 'question': 'paraphrase'})
            variations_per_field: Number of variations per field
            row_idx: Row index
            api_key: API key for services that need it
            
        Returns:
            List of variation dictionaries
        """
        variations = []
        
        # If no variation fields, return base prompt only
        if not variation_fields:
            variations.append({
                'prompt': clean_template.format(**base_values),
                'original_row_index': row_idx,
                'variation_count': 1,
                'field_values': base_values.copy(),
                'original_values': base_values.copy(),
                'template': clean_template
            })
            return variations
        
        # Step 1: Generate variations for each field that has a variation type
        field_variations = {}
        
        # First, add all base values as they are
        for field_name, field_value in base_values.items():
            if field_name not in variation_fields:
                # No variation for this field - use as is
                field_variations[field_name] = [field_value]
        
        # Step 2: For each field with variation type, generate variations
        for field_name, variation_type in variation_fields.items():
            if field_name not in base_values or not base_values[field_name]:
                # Skip if field not in base_values or empty
                field_variations[field_name] = [""]
                continue
            
            field_value = base_values[field_name]
            print(f"🔄 Generating {variation_type} variations for field '{field_name}' with value: '{field_value}'")
            
            # Get the appropriate augmenter class
            augmenter_class = self.VARIATION_TYPE_TO_AUGMENTER.get(
                variation_type, 
                TextSurfaceAugmenter
            )
            
            # Create augmenter with appropriate parameters
            if augmenter_class == Paraphrase and api_key:
                augmenter = augmenter_class(n_augments=variations_per_field, api_key=api_key)
            elif augmenter_class == FewShotAugmenter:
                augmenter = augmenter_class(
                    n_augments=variations_per_field,
                    num_examples=2,
                    mode="both"
                )
            else:
                augmenter = augmenter_class(n_augments=variations_per_field)
            
            # Generate variations for this field
            try:
                # Generate the variations
                field_variations_list = augmenter.augment(field_value)
                
                # Ensure we have a valid list of variations
                if not field_variations_list or not isinstance(field_variations_list, list):
                    field_variations_list = [field_value]
                
                # Ensure we have at least the original value
                if field_value not in field_variations_list:
                    field_variations_list = [field_value] + field_variations_list[:variations_per_field]
                else:
                    field_variations_list = field_variations_list[:variations_per_field + 1]
                
                # Make sure we have at least one variation
                if not field_variations_list:
                    field_variations_list = [field_value]
                
                field_variations[field_name] = field_variations_list
                
                print(f"✅ Generated {len(field_variations[field_name])} variations for '{field_name}'")
                
            except Exception as e:
                print(f"⚠️ Error generating variations for field '{field_name}': {str(e)}")
                # Fallback to original value
                field_variations[field_name] = [field_value]
        
        # Step 3: Generate combinations of all field variations
        print(f"🔧 Creating combinations from field variations...")
        
        # Debug: Print field variations
        print(f"📊 Field variations summary:")
        for field_name, variations_list in field_variations.items():
            print(f"  - {field_name}: {len(variations_list)} variations")
        
        # Get all combinations using itertools.product
        from itertools import product
        
        # Get field names in consistent order
        all_field_names = list(base_values.keys())
        all_field_variations = [field_variations.get(field, [""]) for field in all_field_names]
        
        # Debug: Print what we're about to combine
        print(f"📋 Fields to combine: {all_field_names}")
        print(f"📋 Variations per field: {[len(vars) for vars in all_field_variations]}")
        
        variation_count = 0
        max_combinations = min(50, self.max_variations)  # Reasonable limit
        
        # Debug: Check if we have any variations to combine
        total_possible_combinations = 1
        for variations_list in all_field_variations:
            total_possible_combinations *= len(variations_list)
        print(f"🧮 Total possible combinations: {total_possible_combinations}")
        
        for combination in product(*all_field_variations):
            if variation_count >= max_combinations:
                break
                
            # Create new values dict with this combination
            new_values = {}
            for i, field_name in enumerate(all_field_names):
                new_values[field_name] = combination[i]
            
            # Generate prompt with these values
            try:
                # Process complex instruction if it contains placeholders
                final_values = new_values.copy()
                if 'instruction' in final_values:
                    final_values['instruction'] = self._process_complex_instruction(
                        final_values['instruction'], 
                        {k: v for k, v in final_values.items() if k != 'instruction'}
                    )
                
                # For the main question, we need to create a template without the answer
                # Check if this is a template that should not show answer in main question
                clean_template_for_main = self._create_main_question_template(clean_template, final_values)
                
                new_prompt = clean_template_for_main.format(**final_values)
                
                variations.append({
                    'prompt': new_prompt,
                    'original_row_index': row_idx,
                    'variation_count': variation_count + 1,
                    'field_values': final_values.copy(),
                    'original_values': base_values.copy(),
                    'template': clean_template,
                    'varied_fields': list(variation_fields.keys())
                })
                variation_count += 1
                
            except KeyError as e:
                # Skip if template formatting fails
                print(f"⚠️ Template formatting failed: {str(e)}")
                continue
        
        print(f"✅ Generated {len(variations)} total combinations")
        return variations
    
    def _generate_few_shot_examples(self, few_shot_field, data: pd.DataFrame, current_row_idx: int, clean_template: str = None, instruction: str = None) -> str:
        """
        Generate few-shot examples based on template field configuration.
        
        Args:
            few_shot_field: TemplateField object with few-shot configuration
            data: Full dataset
            current_row_idx: Index of current row (to exclude from examples)
            clean_template: The clean template to extract format from
            instruction: The instruction string to extract format from if template doesn't have it
            
        Returns:
            Formatted few-shot examples string
        """
        if not few_shot_field.few_shot_count:
            return ""
        
        # Determine columns to use for input/output
        # Common patterns: question/answer, input/output, text/label, etc.
        input_cols = [col for col in data.columns if col.lower() in ['question', 'input', 'text', 'prompt']]
        output_cols = [col for col in data.columns if col.lower() in ['answer', 'output', 'label', 'response']]
        
        # Check for options column (for multiple choice)
        options_cols = [col for col in data.columns if col.lower() in ['options', 'choices']]
        
        # Fallback to first few columns if no standard patterns found
        if not input_cols and len(data.columns) >= 1:
            input_cols = [data.columns[0]]
        if not output_cols and len(data.columns) >= 2:
            output_cols = [data.columns[1]]
            
        if not input_cols or not output_cols:
            return ""
        
        input_col = input_cols[0]
        output_col = output_cols[0]
        options_col = options_cols[0] if options_cols else None
        
        # Extract the format from the clean template first
        question_format, answer_format = self._extract_qa_format_from_template(clean_template, input_col, output_col, options_col)
        
        # If we can't extract format from template, try from instruction
        if (question_format is None or answer_format is None) and instruction:
            question_format, answer_format = self._extract_qa_format_from_template(instruction, input_col, output_col, options_col)
        
        # If we still can't extract format, skip few-shot generation
        if question_format is None or answer_format is None:
            return ""
        
        # Filter data based on split if specified
        available_data = data.copy()
        if few_shot_field.few_shot_split and 'split' in data.columns:
            available_data = data[data['split'] == few_shot_field.few_shot_split]
        
        # Exclude current row
        available_data = available_data.drop(index=current_row_idx, errors='ignore')
        
        if len(available_data) == 0:
            return ""
        
        # Sample examples
        num_examples = min(few_shot_field.few_shot_count, len(available_data))
        if few_shot_field.few_shot_format == 'tuple':
            # Same examples for all rows - use fixed seed
            sampled_data = available_data.sample(n=num_examples, random_state=42)
        else:
            # Different examples per row - use row index as seed
            sampled_data = available_data.sample(n=num_examples, random_state=current_row_idx)
        
        # Format examples using the extracted format
        examples = []
        for _, example_row in sampled_data.iterrows():
            input_text = str(example_row[input_col])
            output_text = str(example_row[output_col])
            
            # Handle options for multiple choice
            if options_col and options_col in example_row:
                options_text = str(example_row[options_col])
                # For formats like "Question: {}\nOptions: {}\n", we need to provide both values
                if question_format.count('{}') == 2:
                    example = question_format.format(input_text, options_text) + answer_format.format(output_text)
                else:
                    # Fallback: use first placeholder for question, manually add options
                    question_part = question_format.format(input_text)
                    # Check if options line is separate
                    if 'Options:' in question_format:
                        example = question_part + answer_format.format(output_text)
                    else:
                        example = question_part + f"Options: {options_text}\n" + answer_format.format(output_text)
            else:
                # Use format without options
                example = question_format.format(input_text) + answer_format.format(output_text)
            
            examples.append(example)
        
        return "\n\n".join(examples)
    
    def _extract_qa_format_from_template(self, clean_template: str, input_col: str, output_col: str, options_col: str = None) -> tuple:
        """
        Extract the question/answer format from the template.
        
        Args:
            clean_template: Clean template string
            input_col: Name of the input column (e.g., 'question')
            output_col: Name of the output column (e.g., 'answer')
            options_col: Name of the options column (e.g., 'options')
            
        Returns:
            Tuple of (question_format, answer_format) where each format is a string with {} placeholder,
            or (None, None) if format cannot be extracted
        """
        if not clean_template:
            return None, None
        
        # Split template by lines to analyze each line
        lines = clean_template.split('\n')
        
        question_format_lines = []
        answer_format = None
        
        import re
        
        for line in lines:
            # Look for the input field (question) line
            input_patterns = [
                f'{{{input_col}}}',  # {question}
                re.escape(f'{{{input_col}:') + r'[^}]+}'  # {question:surface}
            ]
            
            question_line_found = False
            for pattern in input_patterns:
                if re.search(pattern, line):
                    # Found question line
                    question_line = re.sub(f'{{{re.escape(input_col)}(?::[^}}]+)?}}', '{}', line)
                    question_format_lines.append(question_line)
                    question_line_found = True
                    break
            
            # Check standard question field names if not found
            if not question_line_found:
                standard_inputs = ['question', 'input', 'text']
                for std_field in standard_inputs:
                    if std_field != input_col:
                        std_patterns = [
                            f'{{{std_field}}}',
                            re.escape(f'{{{std_field}:') + r'[^}]+}'
                        ]
                        for pattern in std_patterns:
                            if re.search(pattern, line):
                                question_line = re.sub(f'{{{re.escape(std_field)}(?::[^}}]+)?}}', '{}', line)
                                question_format_lines.append(question_line)
                                question_line_found = True
                                break
                        if question_line_found:
                            break
            
            # Look for options line if we have options_col
            if options_col:
                options_patterns = [
                    f'{{{options_col}}}',  # {options}
                    re.escape(f'{{{options_col}:') + r'[^}]+}'  # {options:multiple-choice}
                ]
                
                for pattern in options_patterns:
                    if re.search(pattern, line):
                        # Found options line
                        options_line = re.sub(f'{{{re.escape(options_col)}(?::[^}}]+)?}}', '{}', line)
                        question_format_lines.append(options_line)
                        break
            
            # Look for the output field (answer) line
            output_patterns = [
                f'{{{output_col}}}',  # {answer}
                re.escape(f'{{{output_col}:') + r'[^}]+}'  # {answer:variation}
            ]
            
            for pattern in output_patterns:
                if re.search(pattern, line):
                    # Found answer line
                    answer_format = re.sub(f'{{{re.escape(output_col)}(?::[^}}]+)?}}', '{}', line)
                    break
            
            # Check standard answer field names if not found
            if not answer_format:
                standard_outputs = ['answer', 'output', 'label']
                for std_field in standard_outputs:
                    if std_field != output_col:
                        std_patterns = [
                            f'{{{std_field}}}',
                            re.escape(f'{{{std_field}:') + r'[^}]+}'
                        ]
                        for pattern in std_patterns:
                            if re.search(pattern, line):
                                answer_format = re.sub(f'{{{re.escape(std_field)}(?::[^}}]+)?}}', '{}', line)
                                break
                        if answer_format:
                            break
        
        # Combine question format lines
        if question_format_lines:
            question_format = '\n'.join(question_format_lines) + '\n'
        else:
            question_format = None
            
        return question_format, answer_format
    
    def get_stats(self, variations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get statistics about the generated variations.
        
        Args:
            variations: List of variation dictionaries
            
        Returns:
            Dictionary with statistics
        """
        if not variations:
            return {}
        
        # Count variations per original row
        row_counts = {}
        for var in variations:
            row_idx = var.get('original_row_index', 0)
            row_counts[row_idx] = row_counts.get(row_idx, 0) + 1
        
        # Get unique fields
        all_fields = set()
        for var in variations:
            all_fields.update(var.get('field_values', {}).keys())
        
        return {
            'total_variations': len(variations),
            'original_rows': len(row_counts),
            'avg_variations_per_row': sum(row_counts.values()) / len(row_counts) if row_counts else 0,
            'unique_fields': len(all_fields),
            'field_names': list(all_fields),
            'min_variations_per_row': min(row_counts.values()) if row_counts else 0,
            'max_variations_per_row': max(row_counts.values()) if row_counts else 0,
        }
    
    def parse_template(self, template: str) -> Dict[str, str]:
        """
        Parse template to extract fields and their variation types.
        
        Args:
            template: Template string
            
        Returns:
            Dictionary mapping field names to variation types
        """
        try:
            self.template_parser.parse(template)
            return self.template_parser.get_variation_fields()
        except Exception as e:
            raise ValueError(f"Template parsing error: {str(e)}")
    
    def save_variations(
        self,
        variations: List[Dict[str, Any]],
        output_path: str,
        format: str = "json"
    ):
        """
        Save variations to file.
        
        Args:
            variations: List of variation dictionaries
            output_path: Output file path
            format: Output format ('json', 'csv', 'txt')
        """
        if format == "json":
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(variations, f, indent=2, ensure_ascii=False)
        
        elif format == "csv":
            # Flatten for CSV
            flattened = []
            for var in variations:
                flat_var = {
                    'prompt': var['prompt'],
                    'original_row_index': var.get('original_row_index', ''),
                    'variation_count': var.get('variation_count', ''),
                }
                # Add field values
                for key, value in var.get('field_values', {}).items():
                    flat_var[f'field_{key}'] = value
                flattened.append(flat_var)
            
            df = pd.DataFrame(flattened)
            df.to_csv(output_path, index=False, encoding='utf-8')
        
        elif format == "txt":
            with open(output_path, 'w', encoding='utf-8') as f:
                for i, var in enumerate(variations):
                    f.write(f"=== Variation {i+1} ===\n")
                    f.write(var['prompt'])
                    f.write("\n\n")
        
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _process_complex_instruction(self, instruction: str, field_values: Dict[str, str]) -> str:
        """
        Process instruction that may contain placeholders and variations for other fields.
        
        Args:
            instruction: Instruction string that may contain placeholders with variations like {question:surface}, {answer}, etc.
            field_values: Dictionary of field values to substitute
            
        Returns:
            Processed instruction with placeholders and variations replaced by actual values
        """
        if not instruction:
            return instruction
            
        processed_instruction = instruction
        
        # Use template parser to find all fields in the instruction
        try:
            instruction_fields = self.template_parser.parse(instruction)
            
            # Create a clean instruction template (remove variation types)
            clean_instruction = instruction
            for field in instruction_fields:
                if field.variation_type:
                    # Replace {field:variation} with {field}
                    old_pattern = f'{{{field.name}:{field.variation_type}}}'
                    new_pattern = f'{{{field.name}}}'
                    clean_instruction = clean_instruction.replace(old_pattern, new_pattern)
            
            # Now substitute the actual field values
            for field_name, field_value in field_values.items():
                placeholder = f'{{{field_name}}}'
                if placeholder in clean_instruction:
                    clean_instruction = clean_instruction.replace(placeholder, str(field_value))
            
            return clean_instruction
            
        except Exception as e:
            # Fallback to simple replacement if parsing fails
            for field_name, field_value in field_values.items():
                placeholder = f'{{{field_name}}}'
                if placeholder in processed_instruction:
                    processed_instruction = processed_instruction.replace(placeholder, str(field_value))
            
            return processed_instruction 

    def _create_main_question_template(self, clean_template: str, final_values: Dict[str, str]) -> str:
        """
        Create a main question template without the answer.
        
        Args:
            clean_template: Clean template string
            final_values: Dictionary of final values to substitute
            
        Returns:
            Main question template without the answer
        """
        if not clean_template:
            return ""
        
        import re
        
        # Split the template to identify the main question part vs few-shot part
        template_parts = clean_template.split('{few_shot}')
        
        if len(template_parts) == 2:
            # Template has few-shot examples
            instruction_part = template_parts[0].strip()
            main_question_part = template_parts[1].strip()
            
            # Check if main question part has answer field - if so, remove the answer line
            lines = main_question_part.split('\n')
            filtered_lines = []
            
            for line in lines:
                # Skip lines that only contain the answer field
                if re.search(r'^\s*Answer:\s*\{answer\}\s*$', line) or re.search(r'^\s*\{answer\}\s*$', line):
                    continue
                filtered_lines.append(line)
            
            main_question_part = '\n'.join(filtered_lines)
            
            # Reconstruct template
            if 'few_shot' in final_values and final_values['few_shot']:
                result = instruction_part + '\n\n' + final_values['few_shot'] + '\n\n' + main_question_part
            else:
                result = instruction_part + '\n\n' + main_question_part
            
            return result
        else:
            # No few-shot, but still need to remove answer from main template
            lines = clean_template.split('\n')
            filtered_lines = []
            
            for line in lines:
                # Skip lines that only contain the answer field
                if re.search(r'^\s*Answer:\s*\{answer\}\s*$', line) or re.search(r'^\s*\{answer\}\s*$', line):
                    continue
                filtered_lines.append(line)
            
            return '\n'.join(filtered_lines) 