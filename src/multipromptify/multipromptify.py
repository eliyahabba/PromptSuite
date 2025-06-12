"""
MultiPromptify: A library for generating multi-prompt datasets from single-prompt datasets.
"""

import pandas as pd
import json
from typing import Dict, List, Any, Optional

from .template_parser import TemplateParser
from src.augmentations.context_augmenter import ContextAugmenter
from src.augmentations.fewshot_augmenter import FewShotAugmenter
from src.augmentations.multidoc_augmenter import MultiDocAugmenter
from src.augmentations.multiple_choice_augmenter import MultipleChoiceAugmenter
from src.augmentations.paraphrase_instruct import Paraphrase
from src.augmentations.text_surface_augmenter import TextSurfaceAugmenter


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
        "multiple-choice": MultipleChoiceAugmenter,
        "multidoc": MultiDocAugmenter,
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
        gold_field = template.get('gold', None)
        
        # Get instruction template from user - required
        instruction_template = self.template_parser.get_instruction_template()
        if not instruction_template:
            raise ValueError(
                "Instruction template is required. Please specify 'instruction_template' in your template. "
                "Example: \"instruction_template\": \"Answer the question: {question}\\nAnswer: {answer}\""
            )
        
        # Validate gold field requirement
        self._validate_gold_field_requirement(instruction_template, gold_field, few_shot_fields)
        
        # Generate instruction variations if needed
        instruction_variations = self._generate_instruction_variations(
            instruction_template, variation_fields, variations_per_field, api_key
        )
        
        all_variations = []
        
        # For each instruction variation
        for instruction_variant in instruction_variations:
            
            # For each data row
            for row_idx, row in data.iterrows():
                if len(all_variations) >= self.max_variations:
                    break
                
                # Generate few-shot examples if configured
                few_shot_examples = []
                if few_shot_fields:
                    few_shot_examples = self._generate_few_shot_examples_structured(
                        few_shot_fields[0], instruction_variant, data, row_idx, gold_field
                    )
                
                # Create main question (without answer)
                main_question = self._create_main_question(instruction_variant, row, gold_field)
                
                # Build conversation structure
                conversation_messages = []
                
                # Add few-shot examples as conversation history
                for example in few_shot_examples:
                    conversation_messages.append({
                        "role": "user",
                        "content": example["question"]
                    })
                    conversation_messages.append({
                        "role": "assistant", 
                        "content": example["answer"]
                    })
                
                # Add main question as final user message
                if main_question:
                    conversation_messages.append({
                        "role": "user",
                        "content": main_question
                    })
                
                # Build traditional prompt format for backward compatibility
                prompt_parts = []
                if few_shot_examples:
                    few_shot_content = self._format_few_shot_as_string(few_shot_examples)
                    prompt_parts.append(few_shot_content)
                if main_question:
                    prompt_parts.append(main_question)
                
                final_prompt = '\n\n'.join(prompt_parts)
                
                all_variations.append({
                    'prompt': final_prompt,
                    'conversation': conversation_messages,
                    'original_row_index': row_idx,
                    'variation_count': len(all_variations) + 1,
                    'template_config': template,
                    'field_values': {
                        'instruction': instruction_variant, 
                        'few_shot': few_shot_examples
                    },
                })
                
                if len(all_variations) >= self.max_variations:
                    break
        
        return all_variations
    
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
                    all_variations.extend(variations[:variations_per_field])
                    
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
    
    def _generate_few_shot_examples_structured(self, few_shot_field, instruction_variant: str, data: pd.DataFrame, current_row_idx: int, gold_field: str = None) -> List[Dict[str, str]]:
        """Generate few-shot examples as structured conversation data."""
        
        num_examples = few_shot_field.few_shot_count
        
        # Filter data based on split configuration
        if few_shot_field.few_shot_split == 'train' and 'split' in data.columns:
            available_data = data[data['split'] == 'train']
        elif few_shot_field.few_shot_split == 'test' and 'split' in data.columns:
            available_data = data[data['split'] == 'test']
        else:
            available_data = data.copy()
        
        # Exclude current row
        available_data = available_data.drop(index=current_row_idx, errors='ignore')
        
        if len(available_data) == 0:
            return []
        
        # Sample examples based on format
        if few_shot_field.few_shot_format == 'fixed':
            # Same examples for all rows
            sampled_data = available_data.sample(
                n=min(num_examples, len(available_data)), 
                random_state=42
            )
        else:
            # Different examples per row (rotating)
            sampled_data = available_data.sample(
                n=min(num_examples, len(available_data)), 
                random_state=current_row_idx
            )
        
        # Create structured examples
        examples = []
        for _, example_row in sampled_data.iterrows():
            # Fill all placeholders including the gold field with real values
            all_values = {}
            for col in example_row.index:
                if pd.notna(example_row[col]):
                    all_values[col] = str(example_row[col])
            
            # For few-shot examples, fill everything including the gold field
            question_with_answer = self._fill_template_placeholders(instruction_variant, all_values)
            
            if question_with_answer:
                examples.append({
                    "question": question_with_answer,
                    "answer": ""  # Not used in this context
                })
        
        return examples
    
    def _format_few_shot_as_string(self, few_shot_examples: List[Dict[str, str]]) -> str:
        """Convert structured few-shot examples back to string format for backward compatibility."""
        if not few_shot_examples:
            return ""
        
        formatted_examples = []
        for example in few_shot_examples:
            # Format as a complete prompt with question and answer
            formatted_example = example['question']
            formatted_examples.append(formatted_example)
        
        return "\n\n".join(formatted_examples)
    
    def _generate_few_shot_examples(self, few_shot_field, instruction_variant: str, data: pd.DataFrame, current_row_idx: int, gold_field: str = None) -> str:
        """Generate few-shot examples using the configured parameters. (Legacy method for backward compatibility)"""
        
        # Use the new structured method and convert to string
        structured_examples = self._generate_few_shot_examples_structured(few_shot_field, instruction_variant, data, current_row_idx, gold_field)
        return self._format_few_shot_as_string(structured_examples)
    
    def _create_main_question(self, instruction_variant: str, row: pd.Series, gold_field: str = None) -> str:
        """Create main question by filling instruction with row data (excluding answers)."""
        
        row_values = {}
        for col in row.index:
            if pd.notna(row[col]):
                # Skip the gold answer field for the main question
                if gold_field and col == gold_field:
                    continue
                else:
                    row_values[col] = str(row[col])
        
        # Fill template and remove the gold field placeholder completely
        question = self._fill_template_placeholders(instruction_variant, row_values)
        
        # Remove the gold field placeholder completely (including its text/formatting)  
        if gold_field:
            import re
            # Remove {gold_field} placeholder and any text that comes before it on the same line
            gold_placeholder_pattern = f'[^\\n]*\\{{{gold_field}\\}}[^\\n]*'
            question = re.sub(gold_placeholder_pattern, '', question)
            # Clean up any trailing newlines or whitespace
            question = re.sub(r'\n+$', '', question)
        
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
                    f.write(f"=== Variation {i+1} ===\n")
                    f.write(var['prompt'])
                    f.write("\n\n")
        
        else:
            raise ValueError(f"Unsupported format: {format}") 