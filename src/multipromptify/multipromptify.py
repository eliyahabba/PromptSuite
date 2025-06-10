"""
MultiPromptify: A library for generating multi-prompt datasets from single-prompt datasets.
"""

import pandas as pd
import json
import re
from typing import Dict, List, Any, Union, Optional
from itertools import product

from .template_parser import TemplateParser
from src.axis_augmentation.context_augmenter import ContextAugmenter
from src.axis_augmentation.fewshot_augmenter import FewShotAugmenter
from src.axis_augmentation.multidoc_augmenter import MultiDocAugmenter
from src.axis_augmentation.multiple_choice_augmenter import MultipleChoiceAugmenter
from src.axis_augmentation.paraphrase_instruct import Paraphrase
from src.axis_augmentation.text_surface_augmenter import TextSurfaceAugmenter


class MultiPromptify:
    """
    Main class for generating prompt variations based on templates.
    
    Supports two formats:
    1. Simple template: "{instruction:paraphrase} {question:surface}"
    2. Complex template: {'instruction': 'Question: {question}', 'template': '{instruction:paraphrase} {few_shot:(2)}'}
    """
    
    VARIATION_TYPE_TO_AUGMENTER = {
        "paraphrase": Paraphrase,
        "surface": TextSurfaceAugmenter,
        "non-semantic": TextSurfaceAugmenter,
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
        template: Union[str, Dict[str, str]],
        data: Union[pd.DataFrame, str, dict],
        variations_per_field: int = 3,
        api_key: str = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Generate prompt variations based on template and data.
        """
        # Load data
        data = self._load_data(data)
        
        # Parse template format
        instruction_template, processing_template = self._parse_template_format(template)
        
        if not instruction_template:
            raise ValueError("No instruction template found")
        
        # Parse processing template to check for paraphrase and few-shot
        fields = self.template_parser.parse(processing_template)
        variation_fields = self.template_parser.get_variation_fields()
        few_shot_fields = self.template_parser.get_few_shot_fields()
        
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
                
                # Generate few-shot examples (with answers)
                few_shot_content = ""
                if few_shot_fields:
                    few_shot_content = self._generate_few_shot_examples(
                        few_shot_fields[0], instruction_variant, data, row_idx
                    )
                
                # Create main question (without answer)
                main_question = self._create_main_question(instruction_variant, row)
                
                # Build final prompt
                prompt_parts = []
                if few_shot_content:
                    prompt_parts.append(few_shot_content)
                if main_question:
                    prompt_parts.append(main_question)
                
                final_prompt = '\n\n'.join(prompt_parts)
                
                all_variations.append({
                    'prompt': final_prompt,
                    'original_row_index': row_idx,
                    'variation_count': len(all_variations) + 1,
                    'field_values': {'instruction': instruction_variant, 'few_shot': few_shot_content},
                })
                
                if len(all_variations) >= self.max_variations:
                    break
        
        return all_variations
    
    def _load_data(self, data: Union[pd.DataFrame, str, dict]) -> pd.DataFrame:
        """Load data from various formats into DataFrame."""
        if isinstance(data, str):
            if data.endswith('.csv'):
                return pd.read_csv(data)
            elif data.endswith('.json'):
                with open(data, 'r') as f:
                    json_data = json.load(f)
                return pd.DataFrame(json_data)
        elif isinstance(data, dict):
            return pd.DataFrame(data)
        elif isinstance(data, pd.DataFrame):
            return data
        else:
            raise ValueError(f"Unsupported data format: {type(data)}")
    
    def _parse_template_format(self, template: Union[str, Dict[str, str]]) -> tuple:
        """
        Parse template format and return instruction template and processing template.
        
        Returns:
            Tuple of (instruction_template, processing_template)
        """
        if isinstance(template, dict):
            if 'instruction' in template and 'template' in template:
                # New format: {'instruction': 'Question: {question}', 'template': '{instruction:paraphrase}'}
                return template['instruction'], template['template']
            elif 'combined' in template:
                # Fallback combined format
                return None, template['combined']
        
        # Simple string template - no separate instruction
        return None, template
    
    def _validate_required_columns(self, instruction_template: str, data: pd.DataFrame):
        """Validate that all required columns exist in data."""
        if not instruction_template:
            return
            
        # Extract field names from instruction template
        required_fields = set(re.findall(r'\{([^}:]+)(?::[^}]+)?\}', instruction_template))
        required_fields.discard('instruction')  # Instruction is generated, not from data
        required_fields.discard('few_shot')     # Few-shot is generated, not from data
        
        missing_fields = required_fields - set(data.columns)
        if missing_fields:
            raise ValueError(f"Missing required columns in data: {missing_fields}")
    
    def _generate_instruction_variations(
        self, 
        instruction_template: str, 
        variation_fields: Dict[str, str], 
        variations_per_field: int, 
        api_key: str
    ) -> List[str]:
        """Generate variations of the instruction template (before filling with data)."""
        
        if 'instruction' not in variation_fields:
            return [instruction_template]
        
        variation_type = variation_fields['instruction']
        
        try:
            augmenter_class = self.VARIATION_TYPE_TO_AUGMENTER.get(variation_type, TextSurfaceAugmenter)
            
            if augmenter_class == Paraphrase and api_key:
                augmenter = augmenter_class(n_augments=variations_per_field, api_key=api_key)
            else:
                augmenter = augmenter_class(n_augments=variations_per_field)
            
            variations = augmenter.augment(instruction_template)
            
            if not variations or not isinstance(variations, list):
                return [instruction_template]
            
            # Ensure original is included
            if instruction_template not in variations:
                variations = [instruction_template] + variations[:variations_per_field]
            
            return variations[:variations_per_field + 1]
            
        except Exception as e:
            print(f"⚠️ Error generating instruction variations: {e}")
            return [instruction_template]
    
    def _generate_few_shot_examples(self, few_shot_field, instruction_variant: str, data: pd.DataFrame, current_row_idx: int) -> str:
        """Generate few-shot examples by filling instruction_variant with example data (including answers)."""
        
        num_examples = few_shot_field.few_shot_count
        
        # Get example data (excluding current row)
        available_data = data.drop(index=current_row_idx, errors='ignore')
        if len(available_data) == 0:
            return ""
        
        # Sample examples
        if few_shot_field.few_shot_format == 'tuple':
            sampled_data = available_data.sample(n=min(num_examples, len(available_data)), random_state=42)
        else:
            sampled_data = available_data.sample(n=min(num_examples, len(available_data)), random_state=current_row_idx)
        
        # Create examples by filling instruction_variant with ALL data (including answers)
        examples = []
        for _, example_row in sampled_data.iterrows():
            example_values = {}
            for col in example_row.index:
                if pd.notna(example_row[col]):
                    example_values[col] = str(example_row[col])
            
            example_prompt = self._fill_template_placeholders(instruction_variant, example_values)
            examples.append(example_prompt)
        
        return "\n\n".join(examples)
    
    def _create_main_question(self, instruction_variant: str, row: pd.Series) -> str:
        """Create main question by filling instruction_variant with row data (excluding answer)."""
        
        row_values = {}
        for col in row.index:
            if pd.notna(row[col]):
                # Skip answer fields for the main question
                if col.lower() not in ['answer', 'label', 'response', 'output']:
                    row_values[col] = str(row[col])
                else:
                    row_values[col] = ''

        
        return self._fill_template_placeholders(instruction_variant, row_values)
    
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
    
    def parse_template(self, template: Union[str, dict]) -> Dict[str, str]:
        """Parse template to extract fields and their variation types."""
        try:
            if isinstance(template, dict) and 'template' in template:
                # Use processing template part
                template_to_parse = template['template']
            elif isinstance(template, dict) and 'combined' in template:
                template_to_parse = template['combined']
            else:
                template_to_parse = template
                
            self.template_parser.parse(template_to_parse)
            return self.template_parser.get_variation_fields()
        except Exception as e:
            raise ValueError(f"Template parsing error: {str(e)}")
    
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