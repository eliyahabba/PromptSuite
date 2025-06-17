"""
MultiPromptify: A library for generating multi-prompt datasets from single-prompt datasets.

IMPORTANT: MultiPromptify assumes clean input data:
- All DataFrame cells contain simple values (strings, numbers)
- No NaN values (use empty strings instead)
- No nested arrays or complex objects in cells
- All columns exist as specified in the template

If your data doesn't meet these requirements, clean it before passing to MultiPromptify.
"""

import json
from typing import Dict, List, Any

import pandas as pd
from multipromptify.template_parser import TemplateParser
from multipromptify.models import (
    GoldFieldConfig, VariationConfig, VariationContext
)
from multipromptify.generation import VariationGenerator, PromptBuilder, FewShotHandler


class MultiPromptify:
    """
    Main class for generating prompt variations based on dictionary templates.
    
    Template format:
    {
        "instruction_template": "Process the following input: {input}\nOutput: {output}",
        "instruction": ["paraphrase", "surface"],
        "gold": "output",  # Name of the column containing the correct output/label
        "few_shot": {
            "count": 2,
            "format": "fixed",  # or "rotating"
            "split": "train"    # or "test" or "all"
        },
        "input": ["surface"]
    }
    """

    def __init__(self, max_variations: int = 100):
        """Initialize MultiPromptify with maximum variations limit."""
        self.max_variations = max_variations
        self.template_parser = TemplateParser()
        
        # Initialize the new refactored components
        self.variation_generator = VariationGenerator()
        self.prompt_builder = PromptBuilder()
        self.few_shot_handler = FewShotHandler()

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

        # Create configuration objects
        gold_config = GoldFieldConfig.from_template(template.get('gold', None))
        variation_config = VariationConfig(
            variations_per_field=variations_per_field,
            api_key=api_key,
            max_variations=self.max_variations
        )

        # Get instruction template from user - required
        instruction_template = self.template_parser.get_instruction_template()
        if not instruction_template:
            raise ValueError(
                "Instruction template is required. Please specify 'instruction_template' in your template. "
                "Example: \"instruction_template\": \"Process the input: {input}\\nOutput: {output}\""
            )

        # Validate gold field requirement
        self.few_shot_handler.validate_gold_field_requirement(instruction_template, gold_config.field, few_shot_fields)

        all_variations = []

        # For each data row
        for row_idx, row in data.iterrows():
            if len(all_variations) >= self.max_variations:
                break

            # Generate variations for all fields
            field_variations = self.variation_generator.generate_all_field_variations(
                instruction_template, variation_fields, row, variation_config, gold_config
            )

            # Create variation context
            variation_context = VariationContext(
                row_data=row,
                row_index=row_idx,
                template=template,
                field_variations=field_variations,
                gold_config=gold_config,
                variation_config=variation_config,
                data=data
            )

            # Generate row variations
            row_variations = self.few_shot_handler.create_row_variations(
                variation_context, 
                few_shot_fields[0] if few_shot_fields else None,
                self.max_variations,
                self.prompt_builder
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
