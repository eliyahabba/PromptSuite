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
from typing import Dict, List, Any, Optional

import pandas as pd

from multipromptify.core.exceptions import (
    InvalidTemplateError, MissingInstructionTemplateError,
    UnsupportedFileFormatError, UnsupportedExportFormatError
)
from multipromptify.core.models import (
    GoldFieldConfig, VariationConfig, VariationContext
)
from multipromptify.core.template_keys import (
    PROMPT_FORMAT, FEW_SHOT_KEY
)
from multipromptify.core.template_parser import TemplateParser
from multipromptify.generation import VariationGenerator, PromptBuilder, FewShotHandler
from multipromptify.shared.constants import GenerationDefaults


class MultiPromptify:
    """
    Main class for generating prompt variations based on dictionary templates.
    
    Template format:
    {
        "instruction": "Process the following input:\n",
        "instruction variations": ["paraphrase", "surface"],
        "prompt format": "{input}\nOutput: {output}",
        "prompt format variations": ["paraphrase", "surface"],
        "gold": "output",  # Name of the column containing the correct output/label
        "few_shot": {
            "count": 2,
            "format": "fixed",  # or "rotating"
            "split": "train"    # or "test" or "all"
        },
        "input": ["surface"]
    }
    """

    def __init__(self, max_variations_per_row: Optional[int] = GenerationDefaults.MAX_VARIATIONS_PER_ROW):
        """Initialize MultiPromptify with maximum variations limit."""
        self.max_variations_per_row = max_variations_per_row
        self.template_parser = TemplateParser()

        # Initialize the new refactored components
        self.variation_generator = VariationGenerator()
        self.prompt_builder = PromptBuilder()
        self.few_shot_handler = FewShotHandler()

    def generate_variations(
            self,
            template: dict,
            data: pd.DataFrame,
            variations_per_field: int = GenerationDefaults.VARIATIONS_PER_FIELD,
            api_key: str = None,
            seed: Optional[int] = None,
            **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Generate prompt variations based on dictionary template and data.
        
        Args:
            template: Dictionary template with field configurations
            data: DataFrame with the data
            variations_per_field: Number of variations per field
            api_key: API key for services that require it
            seed: Random seed for reproducibility
        
        Returns:
            List of generated variations
        """
        # Validate template
        is_valid, errors = self.template_parser.validate_template(template)
        if not is_valid:
            raise InvalidTemplateError(errors, template)

        # Load data if needed
        if isinstance(data, str):
            data = self._load_data(data)
        else:
            # Even for DataFrames passed directly, check for string lists
            data = self._convert_string_lists_to_lists(data)

        # Parse template
        fields = self.template_parser.parse(template)
        variation_fields = self.template_parser.get_variation_fields()
        few_shot_fields = self.template_parser.get_few_shot_fields()
        enumerate_fields = self.template_parser.get_enumerate_fields()

        # Create configuration objects
        gold_config = GoldFieldConfig.from_template(template.get('gold', None))
        variation_config = VariationConfig(
            variations_per_field=variations_per_field,
            api_key=api_key,
            max_variations_per_row=self.max_variations_per_row,
            seed=seed
        )
        instruction = self.template_parser.get_instruction()

        # Get prompt_format template from user - required
        prompt_format = self.template_parser.get_prompt_format()
        if not prompt_format:
            raise MissingInstructionTemplateError()

        # Validate gold field requirement
        self.few_shot_handler.validate_gold_field_requirement(prompt_format, gold_config.field, few_shot_fields)

        all_variations = []

        # For each data row
        for row_idx, row in data.iterrows():
            # Generate variations for all fields
            field_variations = self.variation_generator.generate_all_field_variations(
                prompt_format,
                instruction,
                variation_fields,
                row,
                variation_config,
                gold_config
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

            # Generate all possible row variations
            all_row_variations = self.few_shot_handler.create_row_variations(
                variation_context,
                few_shot_fields[0] if few_shot_fields else None,
                None,  # No limit - get all possible variations
                self.prompt_builder
            )

            # Sample max_variations_per_row from all possible variations for this row
            if self.max_variations_per_row is not None and len(all_row_variations) > self.max_variations_per_row:
                # Use consistent sampling based on seed
                sampled_variations = self._sample_variations_consistently(
                    all_row_variations, 
                    self.max_variations_per_row, 
                    seed
                )
                all_variations.extend(sampled_variations)
            else:
                # Use all variations if we have fewer than max_variations_per_row
                all_variations.extend(all_row_variations)

        return all_variations

    def _load_data(self, data_path: str) -> pd.DataFrame:
        """Load data from file path and automatically convert string representations of lists."""
        if data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
        elif data_path.endswith('.json'):
            with open(data_path, 'r') as f:
                json_data = json.load(f)
            df = pd.DataFrame(json_data)
        else:
            raise UnsupportedFileFormatError(data_path, ['.csv', '.json'])

        # Auto-convert string representations of lists to actual lists
        return self._convert_string_lists_to_lists(df)

    def _convert_string_lists_to_lists(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert string representations of lists back to actual Python lists.
        
        This handles cases where data was saved/loaded from CSV/JSON and 
        list columns became strings like "['item1', 'item2', 'item3']"
        """
        import ast

        def safe_eval(value):
            """Try to evaluate a string as a Python literal, return original if it fails."""
            if isinstance(value, str):
                try:
                    return ast.literal_eval(value)
                except (ValueError, SyntaxError):
                    return value
            return value

        df_copy = df.copy()

        # Apply safe_eval to all columns - it will only convert what it can
        for column in df_copy.columns:
            original_values = df_copy[column].copy()
            df_copy[column] = df_copy[column].apply(safe_eval)

            # Check if anything actually changed (meaning we converted some values)
            if not df_copy[column].equals(original_values):
                print(f"✅ Converted some values in column '{column}' from strings to Python objects")

        return df_copy

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
        field_count = len([k for k in template_config.keys() if k not in [FEW_SHOT_KEY, PROMPT_FORMAT]])
        has_few_shot = FEW_SHOT_KEY in template_config
        has_custom_prompt_format = PROMPT_FORMAT in template_config

        return {
            'total_variations': len(variations),
            'original_rows': len(row_counts),
            'avg_variations_per_row': sum(row_counts.values()) / len(row_counts) if row_counts else 0,
            'template_fields': field_count,
            'has_few_shot': has_few_shot,
            'has_custom_prompt_format': has_custom_prompt_format,
            'min_variations_per_row': min(row_counts.values()) if row_counts else 0,
            'max_variations_per_row_per_row': max(row_counts.values()) if row_counts else 0,
        }

    def parse_template(self, template: dict) -> Dict[str, List[str]]:
        """Parse template to extract fields and their variation types."""
        self.template_parser.parse(template)
        return self.template_parser.get_variation_fields()

    @staticmethod
    def _prepare_variations_for_conversation_export(variations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Enhance variations with conversation field to match API format.
        This is a shared utility function to ensure consistent conversation JSON output.
        
        Args:
            variations: List of generated variations
            
        Returns:
            List of variations with conversation field added and extra fields removed
        """
        enhanced_variations = []

        for variation in variations:
            # Create a new variation with only the required API fields
            enhanced_var = {
                'prompt': variation.get('prompt', ''),
                'original_row_index': variation.get('original_row_index', 0),
                'variation_count': variation.get('variation_count', 1),
                'template_config': variation.get('template_config', {}),
                'field_values': variation.get('field_values', {}),
                'gold_updates': variation.get('gold_updates')
            }

            # Add conversation field if not already present
            if 'conversation' in variation and variation['conversation']:
                enhanced_var['conversation'] = variation['conversation']
            else:
                # Build conversation from prompt
                prompt = variation.get('prompt', '')

                # Split prompt into conversation parts if it contains few-shot examples
                parts = prompt.split('\n\n')
                conversation = []

                for i, part in enumerate(parts):
                    part = part.strip()
                    if not part:
                        continue

                    # Check if this is the last part (incomplete question)
                    if i == len(parts) - 1:
                        # Last part - this is the question without answer
                        conversation.append({
                            "role": "user",
                            "content": part
                        })
                    else:
                        # This is a complete Q&A pair
                        # Split by the last occurrence of newline to separate question and answer
                        lines = part.split('\n')
                        if len(lines) >= 2:
                            # Assume the last line is the answer
                            answer = lines[-1].strip()
                            question = '\n'.join(lines[:-1]).strip()

                            conversation.append({
                                "role": "user",
                                "content": question
                            })
                            conversation.append({
                                "role": "assistant",
                                "content": answer
                            })
                        else:
                            # Single line - treat as user message
                            conversation.append({
                                "role": "user",
                                "content": part
                            })

                enhanced_var['conversation'] = conversation

            enhanced_variations.append(enhanced_var)

        return enhanced_variations

    def save_variations(self, variations: List[Dict[str, Any]], output_path: str, format: str = "json"):
        """Save variations to file."""
        if format == "json":
            # Prepare variations to conversation format before dumping to JSON
            conversation_variations = MultiPromptify._prepare_variations_for_conversation_export(variations)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(conversation_variations, f, indent=2, ensure_ascii=False)

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
            raise UnsupportedExportFormatError(format, ["json", "csv", "txt"])

    def _sample_variations_consistently(
            self, 
            variations: List[Dict[str, Any]], 
            max_variations_per_row: int, 
            seed: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Sample max_variations_per_row from all possible variations consistently.
        
        This ensures that if we have 10 variations and want 5, we'll always
        select the same 5 indices across all rows (e.g., indices 0,2,4,6,8).
        
        Args:
            variations: All possible variations for a row
            max_variations_per_row: Number of variations to sample
            seed: Random seed for consistent sampling
            
        Returns:
            List of sampled variations
        """
        if len(variations) <= max_variations_per_row:
            return variations
        
        # Set seed for consistent sampling across rows
        if seed is not None:
            import random
            random.seed(seed)
        
        # Generate consistent indices based on the total number of variations
        # This ensures the same pattern is used across all rows
        total_variations = len(variations)
        
        # Create a deterministic pattern based on the total count
        # We'll use a simple pattern: take every nth variation
        if max_variations_per_row >= total_variations:
            return variations
        
        # Calculate step size to get max_variations_per_row samples
        step = total_variations / max_variations_per_row
        
        # Generate indices
        indices = []
        for i in range(max_variations_per_row):
            index = int(i * step)
            if index >= total_variations:
                index = total_variations - 1
            indices.append(index)
        
        # Remove duplicates while preserving order
        unique_indices = []
        for idx in indices:
            if idx not in unique_indices:
                unique_indices.append(idx)
        
        # If we don't have enough unique indices, add more from the end
        while len(unique_indices) < max_variations_per_row and len(unique_indices) < total_variations:
            for i in range(total_variations):
                if i not in unique_indices:
                    unique_indices.append(i)
                    break
        
        # Return sampled variations
        return [variations[i] for i in unique_indices[:max_variations_per_row]]
