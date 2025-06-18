"""
Few Shot Handler: Centralized handling of few-shot examples and row variation creation.
"""

import itertools
from typing import Dict, List, Any, Optional
import pandas as pd
from dataclasses import dataclass

from multipromptify.augmentations.structure.fewshot import FewShotAugmenter
from multipromptify.augmentations.structure.enumerate import EnumeratorAugmenter
from multipromptify.models import VariationContext, FieldVariation, FewShotContext
from multipromptify.utils.formatting import format_field_value
from multipromptify.exceptions import (
    FewShotGoldFieldMissingError, FewShotDataInsufficientError, FewShotConfigurationError
)


@dataclass
class FewShotConfig:
    """Configuration for few-shot examples."""
    count: int = 2
    format: str = "rotating"  # 'fixed' or 'rotating'
    split: str = "all"  # 'train', 'test', or 'all'


class FewShotHandler:
    """
    Centralized handler for few-shot examples and creation of row variations.
    Consolidates all few-shot logic from engine.py, fewshot.py, and template_parser.py.
    """

    def __init__(self):
        self.few_shot_augmenter = FewShotAugmenter()
        self.enumerator_augmenter = EnumeratorAugmenter()

    def validate_gold_field_requirement(
        self, 
        instruction_template: str, 
        gold_field: str, 
        few_shot_fields: list
    ) -> None:
        """
        Validate that gold field is provided when needed for few-shot examples.
        Centralized validation logic from engine.py and fewshot.py.
        """
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
            raise FewShotGoldFieldMissingError()

    def validate_data_sufficiency(
        self,
        data: pd.DataFrame,
        few_shot_config: FewShotConfig,
        current_row_idx: int
    ) -> None:
        """
        Check if we have enough data for few-shot examples.
        Centralized from fewshot.py data sufficiency checking.
        """
        if data is None or len(data) == 0:
            raise FewShotDataInsufficientError(few_shot_config.count, 0)
        
        # Get available data based on split configuration
        available_data = self._filter_data_by_split(data, few_shot_config.split)
        
        # Remove current row to avoid data leakage
        available_data = available_data.drop(current_row_idx, errors='ignore')
        
        if len(available_data) < few_shot_config.count:
            raise FewShotDataInsufficientError(few_shot_config.count, len(available_data), few_shot_config.split)

    def parse_few_shot_config(self, config: dict) -> FewShotConfig:
        """
        Parse and validate few-shot configuration.
        Centralized from template_parser.py logic.
        """
        if not isinstance(config, dict):
            raise FewShotConfigurationError("config_type", type(config).__name__, ["dictionary"])
        
        few_shot_config = FewShotConfig(
            count=config.get("count", 2),
            format=config.get("format", "rotating"),
            split=config.get("split", "all")
        )
        
        # Validate configuration
        if few_shot_config.count <= 0:
            raise FewShotConfigurationError("count", few_shot_config.count)
        
        if few_shot_config.format not in ['fixed', 'rotating']:
            raise FewShotConfigurationError("format", few_shot_config.format, ['fixed', 'rotating'])
        
        if few_shot_config.split not in ['all', 'train', 'test']:
            raise FewShotConfigurationError("split", few_shot_config.split, ['all', 'train', 'test'])
        
        return few_shot_config

    def _filter_data_by_split(self, data: pd.DataFrame, split: str) -> pd.DataFrame:
        """Filter data based on split configuration."""
        if split == "train":
            return data[data.get('split', 'train') == 'train']
        elif split == "test":
            return data[data.get('split', 'train') == 'test']
        else:  # 'all'
            return data



    def create_row_variations(
            self,
            variation_context: VariationContext,
            few_shot_field,
            max_variations: int,
            prompt_builder
    ) -> List[Dict[str, Any]]:
        """Create variations for a single row combining all field variations."""
        variations = []
        varying_fields = list(variation_context.field_variations.keys())
        
        if not varying_fields:
            return variations
        
        # Create all possible combinations of field variations
        variation_combinations = self._create_variation_combinations(variation_context.field_variations)
        
        for combination in variation_combinations:
            if len(variations) >= max_variations:
                break
            
            # Build a single variation
            variation = self._build_single_variation(
                combination, varying_fields, variation_context, 
                few_shot_field, prompt_builder, len(variations) + 1
            )
            
            if variation:
                variations.append(variation)
                
        return variations

    def _create_variation_combinations(
        self, 
        field_variations: Dict[str, List[FieldVariation]]
    ) -> List[tuple]:
        """Create all possible combinations of field variations."""
        return list(itertools.product(*[field_variations[field] for field in field_variations.keys()]))

    def _build_single_variation(
        self,
        combination: tuple,
        varying_fields: List[str],
        variation_context: VariationContext,
        few_shot_field,
        prompt_builder,
        variation_count: int
    ) -> Optional[Dict[str, Any]]:
        """Build a single variation from a combination of field values."""
        
        field_values = dict(zip(varying_fields, combination))
        instruction_variant = field_values.get(
            'instruction', 
            variation_context.field_variations.get('instruction', [FieldVariation(data='', gold_update=None)])[0]
        ).data
        
        # Extract row values and gold updates
        row_values, gold_updates = self._extract_row_values_and_updates(
            variation_context, field_values
        )
        
        # Generate few-shot examples
        few_shot_examples = self._generate_few_shot_examples(
            few_shot_field, instruction_variant, variation_context
        )
        
        # Create main input
        main_input = self._create_main_input(
            instruction_variant, row_values, variation_context.gold_config, prompt_builder
        )
        
        # Format conversation and prompt
        conversation_messages = self._format_conversation(few_shot_examples, main_input)
        final_prompt = self._format_final_prompt(few_shot_examples, main_input)
        
        # Prepare output field values
        output_field_values = {
            field_name: field_data.data 
            for field_name, field_data in field_values.items()
        }
        
        # Also preserve original field values with their original structure
        original_field_values = {}
        for field_name, field_data in field_values.items():
            # Try to get the original value from variation context
            if hasattr(field_data, 'original_value'):
                original_field_values[field_name] = field_data.original_value
            elif field_name in variation_context.row_data.index:
                original_field_values[field_name] = variation_context.row_data[field_name]
            else:
                original_field_values[field_name] = field_data.data
        
        return {
            'prompt': final_prompt,
            'conversation': conversation_messages,
            'original_row_index': variation_context.row_index,
            'variation_count': variation_count,
            'template_config': variation_context.template,
            'field_values': output_field_values,  # Formatted values for display in prompts
            'original_field_values': original_field_values,  # Original structure for metadata
            'gold_updates': gold_updates if gold_updates else None,
            'original_answer': self._get_original_answer(variation_context),  # Always include original answer
        }

    def _extract_row_values_and_updates(
        self, 
        variation_context: VariationContext, 
        field_values: Dict[str, FieldVariation]
    ) -> tuple[Dict[str, str], Dict[str, Any]]:
        """Extract row values and gold updates from field variations."""
        row_values = {}
        gold_updates = {}
        
        # First, get enumerate fields from template
        enumerate_fields_config = self._get_enumerate_fields_config(variation_context.template)
        
        for col in variation_context.row_data.index:
            # Assume clean data - skip empty columns but process all others
            if col in field_values:
                field_data = field_values[col]
                # Ensure even field variations go through formatting
                processed_value = format_field_value(field_data.data)
                
                # Apply enumerate if this field should be enumerated
                processed_value = self._apply_enumerate_if_needed(processed_value, col, enumerate_fields_config)
                
                row_values[col] = processed_value
                if field_data.gold_update:
                    gold_updates.update(field_data.gold_update)
            elif variation_context.gold_config.field and col == variation_context.gold_config.field:
                continue  # Skip gold field
            else:
                processed_value = format_field_value(variation_context.row_data[col])
                
                # Apply enumerate if this field should be enumerated
                processed_value = self._apply_enumerate_if_needed(processed_value, col, enumerate_fields_config)
                
                row_values[col] = processed_value
        
        return row_values, gold_updates

    def _get_enumerate_fields_config(self, template: dict) -> Dict[str, dict]:
        """Extract enumerate field configurations from template."""
        enumerate_config = {}
        if 'enumerate' in template:
            enum_field = template['enumerate'].get('field')
            if enum_field:
                enumerate_config[enum_field] = template['enumerate']
        return enumerate_config

    def _apply_enumerate_if_needed(self, value: str, field_name: str, enumerate_configs: Dict[str, dict]) -> str:
        """Apply enumeration to field value if configured."""
        if field_name in enumerate_configs:
            enum_config = enumerate_configs[field_name]
            enum_type = enum_config.get('type', '1234')
            
            try:
                return self.enumerator_augmenter.enumerate_field(value, enum_type)
            except Exception as e:
                print(f"⚠️ Error enumerating field '{field_name}': {e}")
                return value  # Return original value if enumeration fails
        
        return value

    def _generate_few_shot_examples(
        self, 
        few_shot_field, 
        instruction_variant: str, 
        variation_context: VariationContext
    ) -> List[Dict[str, str]]:
        """Generate few-shot examples if configured."""
        if not few_shot_field or variation_context.data is None:
            return []
        
        few_shot_context = FewShotContext(
            instruction_template=instruction_variant,
            few_shot_field=few_shot_field,
            data=variation_context.data,
            current_row_idx=variation_context.row_index,
            gold_config=variation_context.gold_config
        )
        
        return self.few_shot_augmenter.augment(
            instruction_variant, 
            few_shot_context.to_identification_data()
        )

    def _create_main_input(
        self, 
        instruction_variant: str, 
        row_values: Dict[str, str], 
        gold_config, 
        prompt_builder
    ) -> str:
        """Create the main input by filling template with row values."""
        main_input = prompt_builder.fill_template_placeholders(instruction_variant, row_values)
        
        # Remove gold field placeholder if present
        if gold_config.field:
            main_input = main_input.replace(f'{{{gold_config.field}}}', '')
        
        return main_input.strip()

    def _format_conversation(
        self, 
        few_shot_examples: List[Dict[str, str]], 
        main_input: str
    ) -> List[Dict[str, str]]:
        """Format few-shot examples and main input as conversation messages."""
        conversation_messages = []
        
        # Add few-shot examples as conversation pairs
        for example in few_shot_examples:
            conversation_messages.append({
                "role": "user",
                "content": example["input"]
            })
            conversation_messages.append({
                "role": "assistant",
                "content": example["output"]
            })
        
        # Add main input as final user message
        if main_input:
            conversation_messages.append({
                "role": "user",
                "content": main_input
            })
        
        return conversation_messages

    def _format_final_prompt(
        self, 
        few_shot_examples: List[Dict[str, str]], 
        main_input: str
    ) -> str:
        """Format few-shot examples and main input as a single prompt string."""
        prompt_parts = []
        
        if few_shot_examples:
            few_shot_content = self.few_shot_augmenter.format_few_shot_as_string(few_shot_examples)
            prompt_parts.append(few_shot_content)
        
        if main_input:
            prompt_parts.append(main_input)
        
        return '\n\n'.join(prompt_parts) 

    def _get_original_answer(self, variation_context: VariationContext) -> str:
        """Get the original answer from the variation context."""
        if variation_context.gold_config and variation_context.gold_config.field:
            return variation_context.row_data[variation_context.gold_config.field]
        else:
            return "" 