"""
Few Shot Handler: Centralized handling of few-shot examples and row variation creation.
"""

import itertools
from typing import Dict, List, Any, Optional
import pandas as pd
from dataclasses import dataclass

from multipromptify.augmentations.structure.fewshot import FewShotAugmenter
from multipromptify.models import VariationContext, FieldVariation, FewShotContext
from multipromptify.utils.formatting import format_field_value


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
            raise ValueError(
                "Gold field is required when using few-shot examples. "
                "Please specify the 'gold' field in your template to indicate which column contains the correct outputs/labels. "
                "Example: \"gold\": \"output\" or \"gold\": \"label\""
            )

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
            raise ValueError("No data provided for few-shot examples")
        
        # Get available data based on split configuration
        available_data = self._filter_data_by_split(data, few_shot_config.split)
        
        # Remove current row to avoid data leakage
        available_data = available_data.drop(current_row_idx, errors='ignore')
        
        if len(available_data) < few_shot_config.count:
            raise ValueError(
                f"Not enough data for few-shot examples. "
                f"Requested {few_shot_config.count} examples but only {len(available_data)} available after filtering. "
                f"Consider reducing the few-shot count or providing more data."
            )

    def parse_few_shot_config(self, config: dict) -> FewShotConfig:
        """
        Parse and validate few-shot configuration.
        Centralized from template_parser.py logic.
        """
        if not isinstance(config, dict):
            raise ValueError("few_shot configuration must be a dictionary")
        
        few_shot_config = FewShotConfig(
            count=config.get("count", 2),
            format=config.get("format", "rotating"),
            split=config.get("split", "all")
        )
        
        # Validate configuration
        if few_shot_config.count <= 0:
            raise ValueError(f"Few-shot count must be positive, got {few_shot_config.count}")
        
        if few_shot_config.format not in ['fixed', 'rotating']:
            raise ValueError(f"Few-shot format must be 'fixed' or 'rotating', got {few_shot_config.format}")
        
        if few_shot_config.split not in ['all', 'train', 'test']:
            raise ValueError(f"Few-shot split must be 'all', 'train', or 'test', got {few_shot_config.split}")
        
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
        
        return {
            'prompt': final_prompt,
            'conversation': conversation_messages,
            'original_row_index': variation_context.row_index,
            'variation_count': variation_count,
            'template_config': variation_context.template,
            'field_values': output_field_values,
            'gold_updates': gold_updates if gold_updates else None,
        }

    def _extract_row_values_and_updates(
        self, 
        variation_context: VariationContext, 
        field_values: Dict[str, FieldVariation]
    ) -> tuple[Dict[str, str], Dict[str, Any]]:
        """Extract row values and gold updates from field variations."""
        row_values = {}
        gold_updates = {}
        
        for col in variation_context.row_data.index:
            # Assume clean data - skip empty columns but process all others
            if col in field_values:
                field_data = field_values[col]
                # Ensure even field variations go through formatting
                row_values[col] = format_field_value(field_data.data)
                if field_data.gold_update:
                    gold_updates.update(field_data.gold_update)
            elif variation_context.gold_config.field and col == variation_context.gold_config.field:
                continue  # Skip gold field
            else:
                row_values[col] = format_field_value(variation_context.row_data[col])
        
        return row_values, gold_updates

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