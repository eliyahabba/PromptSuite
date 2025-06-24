"""
Few Shot Handler: Centralized handling of few-shot examples and row variation creation.
"""

import itertools
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

import pandas as pd

from multipromptify.augmentations.structure.enumerate import EnumeratorAugmenter
from multipromptify.augmentations.structure.fewshot import FewShotAugmenter
from multipromptify.core.exceptions import (
    FewShotGoldFieldMissingError, FewShotDataInsufficientError, FewShotConfigurationError
)
from multipromptify.core.models import VariationContext, FieldVariation, FewShotContext
from multipromptify.core.template_keys import (
    PROMPT_FORMAT_VARIATIONS, INSTRUCTION, INSTRUCTION_VARIATIONS
)
from multipromptify.utils.formatting import format_field_value


@dataclass
class FewShotConfig:
    """Configuration for few-shot examples.
    
    Attributes:
        count: Number of few-shot examples to use
        format: Sampling strategy - 'shared_first_n', 'shared_random_n', or 'random_per_row'
        split: Data split to use for few-shot examples - 'train', 'test', or 'all'
    """
    count: int = 2
    format: str = "shared_first_n"  # 'shared_first_n', 'shared_random_n', or 'random_per_row'
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
            prompt_format_template: str,
            gold_field: str,
            few_shot_fields: list
    ) -> None:
        """Validate that gold field is provided when needed for few-shot examples."""
        # Simple check: if we have few-shot fields but no gold field, that's an error
        if few_shot_fields and not gold_field:
            raise FewShotGoldFieldMissingError()

    def validate_data_sufficiency(
            self,
            data: pd.DataFrame,
            few_shot_config: FewShotConfig,
            current_row_idx: int
    ) -> None:
        """Check if we have enough data for few-shot examples."""
        if data is None or len(data) <= few_shot_config.count:
            raise FewShotDataInsufficientError(few_shot_config.count, len(data) if data is not None else 0)

    def parse_few_shot_config(self, config: dict) -> FewShotConfig:
        """
        Parse and validate few-shot configuration.
        Centralized from template_parser.py logic.
        
        Few-shot formats:
        - "shared_first_n": Always use the first N examples from the available data (deterministic, shared for all rows)
        - "shared_random_n": Always use the same N random examples (with fixed seed, shared for all rows)
        - "random_per_row": Randomly sample different examples for each row (using row index as seed)
        """
        if not isinstance(config, dict):
            raise FewShotConfigurationError("config_type", type(config).__name__, ["dictionary"])

        few_shot_config = FewShotConfig(
            count=config.get("count", 2),
            format=config.get("format", "shared_first_n"),
            split=config.get("split", "all")
        )

        # Validate configuration
        if few_shot_config.count <= 0:
            raise FewShotConfigurationError("count", few_shot_config.count)

        if few_shot_config.format not in ['shared_first_n', 'shared_random_n', 'random_per_row']:
            raise FewShotConfigurationError("format", few_shot_config.format, ['shared_first_n', 'shared_random_n', 'random_per_row'])

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
            max_variations_per_row: Optional[int],
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
            # Check limit only if max_variations_per_row is not None
            if max_variations_per_row is not None and len(variations) >= max_variations_per_row:
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
        prompt_format_variant = field_values.get(
            PROMPT_FORMAT_VARIATIONS,
            variation_context.field_variations.get(PROMPT_FORMAT_VARIATIONS,
                                                   [FieldVariation(data='', gold_update=None)])[0]
        ).data
        # Extract row values and gold updates
        row_values, gold_updates = self._extract_row_values_and_updates(
            variation_context, field_values
        )
        # Generate few-shot examples
        few_shot_examples = self._generate_few_shot_examples(
            few_shot_field, prompt_format_variant, variation_context, field_values
        )
        # Create main input
        main_input = self._create_main_input(
            prompt_format_variant, row_values, variation_context.gold_config, prompt_builder
        )
        # Determine which system prompt to use for this variation
        default_instruction = variation_context.template.get(INSTRUCTION)
        instruction_variant = field_values.get(INSTRUCTION_VARIATIONS, None)
        if instruction_variant:
            instruction = instruction_variant.data or default_instruction
        else:
            instruction = default_instruction

        # Fill placeholders in the instruction (system prompt)
        instruction_filled = prompt_builder.fill_template_placeholders(
            instruction,
            row_values
        )

        # Format conversation and prompt using the selected system prompt
        conversation_messages = self._format_conversation(
            few_shot_examples,
            main_input,
            instruction_filled
        )
        final_prompt = self._format_final_prompt(
            few_shot_examples,
            main_input,
            instruction_filled
        )
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
            'field_values': output_field_values,  # Formatted values for display in prompts
            'gold_updates': gold_updates,
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
                # Field variations have already been applied
                processed_value = format_field_value(field_data.data)
                # Apply direct enumerate configuration even if field has other variations
                if 'enumerate' in variation_context.template:
                    processed_value = self._apply_enumerate_if_needed(processed_value, col, enumerate_fields_config)
                row_values[col] = processed_value
                if field_data.gold_update:
                    gold_updates.update(field_data.gold_update)
            elif variation_context.gold_config.field and col == variation_context.gold_config.field:
                # Skip gold field from main prompt - it should only appear in few-shot examples
                continue
            else:
                processed_value = format_field_value(variation_context.row_data[col])
                # Apply enumerate if configured
                processed_value = self._apply_enumerate_if_needed(processed_value, col, enumerate_fields_config)
                row_values[col] = processed_value

        # Always set gold_updates to the original value if not already set
        gold_field = variation_context.gold_config.field
        if gold_field:
            if not gold_updates or gold_field not in gold_updates:
                try:
                    from multipromptify.utils.formatting import extract_gold_value
                    gold_value = extract_gold_value(variation_context.row_data, gold_field)
                    gold_updates[gold_field] = format_field_value(gold_value)
                except Exception as e:
                    print(f"⚠️ Warning: Could not extract gold field '{gold_field}': {e}")
                    # Fallback: try direct column access
                    if gold_field in variation_context.row_data:
                        gold_updates[gold_field] = format_field_value(variation_context.row_data[gold_field])

        return row_values, gold_updates

    def _get_enumerate_fields_config(self, template: dict) -> Dict[str, dict]:
        """Extract enumerate field configurations from template."""
        from multipromptify.core.template_keys import ENUMERATE_VARIATION
        enumerate_config = {}
        
        # Check for direct enumerate configuration (both old and new format)
        if 'enumerate' in template:
            enum_field = template['enumerate'].get('field')
            if enum_field:
                enumerate_config[enum_field] = template['enumerate']
        
        # Check for ENUMERATE_VARIATION as a direct key (new format)
        if ENUMERATE_VARIATION in template:
            enum_field = template[ENUMERATE_VARIATION].get('field')
            if enum_field:
                enumerate_config[enum_field] = template[ENUMERATE_VARIATION]

        # Check for field variations that include enumeration (for few-shot examples only)
        for field_name, variations in template.items():
            if isinstance(variations, list) and ENUMERATE_VARIATION in variations:
                # Field has enumeration as a variation - use the first enumeration type for consistency
                # This matches the deterministic order used in EnumeratorAugmenter
                enum_types = ['1234', 'ABCD', 'abcd', 'roman']
                enumerate_config[field_name] = {'type': enum_types[0]}
        
        return enumerate_config

    def _get_enumerate_fields_config_for_variation(
            self, 
            template: dict, 
            field_values: Dict[str, FieldVariation] = None
    ) -> Dict[str, dict]:
        """Extract enumerate field configurations for a specific variation."""
        from multipromptify.core.template_keys import ENUMERATE_VARIATION
        enumerate_config = {}
        
        # Check for direct enumerate configuration (both old and new format)
        if 'enumerate' in template:
            enum_field = template['enumerate'].get('field')
            if enum_field:
                enumerate_config[enum_field] = template['enumerate']
        
        # Check for ENUMERATE_VARIATION as a direct key (new format)
        if ENUMERATE_VARIATION in template:
            enum_field = template[ENUMERATE_VARIATION].get('field')
            if enum_field:
                enumerate_config[enum_field] = template[ENUMERATE_VARIATION]

        # Check for field variations that include enumeration
        for field_name, variations in template.items():
            if isinstance(variations, list) and ENUMERATE_VARIATION in variations:
                # If we have field_values, try to determine the enumeration type from the actual field value
                if field_values and field_name in field_values:
                    field_data = field_values[field_name].data
                    detected_enum_type = self._detect_enumeration_type(field_data)
                    if detected_enum_type:
                        enumerate_config[field_name] = {'type': detected_enum_type}
                    else:
                        # Fallback to first type
                        enum_types = ['1234', 'ABCD', 'abcd', 'roman']
                        enumerate_config[field_name] = {'type': enum_types[0]}
                else:
                    # Fallback to first type
                    enum_types = ['1234', 'ABCD', 'abcd', 'roman']
                    enumerate_config[field_name] = {'type': enum_types[0]}
        
        return enumerate_config

    def _detect_enumeration_type(self, field_data: str) -> str:
        """Detect the enumeration type from the field data."""
        if not field_data:
            return None
        
        # Look for patterns in the enumerated data
        if '1.' in field_data or '2.' in field_data:
            return '1234'
        elif 'A.' in field_data or 'B.' in field_data:
            return 'ABCD'
        elif 'a.' in field_data or 'b.' in field_data:
            return 'abcd'
        elif 'I.' in field_data or 'II.' in field_data:
            return 'roman'
        
        return None

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
            prompt_format_variant: str,
            variation_context: VariationContext,
            field_values: Dict[str, FieldVariation] = None
    ) -> List[Dict[str, str]]:
        """Generate few-shot examples if configured, with system prompt support."""
        if not few_shot_field or variation_context.data is None:
            return []

        few_shot_context = FewShotContext(
            prompt_format_template=prompt_format_variant,
            few_shot_field=few_shot_field,
            data=variation_context.data,
            current_row_idx=variation_context.row_index,
            gold_config=variation_context.gold_config
        )
        identification_data = few_shot_context.to_identification_data()
        # Add enumeration configuration - use current variation's enumeration type if available
        identification_data['enumerate_configs'] = self._get_enumerate_fields_config_for_variation(
            variation_context.template, field_values
        )
        examples = self.few_shot_augmenter.augment(
            prompt_format_variant,
            identification_data
        )
        # Inject system prompt only in the first example if present
        instruction = variation_context.template.get(INSTRUCTION)
        if instruction and examples:
            examples[0][INSTRUCTION] = instruction
        return examples

    def _create_main_input(
            self,
            prompt_format_variant: str,
            row_values: Dict[str, str],
            gold_config,
            prompt_builder
    ) -> str:
        """Create the main input by filling template with row values."""
        main_input = prompt_builder.fill_template_placeholders(prompt_format_variant, row_values)

        # Remove gold field placeholder (it's always excluded from row_values)
        if gold_config.field:
            main_input = main_input.replace(f'{{{gold_config.field}}}', '')

        return main_input.strip()

    def _format_conversation(
            self,
            few_shot_examples: List[Dict[str, str]],
            main_input: str,
            prompt_format: str = None
    ) -> List[Dict[str, str]]:
        """Format few-shot examples and main input as conversation messages, with system prompt support."""
        conversation_messages = []
        # Always add system prompt if present
        if prompt_format:
            conversation_messages.append({
                "role": "system",
                "content": prompt_format
            })
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
            main_input: str,
            prompt_format: str = None
    ) -> str:
        """Format few-shot examples and main input as a single prompt string, with system prompt support."""
        prompt_parts = []
        # Always add system prompt if present
        if prompt_format:
            prompt_parts.append(prompt_format)
        if few_shot_examples:
            few_shot_content = self.few_shot_augmenter.format_few_shot_as_string(few_shot_examples)
            prompt_parts.append(few_shot_content)
        if main_input:
            prompt_parts.append(main_input)
        return '\n\n'.join(prompt_parts)
