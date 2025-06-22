"""
Variation Generator: Handles generation of field variations and instruction variations.
"""

from typing import Dict, List, Any
import pandas as pd

from multipromptify.augmentations.factory import AugmenterFactory
from multipromptify.core.models import (
    VariationConfig, FieldVariation, FieldAugmentationData
)
from multipromptify.utils.formatting import format_field_value, extract_gold_value
from multipromptify.core.template_keys import (
    INSTRUCTION_TEMPLATE_KEY, INSTRUCTION_KEY, QUESTION_KEY, GOLD_KEY, FEW_SHOT_KEY, OPTIONS_KEY, CONTEXT_KEY, PROBLEM_KEY,
    PARAPHRASE_WITH_LLM, REWORDING, CONTEXT_VARIATION, SHUFFLE_VARIATION, MULTIDOC_VARIATION, ENUMERATE_VARIATION,
    GOLD_FIELD, INSTRUCTION_TEMPLATE_KEY, SYSTEM_PROMPT_KEY
)


class VariationGenerator:
    """
    Handles the generation of variations for fields and instructions.
    """

    def generate_instruction_variations(
            self,
            instruction_template: str,
            variation_fields: Dict[str, List[str]],
            variation_config: VariationConfig
    ) -> List[str]:
        """Generate variations of the instruction template."""

        if INSTRUCTION_KEY not in variation_fields or not variation_fields[INSTRUCTION_KEY]:
            return [instruction_template]

        variation_types = variation_fields[INSTRUCTION_KEY]
        all_variations = []

        # Generate variations for each type
        for variation_type in variation_types:
            try:
                # Use Factory to create augmenter with proper configuration
                augmenter = AugmenterFactory.create(
                    variation_type=variation_type,
                    n_augments=variation_config.variations_per_field,
                    api_key=variation_config.api_key
                )

                # Use Factory to handle augmentation with special cases
                variations = AugmenterFactory.augment_with_special_handling(
                    augmenter=augmenter,
                    text=instruction_template,
                    variation_type=variation_type
                )

                # Extract text from results using Factory method
                string_variations = AugmenterFactory.extract_text_from_result(variations, variation_type)
                all_variations.extend(string_variations[:variation_config.variations_per_field])

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

        return unique_variations[:variation_config.variations_per_field + 1]

    def generate_system_prompt_variations(
            self,
            system_prompt_template: str,
            variation_fields: Dict[str, List[str]],
            variation_config: VariationConfig
    ) -> List[str]:
        """Generate variations of the system prompt template."""
        if SYSTEM_PROMPT_KEY not in variation_fields or not variation_fields[SYSTEM_PROMPT_KEY]:
            return [system_prompt_template]
        variation_types = variation_fields[SYSTEM_PROMPT_KEY]
        all_variations = []
        for variation_type in variation_types:
            try:
                augmenter = AugmenterFactory.create(
                    variation_type=variation_type,
                    n_augments=variation_config.variations_per_field,
                    api_key=variation_config.api_key
                )
                variations = AugmenterFactory.augment_with_special_handling(
                    augmenter=augmenter,
                    text=system_prompt_template,
                    variation_type=variation_type
                )
                string_variations = AugmenterFactory.extract_text_from_result(variations, variation_type)
                all_variations.extend(string_variations[:variation_config.variations_per_field])
            except Exception as e:
                print(f"⚠️ Error generating {variation_type} variations for system_prompt: {e}")
                continue
        unique_variations = []
        seen = set()
        for var in all_variations:
            if var not in seen:
                unique_variations.append(var)
                seen.add(var)
        if system_prompt_template not in unique_variations:
            unique_variations.insert(0, system_prompt_template)
        return unique_variations[:variation_config.variations_per_field]

    def generate_all_field_variations(
            self,
            instruction_template: str,
            system_prompt_template: str,
            variation_fields: Dict[str, List[str]],
            row: pd.Series,
            variation_config: VariationConfig,
            gold_config
    ) -> Dict[str, List[FieldVariation]]:
        """Generate variations for all fields that have variation types specified."""

        field_variations = {}

        # Generate instruction variations
        if INSTRUCTION_KEY in variation_fields and variation_fields[INSTRUCTION_KEY]:
            instruction_vars = self.generate_instruction_variations(
                instruction_template, variation_fields, variation_config
            )
            # Convert to FieldVariation objects
            field_variations[INSTRUCTION_KEY] = [FieldVariation(data=var, gold_update=None) for var in instruction_vars]
        else:
            field_variations[INSTRUCTION_KEY] = [FieldVariation(data=instruction_template, gold_update=None)]

        # Generate variations for other fields (including system_prompt)
        for field_name, variation_types in variation_fields.items():
            if field_name == INSTRUCTION_KEY:
                continue  # Already handled above

            # Special handling for system_prompt variations – הם פועלים על הטמפלט, לא על ערך שורה
            if field_name == SYSTEM_PROMPT_KEY:
                # If system_prompt_template is not provided, skip
                if system_prompt_template is None:
                    continue

                sys_prompt_vars = self.generate_system_prompt_variations(
                    system_prompt_template,
                    variation_fields,
                    variation_config
                )

                field_variations[SYSTEM_PROMPT_KEY] = [FieldVariation(data=var, gold_update=None) for var in sys_prompt_vars]
                continue

            # Assume clean data - process all fields that exist in the row
            if field_name in row.index:
                field_value = format_field_value(row[field_name])
                field_data = FieldAugmentationData(
                    field_name=field_name,
                    field_value=field_value,
                    variation_types=variation_types,
                    variation_config=variation_config,
                    row_data=row,
                    gold_config=gold_config
                )
                field_variations[field_name] = self.generate_field_variations(field_data)
            else:
                # If field not in data, use empty variations
                field_variations[field_name] = [FieldVariation(data='', gold_update=None)]

        # Special handling for shuffle augmenter (and others) - update gold value extraction
        # (No direct row[gold_config.field] here, but if you add, use extract_gold_value)

        return field_variations

    def generate_field_variations(
            self,
            field_data: FieldAugmentationData
    ) -> List[FieldVariation]:
        """Generate variations for a specific field."""

        # Start with original - ensure it's formatted even if no variations are applied
        original_formatted = format_field_value(field_data.field_value)
        # If this is the gold field, set gold_update to the original value
        if field_data.gold_config and field_data.gold_config.field == field_data.field_name:
            original_gold_update = {field_data.field_name: original_formatted}
        else:
            original_gold_update = None
        all_variations = [FieldVariation(data=original_formatted, gold_update=original_gold_update)]

        for variation_type in field_data.variation_types:
            try:
                # Use Factory to create augmenter with proper configuration
                augmenter = AugmenterFactory.create(
                    variation_type=variation_type,
                    n_augments=field_data.variation_config.variations_per_field,
                    api_key=field_data.variation_config.api_key
                )

                # Special handling for shuffle augmenter
                if variation_type == 'shuffle':
                    if not field_data.has_gold_field():
                        print(f"⚠️ Shuffle augmenter requires gold field '{field_data.gold_config.field}' to be present in data")
                        continue

                    # Prepare identification data based on gold type
                    if field_data.gold_config.type == 'index':
                        # For index-based gold, pass the index directly
                        try:
                            gold_index = int(extract_gold_value(field_data.row_data, field_data.gold_config.field))
                            identification_data = {
                                'gold_field': field_data.gold_config.field,
                                'gold_value': str(gold_index)
                            }
                        except (ValueError, TypeError):
                            print(
                                f"⚠️ Gold field '{field_data.gold_config.field}' must contain valid integer indices for shuffle operation")
                            continue
                    else:
                        # For value-based gold, pass the value and let augmenter find the index
                        identification_data = {
                            'gold_field': field_data.gold_config.field,
                            'gold_value': str(extract_gold_value(field_data.row_data, field_data.gold_config.field))
                        }

                    variations = AugmenterFactory.augment_with_special_handling(
                        augmenter=augmenter,
                        text=field_data.field_value,
                        variation_type=variation_type,
                        identification_data=identification_data
                    )

                    if variations and isinstance(variations, list):
                        for var in variations:
                            if isinstance(var, dict) and 'shuffled_data' in var and 'new_gold_index' in var:
                                # For index-based gold, update with new index
                                # For value-based gold, convert index back to value if needed
                                if field_data.gold_config.type == 'index':
                                    gold_update_value = var['new_gold_index']
                                else:
                                    # For value-based, we might need to extract the actual value
                                    # from the shuffled options, but for now keep the index
                                    gold_update_value = var['new_gold_index']

                                variation_data = FieldVariation(
                                    data=var['shuffled_data'],
                                    gold_update={field_data.gold_config.field: gold_update_value}
                                )
                                if variation_data not in all_variations:
                                    all_variations.append(variation_data)
                else:
                    # Regular augmenters
                    variations = AugmenterFactory.augment_with_special_handling(
                        augmenter=augmenter,
                        text=field_data.field_value,
                        variation_type=variation_type
                    )

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
                                variation_data = FieldVariation(data=text_data, gold_update=None)
                            else:
                                # Standard string return
                                variation_data = FieldVariation(data=var, gold_update=None)

                            if variation_data not in all_variations:
                                all_variations.append(variation_data)

            except Exception as e:
                print(f"⚠️ Error generating {variation_type} variations for field {field_data.field_name}: {e}")
                continue

        # Remove duplicates while preserving order and limit to variations_per_field (original)
        unique_variations = []
        seen = set()
        for var in all_variations:
            var_key = (var.data, str(var.gold_update))
            if var_key not in seen:
                unique_variations.append(var)
                seen.add(var_key)

        return unique_variations[:field_data.variation_config.variations_per_field] 