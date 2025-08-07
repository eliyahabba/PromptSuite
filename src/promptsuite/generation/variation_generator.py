"""
Variation Generator: Handles generation of field variations and prompt_format variations.
"""

import random
from typing import Dict, List

import pandas as pd

from promptsuite.augmentations.factory import AugmenterFactory
from promptsuite.core.models import (
    VariationConfig, FieldVariation, FieldAugmentationData
)
from promptsuite.core.template_keys import (
    PROMPT_FORMAT_VARIATIONS, SHUFFLE_VARIATION, ENUMERATE_VARIATION,
    INSTRUCTION_VARIATIONS, FEW_SHOT_KEY
)
from promptsuite.utils.formatting import format_field_value, extract_gold_value


class VariationGenerator:
    """
    Handles the generation of variations for fields and prompt_formats.
    """

    def generate_prompt_format_variations(
            self,
            prompt_format: str,
            variation_fields: Dict[str, List[str]],
            variation_config: VariationConfig
    ) -> List[str]:
        """Generate variations of the prompt_format template."""

        if PROMPT_FORMAT_VARIATIONS not in variation_fields or not variation_fields[PROMPT_FORMAT_VARIATIONS]:
            return [prompt_format]

        variation_types = variation_fields[PROMPT_FORMAT_VARIATIONS]
        all_variations = []

        # Generate variations for each type
        for variation_type in variation_types:
            try:
                # Use Factory to create augmenter with proper configuration
                augmenter = AugmenterFactory.create(
                    variation_type=variation_type,
                    n_augments=variation_config.variations_per_field,
                    api_key=variation_config.api_key,
                    seed=variation_config.seed,
                    model_name=variation_config.model_name,
                    api_platform=variation_config.api_platform
                )

                # Use Factory to handle augmentation with special cases
                variations = AugmenterFactory.augment_with_special_handling(
                    augmenter=augmenter,
                    text=prompt_format,
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
        if prompt_format not in unique_variations:
            unique_variations.insert(0, prompt_format)

        return unique_variations[:variation_config.variations_per_field]

    def generate_instruction_variations(
            self,
            instruction: str,
            variation_fields: Dict[str, List[str]],
            variation_config: VariationConfig
    ) -> List[str]:
        """Generate variations of the system prompt template."""
        if INSTRUCTION_VARIATIONS not in variation_fields or not variation_fields[INSTRUCTION_VARIATIONS]:
            return [instruction]
        variation_types = variation_fields[INSTRUCTION_VARIATIONS]
        all_variations = []
        for variation_type in variation_types:
            try:
                augmenter = AugmenterFactory.create(
                    variation_type=variation_type,
                    n_augments=variation_config.variations_per_field,
                    api_key=variation_config.api_key,
                    seed=variation_config.seed,
                    model_name=variation_config.model_name,
                    api_platform=variation_config.api_platform
                )
                variations = AugmenterFactory.augment_with_special_handling(
                    augmenter=augmenter,
                    text=instruction,
                    variation_type=variation_type
                )
                string_variations = AugmenterFactory.extract_text_from_result(variations, variation_type)
                all_variations.extend(string_variations[:variation_config.variations_per_field])
            except Exception as e:
                print(f"⚠️ Error generating {variation_type} variations for instruction: {e}")
                continue
        unique_variations = []
        seen = set()
        for var in all_variations:
            if var not in seen:
                unique_variations.append(var)
                seen.add(var)
        if instruction not in unique_variations:
            unique_variations.insert(0, instruction)
        return unique_variations[:variation_config.variations_per_field]

    def generate_few_shot_variations(
            self,
            few_shot_config: dict,
            variation_config: VariationConfig
    ) -> List[FieldVariation]:
        """
        Generate few-shot variations for formats that create meaningful variations.
        
        Args:
            few_shot_config: Few-shot configuration from template
            variation_config: Variation configuration
            
        Returns:
            List of FieldVariation objects representing different few-shot configurations
        """
        few_shot_format = few_shot_config.get('format', 'same_examples__no_variations')

        # If format is 'same_examples__no_variations', no variations are generated for few-shot examples.
        # This implicitly handles the previous 'generate_variations: False' case.
        if few_shot_format == 'same_examples__no_variations':
            # Create a copy to ensure _order_seed or _selection_seed are not added for this format
            config_copy = few_shot_config.copy()
            config_copy.pop('_order_seed', None) # Remove if exists
            config_copy.pop('_selection_seed', None) # Remove if exists
            return [FieldVariation(data=config_copy, gold_update=None)]

        variations = []

        if few_shot_format == 'same_examples__synchronized_order_variations':
            # For synchronized order variations, we need to find seeds that actually produce different orderings
            # Since we only have 2 items, there are only 2 possible orderings: [0,1] and [1,0]
            few_shot_count = few_shot_config.get('count', 2)
            
            # Test a wide range of seeds to find ones that produce different orderings
            import random
            seen_orderings = set()
            tested_seeds = []
            
            # Test many seeds to find diverse orderings using pandas (to match actual behavior)
            import pandas as pd
            temp_data = pd.DataFrame({'idx': range(few_shot_count)})
            
            for i in range(1000):  # Test up to 1000 seeds
                test_seed = i + 1  # Start from 1
                # Use pandas sampling to match the actual behavior in FewShotAugmenter
                shuffled = temp_data.sample(frac=1.0, random_state=test_seed)
                ordering_signature = tuple(shuffled['idx'].tolist())
                
                if ordering_signature not in seen_orderings:
                    seen_orderings.add(ordering_signature)
                    tested_seeds.append(test_seed)
                
                # Stop if we have enough unique orderings
                if len(tested_seeds) >= variation_config.variations_per_field:
                    break
            
            # Create variations using only the unique seeds found
            # Return only the number of unique orderings possible, don't repeat
            for seed in tested_seeds:
                config_variation = few_shot_config.copy()
                config_variation['_order_seed'] = seed
                variations.append(FieldVariation(data=config_variation, gold_update=None))
        
        elif few_shot_format == 'different_examples__different_order_per_variation':
            # For different examples and different order per variation
            # The number of possible unique selections depends on the dataset size and few-shot count
            # We'll generate up to the requested number, but not more than what makes sense
            for i in range(variation_config.variations_per_field):
                config_variation = few_shot_config.copy()
                # Use a more spread out seed range to ensure different selections
                config_variation['_selection_seed'] = (variation_config.seed or 42) * 100 + i * 23
                variations.append(FieldVariation(data=config_variation, gold_update=None))
        elif few_shot_format == 'different_examples__same_shuffling_order_across_rows':
            # Generate variations by selecting different examples but applying the same shuffling order for each variation.
            for i in range(variation_config.variations_per_field):
                config_variation = few_shot_config.copy()
                # Selection seed varies per variation to get different examples
                config_variation['_selection_seed'] = (variation_config.seed or 42) * 1000 + i * 73
                # Order seed is fixed for this specific few-shot variation, so all rows use the same shuffle order for this variation
                config_variation['_order_seed'] = (variation_config.seed or 1) * 50 + i * 13 # A new deterministic seed for order
                variations.append(FieldVariation(data=config_variation, gold_update=None))
        else:
            # Fallback for unexpected formats (should ideally not happen with valid inputs)
            print(f"⚠️ Unexpected few-shot format '{few_shot_format}' in VariationGenerator, returning single configuration.")
            return [FieldVariation(data=few_shot_config, gold_update=None)]
        
        # Return only the requested number of variations
        return variations[:variation_config.variations_per_field]

    def generate_row_specific_field_variations(
            self,
            variation_fields: Dict[str, List[str]],
            row: pd.Series,
            variation_config: VariationConfig,
            gold_config,
            pre_generated_variations: Dict[str, List[FieldVariation]],
            template: dict = None
    ) -> Dict[str, List[FieldVariation]]:
        """
        Generate variations for row-specific fields only (excluding instruction and prompt format variations).
        This method uses pre-generated variations for instruction and prompt format to avoid
        running the same augmenters multiple times.
        """
        field_variations = {}

        # Use pre-generated instruction variations
        field_variations[INSTRUCTION_VARIATIONS] = pre_generated_variations[INSTRUCTION_VARIATIONS]

        # Use pre-generated prompt format variations
        field_variations[PROMPT_FORMAT_VARIATIONS] = pre_generated_variations[PROMPT_FORMAT_VARIATIONS]

        # Generate variations for other fields (row-specific fields only)
        for field_name, variation_types in variation_fields.items():
            if field_name in [PROMPT_FORMAT_VARIATIONS, INSTRUCTION_VARIATIONS]:
                continue  # Already handled above

            # Special handling for few-shot variations
            if field_name == FEW_SHOT_KEY and 'few_shot_variation' in variation_types:
                if template and FEW_SHOT_KEY in template:
                    few_shot_config = template[FEW_SHOT_KEY]
                    field_variations[field_name] = self.generate_few_shot_variations(
                        few_shot_config, variation_config
                    )
                else:
                    # Fallback if template not available
                    field_variations[field_name] = [FieldVariation(data={}, gold_update=None)]
                continue

            # Assume clean data - process all fields that exist in the row
            if field_name in row.index:
                field_value = row[field_name]  # Keep original value (don't format yet)
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

        return field_variations

    def generate_field_variations(
            self,
            field_data: FieldAugmentationData
    ) -> List[FieldVariation]:
        """
        Generate chained variations for a specific field.
        If multiple augmenters are specified (e.g., shuffle and enumerate),
        apply them in a fixed order: shuffle first, then enumerate, regardless of their order in the template.
        Use deterministic sampling to select a subset of variations for consistency across rows.
        """
        
        # If no variation types, return the original value (formatted)
        if not field_data.variation_types:
            original_formatted = format_field_value(field_data.field_value)
            if field_data.gold_config and field_data.gold_config.field == field_data.field_name:
                original_gold_update = {field_data.field_name: original_formatted}
            else:
                original_gold_update = None
            return [FieldVariation(data=original_formatted, gold_update=original_gold_update)]

        # Always apply shuffle before enumerate if both are present
        variation_types = list(field_data.variation_types)
        ordered_types = []
        if SHUFFLE_VARIATION in variation_types:
            ordered_types.append(SHUFFLE_VARIATION)
        if ENUMERATE_VARIATION in variation_types:
            ordered_types.append(ENUMERATE_VARIATION)
        # Add any other augmenters (excluding shuffle/enumerate) in their original order
        for vtype in variation_types:
            if vtype not in ordered_types:
                ordered_types.append(vtype)

        # Start with the original value - keep it as is for processing
        # Don't format until the very end to preserve list structure for enumerate/shuffle
        current_variations = [field_data.field_value]
        current_gold_updates = [None] * len(current_variations)

        for variation_type in ordered_types:
            next_variations = []
            next_gold_updates = []
            for idx, var in enumerate(current_variations):
                augmenter = AugmenterFactory.create(
                    variation_type=variation_type,
                    n_augments=field_data.variation_config.variations_per_field,
                    api_key=field_data.variation_config.api_key,
                    seed=field_data.variation_config.seed,
                    model_name=field_data.variation_config.model_name,
                    api_platform=field_data.variation_config.api_platform
                )
                # Special handling for shuffle
                if variation_type == SHUFFLE_VARIATION:
                    if not field_data.has_gold_field():
                        print(
                            f"⚠️ Shuffle augmenter requires gold field '{field_data.gold_config.field}' to be present in data")
                        continue
                    # Prepare identification data for shuffle
                    if field_data.gold_config.type == 'index':
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
                        identification_data = {
                            'gold_field': field_data.gold_config.field,
                            'gold_value': str(extract_gold_value(field_data.row_data, field_data.gold_config.field))
                        }
                    variations = AugmenterFactory.augment_with_special_handling(
                        augmenter=augmenter,
                        text=var,  # Pass original value (could be list)
                        variation_type=variation_type,
                        identification_data=identification_data
                    )
                    # Each shuffle variation is a dict with 'shuffled_data' and 'new_gold_index'
                    if variations and isinstance(variations, list):
                        for v in variations:
                            if isinstance(v, dict) and 'shuffled_data' in v:
                                next_variations.append(v['shuffled_data'])
                                # Track gold update if needed
                                if (field_data.gold_config
                                        and field_data.gold_config.field
                                        and 'new_gold_index' in v
                                        and field_data.gold_config.type == 'index'):
                                    # Always update the gold field specified in the gold configuration
                                    next_gold_updates.append({field_data.gold_config.field: v['new_gold_index']})
                                else:
                                    next_gold_updates.append(None)
                # Special handling for enumerate
                elif variation_type == ENUMERATE_VARIATION:
                    variations = AugmenterFactory.augment_with_special_handling(
                        augmenter=augmenter,
                        text=var,  # Pass original value (could be list)
                        variation_type=variation_type
                    )
                    # Extract the full results from AugmenterFactory (which preserves metadata for enumerate)
                    extracted_results = AugmenterFactory.extract_text_from_result(variations, variation_type)
                    if extracted_results:
                        for result in extracted_results:
                            if isinstance(result, dict) and 'data' in result:
                                # Store the text and preserve enumeration type metadata
                                next_variations.append(result['data'])
                                # Store metadata about enumeration type in gold_updates for later use
                                # We'll pass it through the system and extract it in few_shot_handler
                                # Create a copy of the original gold_update for this specific variation
                                base_gold_update = current_gold_updates[idx]
                                if base_gold_update:
                                    gold_update = base_gold_update.copy() if isinstance(base_gold_update, dict) else {}
                                else:
                                    gold_update = {}
                                gold_update['_enum_type'] = result.get('enum_type', '1234')
                                next_gold_updates.append(gold_update)
                            else:
                                # Fallback for string results
                                next_variations.append(str(result))
                                next_gold_updates.append(current_gold_updates[idx])
                # Other augmenters (if any)
                else:
                    # For other augmenters, format the value first
                    formatted_var = format_field_value(var) if not isinstance(var, str) else var
                    variations = AugmenterFactory.augment_with_special_handling(
                        augmenter=augmenter,
                        text=formatted_var,
                        variation_type=variation_type
                    )
                    if variations and isinstance(variations, list):
                        for v in variations:
                            next_variations.append(v)
                            next_gold_updates.append(current_gold_updates[idx])
            # Update for next augmenter in the chain
            current_variations = next_variations
            current_gold_updates = next_gold_updates

        # Remove duplicates while preserving order and ensure all values are formatted
        unique = []
        seen = set()
        for i, v in enumerate(current_variations):
            # Always format to string at the very end (for display)
            formatted_v = format_field_value(v) if not isinstance(v, str) else v
            key = (formatted_v, str(current_gold_updates[i]))
            if key not in seen:
                # Extract enumeration metadata from gold_updates if present
                gold_update = current_gold_updates[i]
                metadata = None
                if isinstance(gold_update, dict) and '_enum_type' in gold_update:
                    metadata = {'enum_type': gold_update['_enum_type']}
                    # Remove the temporary key from gold_update
                    clean_gold_update = {k: v for k, v in gold_update.items() if k != '_enum_type'}
                    gold_update = clean_gold_update if clean_gold_update else None
                
                unique.append(FieldVariation(data=formatted_v, gold_update=gold_update, metadata=metadata))
                seen.add(key)
        # Deterministically sample the required number of variations using the configured random seed
        sample_seed = getattr(field_data.variation_config, 'random_seed', 42)
        sampled = self.deterministic_sample(unique, field_data.variation_config.variations_per_field, seed=sample_seed)
        return sampled

    @staticmethod
    def deterministic_sample(lst, k, seed=42):
        """
        Return a deterministic random sample of k elements from lst using the given seed.
        If lst has k or fewer elements, return all of them.
        """
        if len(lst) <= k:
            return lst
        rnd = random.Random(seed)
        idxs = list(range(len(lst)))
        rnd.shuffle(idxs)
        return [lst[i] for i in idxs[:k]]
