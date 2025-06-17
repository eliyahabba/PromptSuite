"""
Few Shot Handler: Handles few-shot examples and row variation creation.
"""

import itertools
from typing import Dict, List, Any
import pandas as pd

from multipromptify.augmentations.structure.fewshot import FewShotAugmenter
from multipromptify.models import VariationContext, FieldVariation, FewShotContext


class FewShotHandler:
    """
    Handles few-shot examples and creation of row variations.
    """

    def __init__(self):
        self.few_shot_augmenter = FewShotAugmenter()

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
        
        if varying_fields:
            variation_combinations = list(itertools.product(*[variation_context.field_variations[field] for field in varying_fields]))
            
            for combination in variation_combinations:
                if len(variations) >= max_variations:
                    break
                    
                field_values = dict(zip(varying_fields, combination))
                instruction_variant = field_values.get('instruction', variation_context.field_variations.get('instruction', [
                    FieldVariation(data='', gold_update=None)])[0]).data
                    
                row_values = {}
                gold_updates = {}
                
                for col in variation_context.row_data.index:
                    # Handle pandas array comparison issue
                    try:
                        is_not_na = pd.notna(variation_context.row_data[col])
                        if hasattr(is_not_na, '__len__') and len(is_not_na) > 1:
                            # For arrays/lists, check if any element is not na
                            is_not_na = is_not_na.any() if hasattr(is_not_na, 'any') else True
                    except (ValueError, TypeError):
                        # Fallback: assume not na if we can't check
                        is_not_na = variation_context.row_data[col] is not None

                    if is_not_na:
                        if col in field_values:
                            field_data = field_values[col]
                            row_values[col] = field_data.data
                            if field_data.gold_update:
                                gold_updates.update(field_data.gold_update)
                        elif variation_context.gold_config.field and col == variation_context.gold_config.field:
                            continue
                        else:
                            row_values[col] = str(variation_context.row_data[col])
                
                # Generate few-shot examples for each instruction_variant
                few_shot_examples = []
                if few_shot_field and variation_context.data is not None:
                    # Create few-shot context
                    few_shot_context = FewShotContext(
                        instruction_template=instruction_variant,
                        few_shot_field=few_shot_field,
                        data=variation_context.data,
                        current_row_idx=variation_context.row_index,
                        gold_config=variation_context.gold_config
                    )
                    few_shot_examples = self.few_shot_augmenter.augment(
                        instruction_variant, 
                        few_shot_context.to_identification_data()
                    )
                
                main_input = prompt_builder.fill_template_placeholders(instruction_variant, row_values)
                if variation_context.gold_config.field:
                    main_input = main_input.replace(f'{{{variation_context.gold_config.field}}}', '')
                main_input = main_input.strip()
                
                conversation_messages = []
                for example in few_shot_examples:
                    conversation_messages.append({
                        "role": "user",
                        "content": example["question"]
                    })
                    conversation_messages.append({
                        "role": "assistant",
                        "content": example["answer"]
                    })
                
                if main_input:
                    conversation_messages.append({
                        "role": "user",
                        "content": main_input
                    })
                
                prompt_parts = []
                if few_shot_examples:
                    few_shot_content = self.few_shot_augmenter.format_few_shot_as_string(few_shot_examples)
                    prompt_parts.append(few_shot_content)
                
                if main_input:
                    prompt_parts.append(main_input)
                
                final_prompt = '\n\n'.join(prompt_parts)
                
                output_field_values = {}
                for field_name, field_data in field_values.items():
                    output_field_values[field_name] = field_data.data
                
                variations.append({
                    'prompt': final_prompt,
                    'conversation': conversation_messages,
                    'original_row_index': variation_context.row_index,
                    'variation_count': len(variations) + 1,
                    'template_config': variation_context.template,
                    'field_values': output_field_values,
                    'gold_updates': gold_updates if gold_updates else None,
                })
                
        return variations 