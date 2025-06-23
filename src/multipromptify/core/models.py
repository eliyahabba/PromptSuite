"""
Data models for MultiPromptify to manage parameters and context.
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional

import pandas as pd

from multipromptify.shared.constants import GenerationDefaults


@dataclass
class GoldFieldConfig:
    """Configuration for the gold field (correct output/label field)."""
    field: Optional[str] = None
    type: str = 'value'  # 'value' or 'index'
    options_field: Optional[str] = None

    @classmethod
    def from_template(cls, gold_config: Any) -> 'GoldFieldConfig':
        """Create GoldFieldConfig from template configuration."""
        if isinstance(gold_config, str):
            # Backward compatibility: old format
            return cls(field=gold_config, type='value', options_field=None)
        elif isinstance(gold_config, dict):
            # New format
            return cls(
                field=gold_config.get('field'),
                type=gold_config.get('type', 'value'),
                options_field=gold_config.get('options_field')
            )
        else:
            return cls(field=None, type='value', options_field=None)


@dataclass
class VariationConfig:
    """Configuration for generating variations."""
    variations_per_field: int = GenerationDefaults.VARIATIONS_PER_FIELD
    api_key: Optional[str] = None
    max_variations_per_row: Optional[int] = GenerationDefaults.MAX_VARIATIONS_PER_ROW
    seed: Optional[int] = GenerationDefaults.RANDOM_SEED


@dataclass
class FieldVariation:
    """Represents a single field variation with its data and potential gold updates."""
    data: str
    gold_update: Optional[Dict[str, Any]] = None


@dataclass
class VariationContext:
    """Context for generating variations for a single row."""
    row_data: pd.Series
    row_index: int
    template: dict
    field_variations: Dict[str, List[FieldVariation]]
    gold_config: GoldFieldConfig
    variation_config: VariationConfig
    data: Optional[pd.DataFrame] = None  # Full dataset for few-shot examples

    def get_field_value(self, field_name: str) -> Optional[str]:
        """Get field value from row data. Assumes clean data."""
        if field_name not in self.row_data.index:
            return None
        return str(self.row_data[field_name])


@dataclass
class FieldAugmentationData:
    """Data needed for generating variations on a specific field."""
    field_name: str
    field_value: str
    variation_types: List[str]
    variation_config: VariationConfig
    row_data: Optional[pd.Series] = None
    gold_config: Optional[GoldFieldConfig] = None

    def has_gold_field(self) -> bool:
        """Check if gold field is properly configured."""
        return (self.gold_config is not None and
                self.gold_config.field is not None and
                self.row_data is not None and
                self.gold_config.field in self.row_data.index)


@dataclass
class FewShotContext:
    """Context for generating few-shot examples."""
    prompt_format_template: str
    few_shot_field: Any
    data: pd.DataFrame
    current_row_idx: int
    gold_config: GoldFieldConfig

    def to_identification_data(self) -> Dict[str, Any]:
        """Convert to identification data format expected by FewShotAugmenter."""
        return {
            'few_shot_field': self.few_shot_field,
            'data': self.data,
            'current_row_idx': self.current_row_idx,
            'gold_field': self.gold_config.field,
            'gold_type': self.gold_config.type,
            'options_field': self.gold_config.options_field
        }
