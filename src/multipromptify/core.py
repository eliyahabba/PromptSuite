"""
Core MultiPromptify class that orchestrates the entire prompt variation generation process.
"""

import pandas as pd
import json
import os
from typing import Dict, List, Any, Union, Optional

try:
    from datasets import Dataset
    HF_DATASETS_AVAILABLE = True
except ImportError:
    HF_DATASETS_AVAILABLE = False
    Dataset = None

from .template_parser import TemplateParser
from .variation_generator import VariationGenerator


class MultiPromptify:
    """
    Main class for generating multi-prompt datasets from single-prompt datasets.
    
    This class handles:
    - Template parsing with variation annotations
    - Data loading from various formats (CSV, DataFrame, HuggingFace datasets)
    - Variation generation based on specified types
    - Output formatting and saving
    """
    
    def __init__(self, max_variations: int = 100):
        """
        Initialize MultiPromptify.
        
        Args:
            max_variations: Maximum number of variations to generate
        """
        self.max_variations = max_variations
        self.template_parser = TemplateParser()
        self.variation_generator = VariationGenerator(max_variations)
    
    def generate_variations(
        self,
        template: str,
        data: Union[pd.DataFrame, str, dict, Any],
        instruction: Optional[str] = None,
        few_shot: Optional[Union[list, tuple]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Generate prompt variations based on template and data.
        
        Args:
            template: Template string with variation annotations
            data: Input data (DataFrame, CSV path, dict, or HuggingFace Dataset)
            instruction: Static instruction text (optional)
            few_shot: Few-shot examples (optional)
            **kwargs: Additional field values
            
        Returns:
            List of generated prompt variations
            
        Examples:
            >>> mp = MultiPromptify()
            >>> data = pd.DataFrame({'question': ['What is 2+2?'], 'answer': ['4']})
            >>> template = "{instruction:semantic}: {question:paraphrase}"
            >>> variations = mp.generate_variations(template, data, instruction="Answer this")
            >>> len(variations) > 1
            True
        """
        # Validate and parse template
        is_valid, errors = self.template_parser.validate_template(template)
        if not is_valid:
            raise ValueError(f"Invalid template: {'; '.join(errors)}")
        
        # Parse template to get fields and variation types
        fields = self.template_parser.parse(template)
        
        # Load and validate data
        df = self._load_data(data)
        
        # Validate that required columns exist
        required_columns = self.template_parser.get_required_columns()
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Prepare additional field values
        additional_fields = {
            'instruction': instruction,
            'few_shot': few_shot,
            **kwargs
        }
        
        # Generate variations for each row
        all_variations = []
        
        for idx, row in df.iterrows():
            row_variations = self._generate_variations_for_row(
                template, fields, row, additional_fields
            )
            
            # Add metadata
            for variation in row_variations:
                variation.update({
                    'original_row_index': idx,
                    'template': template,
                    'variation_count': len(row_variations)
                })
            
            all_variations.extend(row_variations)
        
        return all_variations
    
    def _load_data(self, data: Union[pd.DataFrame, str, dict, Any]) -> pd.DataFrame:
        """
        Load data from various formats into a pandas DataFrame.
        
        Args:
            data: Input data in various formats
            
        Returns:
            pandas DataFrame
        """
        if isinstance(data, pd.DataFrame):
            return data
        elif isinstance(data, str):
            # Assume it's a file path
            if data.endswith('.csv'):
                return pd.read_csv(data)
            elif data.endswith('.json'):
                return pd.read_json(data)
            else:
                raise ValueError(f"Unsupported file format: {data}")
        elif isinstance(data, dict):
            return pd.DataFrame(data)
        elif HF_DATASETS_AVAILABLE and hasattr(data, 'to_pandas'):  # HuggingFace Dataset
            return data.to_pandas()
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
    
    def _generate_variations_for_row(
        self,
        template: str,
        fields: List,
        row: pd.Series,
        additional_fields: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Generate variations for a single data row.
        
        Args:
            template: Template string
            fields: Parsed template fields
            row: Data row
            additional_fields: Additional field values
            
        Returns:
            List of variations for this row
        """
        # Prepare field values for this row
        field_values = {}
        field_variations = {}
        
        for field in fields:
            field_name = field.name
            
            # Get the value for this field
            if field_name in additional_fields and additional_fields[field_name] is not None:
                value = self._process_field_value(additional_fields[field_name], row.name)
            elif field_name in row:
                value = str(row[field_name])
            else:
                continue  # Skip missing fields
            
            field_values[field_name] = value
            
            # Generate variations if variation type is specified
            if field.variation_type:
                variations = self.variation_generator.generate_variations(
                    field_name, value, field.variation_type, count=3
                )
                field_variations[field_name] = variations
            else:
                field_variations[field_name] = [value]
        
        # Generate all combinations of variations
        combinations = self.variation_generator.generate_combinations(
            field_variations, max_combinations=self.max_variations
        )
        
        # Format each combination into a prompt
        formatted_variations = []
        for combo in combinations:
            try:
                formatted_prompt = self.template_parser.format_template(template, combo)
                formatted_variations.append({
                    'prompt': formatted_prompt,
                    'field_values': combo.copy(),
                    'original_values': field_values.copy()
                })
            except Exception as e:
                # Skip combinations that can't be formatted
                continue
        
        return formatted_variations
    
    def _process_field_value(self, value: Any, row_index: int) -> str:
        """
        Process field values based on their type.
        
        Args:
            value: Field value (can be literal, list, or tuple)
            row_index: Current row index
            
        Returns:
            Processed string value
        """
        if isinstance(value, (str, int, float)):
            # Literal value - use as is
            return str(value)
        elif isinstance(value, tuple):
            # Tuple - same values for entire dataset
            return " ".join(str(v) for v in value)
        elif isinstance(value, list):
            # List - different values per row
            if len(value) > row_index:
                item = value[row_index]
                if isinstance(item, list):
                    # List of lists - join inner list
                    return " ".join(str(v) for v in item)
                else:
                    return str(item)
            else:
                # Not enough values in list, use last available
                if value:
                    item = value[-1]
                    if isinstance(item, list):
                        return " ".join(str(v) for v in item)
                    else:
                        return str(item)
                else:
                    return ""
        else:
            return str(value)
    
    def parse_template(self, template: str) -> Dict[str, str]:
        """
        Parse template to extract columns and variation types.
        
        Args:
            template: Template string to parse
            
        Returns:
            Dictionary mapping field names to variation types
        """
        fields = self.template_parser.parse(template)
        return {field.name: field.variation_type for field in fields}
    
    def save_variations(
        self,
        variations: List[Dict[str, Any]],
        output_path: str,
        format: str = "json"
    ):
        """
        Save variations to file.
        
        Args:
            variations: List of generated variations
            output_path: Output file path
            format: Output format ('json', 'csv', 'hf')
        """
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        if format.lower() == "json":
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(variations, f, indent=2, ensure_ascii=False)
        
        elif format.lower() == "csv":
            # Flatten the variations for CSV format
            flattened = []
            for var in variations:
                flat_var = {
                    'prompt': var['prompt'],
                    'original_row_index': var.get('original_row_index', ''),
                    'variation_count': var.get('variation_count', ''),
                }
                # Add field values
                for key, value in var.get('field_values', {}).items():
                    flat_var[f'field_{key}'] = value
                flattened.append(flat_var)
            
            df = pd.DataFrame(flattened)
            df.to_csv(output_path, index=False, encoding='utf-8')
        
        elif format.lower() == "hf":
            # Save as HuggingFace dataset
            if not HF_DATASETS_AVAILABLE:
                raise ImportError("HuggingFace datasets library required for 'hf' format")
            
            # Prepare data for HuggingFace format
            dataset_dict = {
                'prompt': [var['prompt'] for var in variations],
                'original_row_index': [var.get('original_row_index', -1) for var in variations],
                'variation_count': [var.get('variation_count', 0) for var in variations],
            }
            
            # Add field values as separate columns
            if variations:
                field_keys = set()
                for var in variations:
                    field_keys.update(var.get('field_values', {}).keys())
                
                for key in field_keys:
                    dataset_dict[f'field_{key}'] = [
                        var.get('field_values', {}).get(key, '') for var in variations
                    ]
            
            from datasets import Dataset
            dataset = Dataset.from_dict(dataset_dict)
            dataset.save_to_disk(output_path)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def get_stats(self, variations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get statistics about generated variations.
        
        Args:
            variations: List of generated variations
            
        Returns:
            Dictionary with statistics
        """
        if not variations:
            return {'total_variations': 0}
        
        # Count variations per original row
        row_counts = {}
        field_types = set()
        
        for var in variations:
            row_idx = var.get('original_row_index', 0)
            row_counts[row_idx] = row_counts.get(row_idx, 0) + 1
            field_types.update(var.get('field_values', {}).keys())
        
        return {
            'total_variations': len(variations),
            'original_rows': len(row_counts),
            'avg_variations_per_row': sum(row_counts.values()) / len(row_counts),
            'max_variations_per_row': max(row_counts.values()),
            'min_variations_per_row': min(row_counts.values()),
            'unique_fields': len(field_types),
            'field_names': sorted(field_types)
        } 