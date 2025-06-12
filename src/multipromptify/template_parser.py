"""
Template parser for MultiPromptify templates with dictionary format.
"""

from typing import Dict, List, Tuple, Set, Optional, Union
from dataclasses import dataclass


@dataclass
class TemplateField:
    """Represents a field in a template with its variation types."""
    name: str
    variation_types: List[str] = None
    is_literal: bool = False
    # Few-shot specific parameters
    few_shot_count: Optional[int] = None
    few_shot_format: Optional[str] = None  # 'fixed' for same examples, 'rotating' for different
    few_shot_split: Optional[str] = None   # 'train', 'test', or 'all' for data splitting
    
    def __post_init__(self):
        """Ensure variation_types is always a list"""
        if self.variation_types is None:
            self.variation_types = []


class TemplateParser:
    """
    Parses MultiPromptify templates with dictionary format.
    
    Dictionary format:
    {
        "instruction_template": "Answer the following question: {question}\nAnswer: {answer}",
        "instruction": ["paraphrase", "surface"],
        "gold": "answer",  # Name of the column containing the correct answer/label
        "few_shot": {
            "count": 2,
            "format": "fixed",  # or "rotating"
            "split": "train"    # or "test" or "all"
        },
        "question": ["surface"]
    }
    """
    
    def __init__(self):
        self.fields: List[TemplateField] = []
        self.instruction_template: Optional[str] = None
        
    def parse(self, template: dict) -> List[TemplateField]:
        """
        Parse a dictionary template to extract fields and their variation types.
        
        Args:
            template: Dictionary template with field names as keys
            
        Returns:
            List of TemplateField objects
        """
        if not isinstance(template, dict):
            raise ValueError("Template must be a dictionary")
        
        self.fields = []
        self.instruction_template = None
        
        # Extract instruction template if provided
        if 'instruction_template' in template:
            self.instruction_template = template['instruction_template']
        
        for field_name, config in template.items():
            if field_name == "instruction_template":
                # Skip - already handled above
                continue
            elif field_name == "gold":
                # Skip - gold is metadata, not a field
                continue
            elif field_name == "few_shot":
                # Special handling for few_shot
                if isinstance(config, dict):
                    field = TemplateField(
                        name="few_shot",
                        variation_types=[],
                        few_shot_count=config.get("count", 2),
                        few_shot_format=config.get("format", "rotating"),
                        few_shot_split=config.get("split", "all")
                    )
                else:
                    raise ValueError("few_shot configuration must be a dictionary with 'count', 'format', and 'split' keys")
            else:
                # Regular fields with variation list
                if isinstance(config, list):
                    variation_types = config
                elif isinstance(config, str):
                    variation_types = [config]
                else:
                    variation_types = []
                
                field = TemplateField(
                    name=field_name,
                    variation_types=variation_types,
                    is_literal=field_name.startswith('_')
                )
            
            self.fields.append(field)
        
        return self.fields
    
    def get_instruction_template(self) -> Optional[str]:
        """Get the instruction template string."""
        return self.instruction_template
    
    def get_required_columns(self, template: dict = None) -> Set[str]:
        """
        Get the set of column names required from the data.
        
        Args:
            template: Optional template dict to check for gold field
        
        Returns:
            Set of column names that should be present in the input data
        """
        required = set()
        
        # Extract from instruction template
        if self.instruction_template:
            import re
            placeholders = re.findall(r'\{([^}]+)\}', self.instruction_template)
            for placeholder in placeholders:
                # Remove any variation annotations if present
                field_name = placeholder.split(':')[0].strip()
                if field_name not in {'instruction', 'few_shot'}:
                    required.add(field_name)
        
        # Extract from field definitions
        for field in self.fields:
            if not field.is_literal and field.name not in {'instruction', 'few_shot', 'gold'}:
                required.add(field.name)
            
            # For few-shot with split, we might need a split column
            if field.name == 'few_shot' and field.few_shot_split in ['train', 'test']:
                required.add('split')  # Convention: 'split' column indicates train/test
        
        # Check if gold field value exists in columns
        if template and 'gold' in template:
            gold_column = template['gold']
            required.add(gold_column)
                
        return required
    
    def get_variation_fields(self) -> Dict[str, List[str]]:
        """
        Get mapping of field names to their variation types.
        
        Returns:
            Dictionary mapping field names to lists of variation types
        """
        return {
            field.name: field.variation_types 
            for field in self.fields 
            if field.variation_types
        }
    
    def get_few_shot_fields(self) -> List[TemplateField]:
        """
        Get all few-shot fields with their parameters.
        
        Returns:
            List of TemplateField objects that are few-shot fields
        """
        return [field for field in self.fields if field.name == 'few_shot']
    
    def validate_template(self, template: dict) -> Tuple[bool, List[str]]:
        """
        Validate a template dictionary and return any errors.
        
        Args:
            template: Template dictionary to validate
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        if not isinstance(template, dict):
            return False, ["Template must be a dictionary"]
        
        if not template:
            return False, ["Template cannot be empty"]
        
        # Check if instruction_template is provided when instruction variations are requested
        if 'instruction' in template and template['instruction']:
            if 'instruction_template' not in template:
                errors.append("instruction_template is required when instruction variations are specified")
        
        try:
            fields = self.parse(template)
            # Check required columns
            required_columns = self.get_required_columns(template)
        except ValueError as e:
            return False, [str(e)]
        
        # Validate few-shot configuration
        for field in fields:
            if field.name == 'few_shot':
                if field.few_shot_count and field.few_shot_count <= 0:
                    errors.append(f"Few-shot count must be positive, got {field.few_shot_count}")
                
                if field.few_shot_format not in ['fixed', 'rotating']:
                    errors.append(f"Few-shot format must be 'fixed' or 'rotating', got {field.few_shot_format}")
                
                if field.few_shot_split not in ['all', 'train', 'test']:
                    errors.append(f"Few-shot split must be 'all', 'train', or 'test', got {field.few_shot_split}")
        
        # Check for valid variation types
        valid_variations = {'paraphrase', 'surface', 'context', 'multiple-choice', 'multidoc'}
        for field in fields:
            if field.name != 'few_shot':
                for variation_type in field.variation_types:
                    if variation_type not in valid_variations:
                        errors.append(f"Unknown variation type '{variation_type}' for field '{field.name}'. Valid types: {valid_variations}")
        
        # Validate instruction template syntax if provided
        if self.instruction_template:
            if self.instruction_template.count('{') != self.instruction_template.count('}'):
                errors.append("Mismatched brackets in instruction_template")
        
        return len(errors) == 0, errors 