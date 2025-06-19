"""
Template parser for MultiPromptify templates with dictionary format.
"""

from typing import Dict, List, Tuple, Set, Optional, Union
from dataclasses import dataclass
from multipromptify.core.exceptions import InvalidTemplateFieldError
from multipromptify.core.template_keys import (
    INSTRUCTION_TEMPLATE_KEY, INSTRUCTION_KEY, QUESTION_KEY, GOLD_KEY, FEW_SHOT_KEY,
    PARAPHRASE_WITH_LLM, REWORDING
)


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
    # Enumerate specific parameters
    enumerate_field: Optional[str] = None  # Which field to enumerate
    enumerate_type: Optional[str] = None   # Type of enumeration ('1234', 'ABCD', etc.)
    
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
        "gold": "output",  # Name of the column containing the correct output/label
        "few_shot": {
            "count": 2,
            "format": "fixed",  # or "rotating"
            "split": "train"    # or "test" or "all"
        },
        "input": ["surface"]
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
            raise InvalidTemplateFieldError("template", template, "dictionary")
        
        self.fields = []
        self.instruction_template = None
        
        # Extract instruction template if provided
        if INSTRUCTION_TEMPLATE_KEY in template:
            self.instruction_template = template[INSTRUCTION_TEMPLATE_KEY]
        
        for field_name, config in template.items():
            if field_name == INSTRUCTION_TEMPLATE_KEY:
                # Skip - already handled above
                continue
            elif field_name == GOLD_KEY:
                # Skip - gold is metadata, not a field
                continue
            elif field_name == FEW_SHOT_KEY:
                # Special handling for few_shot
                if isinstance(config, dict):
                    field = TemplateField(
                        name=FEW_SHOT_KEY,
                        variation_types=[],
                        few_shot_count=config.get("count", 2),
                        few_shot_format=config.get("format", "rotating"),
                        few_shot_split=config.get("split", "all")
                    )
                    self.fields.append(field)
                    continue
                else:
                    raise InvalidTemplateFieldError("few_shot", config, "dictionary with 'count', 'format', and 'split' keys")
            elif field_name == "enumerate":
                # Special handling for enumerate
                if isinstance(config, dict):
                    field = TemplateField(
                        name="enumerate",
                        variation_types=[],
                        enumerate_field=config.get("field", None),
                        enumerate_type=config.get("type", "1234")
                    )
                    self.fields.append(field)
                    continue
                else:
                    raise InvalidTemplateFieldError("enumerate", config, "dictionary with 'field' and 'type' keys")
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
                if field_name not in {INSTRUCTION_KEY, FEW_SHOT_KEY}:
                    required.add(field_name)
        
        # Extract from field definitions
        for field in self.fields:
            if not field.is_literal and field.name not in {INSTRUCTION_KEY, FEW_SHOT_KEY, GOLD_KEY}:
                required.add(field.name)
            
            # For few-shot with split, we might need a split column
            if field.name == FEW_SHOT_KEY and field.few_shot_split in ['train', 'test']:
                required.add('split')  # Convention: 'split' column indicates train/test
        
        # Check if gold field value exists in columns
        if template and GOLD_KEY in template:
            gold_config = template[GOLD_KEY]
            if isinstance(gold_config, str):
                # Old format: gold field is just the column name
                required.add(gold_config)
            elif isinstance(gold_config, dict) and 'field' in gold_config:
                # New format: gold field is a dict with 'field' key
                required.add(gold_config['field'])
                # If there's an options_field specified, add it too
                if 'options_field' in gold_config:
                    required.add(gold_config['options_field'])
                
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
        return [field for field in self.fields if field.name == FEW_SHOT_KEY]
    
    def get_enumerate_fields(self) -> List[TemplateField]:
        """
        Get all enumerate fields with their parameters.
        
        Returns:
            List of TemplateField objects that are enumerate fields
        """
        return [field for field in self.fields if field.name == "enumerate"]
    
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
        if INSTRUCTION_KEY in template and template[INSTRUCTION_KEY]:
            if INSTRUCTION_TEMPLATE_KEY not in template:
                errors.append("instruction_template is required when instruction variations are specified")
        
        try:
            fields = self.parse(template)
            # Check required columns
            required_columns = self.get_required_columns(template)
        except ValueError as e:
            return False, [str(e)]
        
        # Validate few-shot configuration
        for field in fields:
            if field.name == FEW_SHOT_KEY:
                if field.few_shot_count and field.few_shot_count <= 0:
                    errors.append(f"Few-shot count must be positive, got {field.few_shot_count}")
                
                if field.few_shot_format not in ['fixed', 'rotating']:
                    errors.append(f"Few-shot format must be 'fixed' or 'rotating', got {field.few_shot_format}")
                
                if field.few_shot_split not in ['all', 'train', 'test']:
                    errors.append(f"Few-shot split must be 'all', 'train', or 'test', got {field.few_shot_split}")
        
        # Validate enumerate configuration
        for field in fields:
            if field.name == "enumerate":
                if not field.enumerate_field:
                    errors.append("Enumerate field must specify which field to enumerate")
                
                if not field.enumerate_type:
                    errors.append("Enumerate type cannot be empty")
        
        # Check for valid variation types
        valid_variations = {PARAPHRASE_WITH_LLM, REWORDING, 'context', 'shuffle', 'multidoc', 'enumerate'}
        for field in fields:
            if field.name != FEW_SHOT_KEY:
                for variation_type in field.variation_types:
                    if variation_type not in valid_variations:
                        errors.append(f"Unknown variation type '{variation_type}' for field '{field.name}'. Valid types: {valid_variations}")
        
        # Validate instruction template syntax if provided
        if self.instruction_template:
            if self.instruction_template.count('{') != self.instruction_template.count('}'):
                errors.append("Mismatched brackets in instruction_template")
        
        return len(errors) == 0, errors 