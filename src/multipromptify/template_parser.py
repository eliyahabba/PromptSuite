"""
Template parser for MultiPromptify templates with variation annotations.
"""

import re
from typing import Dict, List, Tuple, Set, Optional, Union
from dataclasses import dataclass


@dataclass
class TemplateField:
    """Represents a field in a template with its variation type."""
    name: str
    variation_type: str = None
    is_literal: bool = False
    # Few-shot specific parameters
    few_shot_count: Optional[int] = None
    few_shot_format: Optional[str] = None  # 'list' for different per row, 'tuple' for same for all
    few_shot_split: Optional[str] = None   # 'train' or 'test' for data splitting


class TemplateParser:
    """
    Parses MultiPromptify templates with Python f-string syntax and variation annotations.
    
    Supports formats like:
    - {instruction:paraphrase}
    - {question:surface}
    - {few_shot:5} - 5 few-shot examples (default format)
    - {few_shot:[5} - 5 different examples per row (list format)
    - {few_shot:(5)} - 5 same examples for all rows (tuple format)
    - {few_shot:train[4} - 4 examples from train split, different per row
    - {few_shot:test(3)} - 3 examples from test split, same for all
    """
    
    # Regex pattern to match template fields including few-shot syntax
    FIELD_PATTERN = re.compile(r'\{([^}]+)\}')
    
    # Few-shot pattern for various formats:
    # {few_shot:5}, {few_shot:[5}, {few_shot:(5)}, {few_shot:train[2}, {few_shot:test(3)}
    FEW_SHOT_PATTERN = re.compile(r'^(?P<field>few_shot):(?:(?P<split>train|test))?(?P<format_marker>[\[\(])?(?P<count>\d+)(?P<closing_marker>[\]\)])?$')
    
    def __init__(self):
        self.fields: List[TemplateField] = []
        self.template: str = ""
        
    def parse(self, template: Union[str, dict]) -> List[TemplateField]:
        """
        Parse a template string to extract fields and their variation types.
        
        Args:
            template: Template string with f-string syntax and optional variation annotations,
                     OR dictionary with 'instruction' and 'template' keys
            
        Returns:
            List of TemplateField objects
        """
        # Handle both old string format and new dictionary format
        if isinstance(template, dict):
            # New format: {'instruction': '...', 'template': '...'}
            if 'instruction' in template and 'template' in template:
                # Parse both instruction and template parts
                instruction_text = template['instruction']
                template_text = template['template']
                
                # Combine fields from both parts
                self.template = template_text  # Store template part for main processing
                self.fields = []
                
                # Parse template part first (this contains the processing rules)
                template_matches = self.FIELD_PATTERN.findall(template_text)
                for match in template_matches:
                    field = self._parse_field(match)
                    if field not in self.fields:
                        self.fields.append(field)
                
                # Parse instruction part to find data fields (without variation types)
                instruction_matches = self.FIELD_PATTERN.findall(instruction_text)
                for match in instruction_matches:
                    # For instruction fields, we only care about the field name (no variations)
                    field_name = match.split(':')[0].strip() if ':' in match else match.strip()
                    
                    # Only add if it's not already in the list and not a special field
                    if field_name not in [f.name for f in self.fields] and field_name not in ['instruction', 'few_shot']:
                        field = TemplateField(name=field_name, variation_type=None, is_literal=field_name.startswith('_'))
                        self.fields.append(field)
                        
                return self.fields
            elif 'combined' in template:
                # Fallback for combined format
                self.template = template['combined']
                template_str = template['combined']
            else:
                # Unknown dictionary format, convert to string
                self.template = str(template)
                template_str = str(template)
        else:
            # Old format: simple string
            self.template = template
            template_str = template
        
        # Parse as string (old format or fallback)
        self.fields = []
        
        # Find all template fields
        matches = self.FIELD_PATTERN.findall(template_str)
        
        for match in matches:
            field = self._parse_field(match)
            if field not in self.fields:  # Avoid duplicates
                self.fields.append(field)
                
        return self.fields
    
    def _parse_field(self, field_str: str) -> TemplateField:
        """
        Parse a single field string to extract name and variation type.
        
        Args:
            field_str: Field string like 'instruction:paraphrase', 'few_shot:5', '[few_shot:[5', etc.
            
        Returns:
            TemplateField object
        """
        # Check if this is a few-shot field
        few_shot_match = self.FEW_SHOT_PATTERN.match(field_str)
        if few_shot_match:
            return self._parse_few_shot_field(few_shot_match)
        
        # Regular field parsing
        if ':' in field_str:
            name, variation_type = field_str.split(':', 1)
            variation_type = variation_type.strip()
        else:
            name = field_str
            variation_type = None
            
        name = name.strip()
        
        # Check if this is a literal field (starts with underscore by convention)
        is_literal = name.startswith('_')
        
        return TemplateField(
            name=name,
            variation_type=variation_type,
            is_literal=is_literal
        )
    
    def _parse_few_shot_field(self, match) -> TemplateField:
        """
        Parse a few-shot field with special syntax.
        
        Args:
            match: Regex match object from FEW_SHOT_PATTERN
            
        Returns:
            TemplateField object with few-shot parameters
        """
        field_name = match.group('field')  # 'few_shot'
        split = match.group('split')  # 'train' or 'test' or None
        format_marker = match.group('format_marker')  # '[' or '(' or None
        count = int(match.group('count'))  # number
        closing_marker = match.group('closing_marker')  # ']' or ')' or None
        
        # Determine format
        few_shot_format = None
        if format_marker == '[':
            few_shot_format = 'list'  # Different examples per row - {few_shot:[N} or {few_shot:[N]} or {few_shot:train[N]}
        elif format_marker == '(' and closing_marker == ')':
            few_shot_format = 'tuple'  # Same examples for all rows - {few_shot:(N)} or {few_shot:test(N)}
        # If no format specified, default to different per row
        
        return TemplateField(
            name=field_name,
            variation_type=None,  # Few-shot should not be a variation type
            is_literal=False,
            few_shot_count=count,
            few_shot_format=few_shot_format,
            few_shot_split=split
        )
    
    def get_required_columns(self) -> Set[str]:
        """
        Get the set of column names required from the data.
        
        Returns:
            Set of column names that should be present in the input data
        """
        required = set()
        for field in self.fields:
            if not field.is_literal and field.name not in {'instruction', 'few_shot'}:
                required.add(field.name)
            
            # For few-shot with split, we might need a split column
            if field.name == 'few_shot' and field.few_shot_split:
                required.add('split')  # Convention: 'split' column indicates train/test
                
        return required
    
    def get_variation_fields(self) -> Dict[str, str]:
        """
        Get mapping of field names to their variation types.
        
        Returns:
            Dictionary mapping field names to variation types
        """
        return {
            field.name: field.variation_type 
            for field in self.fields 
            if field.variation_type is not None
        }
    
    def get_few_shot_fields(self) -> List[TemplateField]:
        """
        Get all few-shot fields with their parameters.
        
        Returns:
            List of TemplateField objects that are few-shot fields
        """
        return [field for field in self.fields if field.name == 'few_shot']
    
    def validate_template(self, template: Union[str, dict]) -> Tuple[bool, List[str]]:
        """
        Validate a template string and return any errors.
        
        Args:
            template: Template string to validate OR dictionary with 'instruction' and 'template' keys
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        try:
            fields = self.parse(template)
        except ValueError as e:
            return False, [str(e)]
        
        # Get template string for validation
        if isinstance(template, dict):
            if 'instruction' in template and 'template' in template:
                template_str = template['template']
                instruction_str = template['instruction']
                
                # Validate both parts
                if template_str.count('{') != template_str.count('}'):
                    errors.append("Mismatched brackets in template part")
                if instruction_str.count('{') != instruction_str.count('}'):
                    errors.append("Mismatched brackets in instruction part")
            elif 'combined' in template:
                template_str = template['combined']
            else:
                template_str = str(template)
        else:
            template_str = template
        
        # Check for duplicate field names
        field_names = [f.name for f in fields]
        duplicates = set([name for name in field_names if field_names.count(name) > 1])
        if duplicates:
            errors.append(f"Duplicate field names: {', '.join(duplicates)}")
        
        # Check for empty template
        if not fields:
            errors.append("Template contains no fields")
        
        # Check for malformed brackets (for single string templates)
        if isinstance(template, str) and template.count('{') != template.count('}'):
            errors.append("Mismatched brackets in template")
            
        # Validate few-shot syntax
        for field in fields:
            if field.name == 'few_shot':
                if field.few_shot_count and field.few_shot_count <= 0:
                    errors.append(f"Few-shot count must be positive, got {field.few_shot_count}")
                    
                # Check format consistency for single string templates
                if isinstance(template, str):
                    template_to_check = template
                elif isinstance(template, dict) and 'template' in template:
                    template_to_check = template['template']
                else:
                    template_to_check = template_str
                    
                if field.few_shot_format == 'list':
                    # Check for {few_shot:[N} or {few_shot:[N]} or {few_shot:train[N]} pattern
                    if not re.search(r'\{few_shot:(?:train|test)?\[\d+\]?\}', template_to_check):
                        errors.append("List format few-shot should use syntax like {few_shot:[N]} or {few_shot:train[N]}")
                elif field.few_shot_format == 'tuple':
                    # Check for {few_shot:(N)} or {few_shot:test(N)} pattern
                    if not re.search(r'\{few_shot:(?:train|test)?\(\d+\)\}', template_to_check):
                        errors.append("Tuple format few-shot should use syntax like {few_shot:(N)} or {few_shot:test(N)}")
        
        return len(errors) == 0, errors
    
    def format_template(self, template: Union[str, dict], values: Dict[str, str]) -> str:
        """
        Format a template with provided values.
        
        Args:
            template: Template string OR dictionary with 'instruction' and 'template' keys
            values: Dictionary of field names to values
            
        Returns:
            Formatted string
        """
        # Handle both old string format and new dictionary format
        if isinstance(template, dict):
            if 'instruction' in template and 'template' in template:
                # New format: process instruction first, then template
                instruction_text = template['instruction']
                template_text = template['template']
                
                # First, format the instruction with data values
                formatted_instruction = instruction_text
                for field_name, field_value in values.items():
                    placeholder = f'{{{field_name}}}'
                    if placeholder in formatted_instruction:
                        formatted_instruction = formatted_instruction.replace(placeholder, str(field_value))
                
                # Add the formatted instruction to values
                values_with_instruction = values.copy()
                values_with_instruction['instruction'] = formatted_instruction
                
                # Now format the template part
                template_to_format = template_text
            elif 'combined' in template:
                # Fallback for combined format
                template_to_format = template['combined']
                values_with_instruction = values
            else:
                # Unknown dictionary format
                template_to_format = str(template)
                values_with_instruction = values
        else:
            # Old format: simple string
            template_to_format = template
            values_with_instruction = values
        
        # Create a copy of values with field names only (remove variation types and few-shot syntax)
        format_values = {}
        
        for field_match in self.FIELD_PATTERN.findall(template_to_format):
            # For few-shot fields, extract just the field name
            few_shot_match = self.FEW_SHOT_PATTERN.match(field_match)
            if few_shot_match:
                field_name = few_shot_match.group('field')
                if field_name in values_with_instruction:
                    format_values[field_match] = values_with_instruction[field_name]
            else:
                # Regular field
                field_name = field_match.split(':')[0].strip()
                if field_name in values_with_instruction:
                    format_values[field_match] = values_with_instruction[field_name]
        
        # Replace fields in template
        formatted = template_to_format
        for field_match in self.FIELD_PATTERN.findall(template_to_format):
            if field_match in format_values:
                formatted = formatted.replace(
                    f'{{{field_match}}}', 
                    str(format_values[field_match])
                )
        
        return formatted 