"""
MultiPromptify: A tool that creates multi-prompt datasets from single-prompt datasets using templates.
"""

__version__ = "2.0.0"

# Import main classes for easier access
from .engine import MultiPromptify
from .api import MultiPromptifyAPI
from .template_parser import TemplateParser

__all__ = ["MultiPromptify", "MultiPromptifyAPI", "TemplateParser"]