"""
MultiPromptify - A tool for creating multi-prompt datasets from single-prompt datasets.
"""

from .multipromptify import MultiPromptify
from .template_parser import TemplateParser

__version__ = "2.0.0"
__author__ = "MultiPromptify Team"

__all__ = ["MultiPromptify", "TemplateParser"] 