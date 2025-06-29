"""
Data Generation Module
Contains task classes for generating prompt variations for different NLP tasks.
"""

from .base_task import BaseTask
from .mmlu_task import MMLUTask
from .translation_task import TranslationTask
from .qa_task import QATask
from .summarization_task import SummarizationTask
from .sentiment_task import SentimentTask

__all__ = ['BaseTask', 'MMLUTask', 'TranslationTask', 'QATask', 'SummarizationTask', 'SentimentTask'] 