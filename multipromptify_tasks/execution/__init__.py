"""
Execution Module
Contains scripts for running language models on generated variations and computing metrics.
"""

from .shared_metrics import (
    calculate_summarization_metrics, 
    calculate_text_generation_metrics,
    calculate_translation_correctness_and_metrics,
    calculate_mmlu_correctness_and_metrics,
    calculate_sentiment_correctness_and_metrics,
    calculate_bertscore_metrics
)
from .add_metrics_to_csv import add_bertscore_to_csv, add_metric_to_csv

__all__ = [
    'calculate_summarization_metrics', 
    'calculate_text_generation_metrics',
    'calculate_translation_correctness_and_metrics',
    'calculate_mmlu_correctness_and_metrics',
    'calculate_sentiment_correctness_and_metrics',
    'calculate_bertscore_metrics',
    'add_bertscore_to_csv', 
    'add_metric_to_csv'
] 