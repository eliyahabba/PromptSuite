"""
Analysis Module
Contains scripts for analyzing results and generating visualizations.
Note: Metric calculation functions have been moved to the execution module.
"""

from .shared_analysis import analyze_task_variations, analyze_multiple_metrics

__all__ = ['analyze_task_variations', 'analyze_multiple_metrics'] 