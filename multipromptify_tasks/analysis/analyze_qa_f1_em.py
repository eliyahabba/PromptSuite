#!/usr/bin/env python3
"""
Question Answering F1 & Exact Match Analysis Script
Analyzes QA results specifically for word-level F1 and exact match metrics.
This should be run after evaluate_qa_f1.py has added the metrics to the results.

Example usage:
python analyze_qa_f1_em.py gpt_4o_mini                      # Analyze specific model
python analyze_qa_f1_em.py gpt_4o_mini --metric exact_match # Only exact match analysis
python analyze_qa_f1_em.py gpt_4o_mini --metric word_f1     # Only word F1 analysis
python analyze_qa_f1_em.py gpt_4o_mini --multi              # All metrics in subplots
"""

import argparse
from pathlib import Path
import sys

# Add the current directory to the path to import shared_analysis
sys.path.append(str(Path(__file__).parent))
from shared_analysis import analyze_task_variations, analyze_multiple_metrics


def main():
    parser = argparse.ArgumentParser(description='Analyze QA F1 and exact match variation performance')
    parser.add_argument('model', help='Model directory name (e.g., gpt_4o_mini)')
    parser.add_argument('--metric', '-m', 
                       choices=['exact_match', 'word_f1', 'word_precision', 'word_recall', 'all'],
                       default='exact_match',
                       help='Metric to analyze (default: exact_match)')
    parser.add_argument('--multi', action='store_true',
                       help='Analyze all available QA F1 and exact match metrics in subplots')
    args = parser.parse_args()
    
    base_dir = Path(__file__).parent.parent / "tasks_data" / "results" / "qa"
    model_dir = base_dir / args.model
    
    if not model_dir.exists():
        print(f"Model directory not found: {model_dir}")
        return
    
    # Define all QA F1 and exact match metrics
    qa_metrics = ['exact_match', 'word_f1', 'word_precision', 'word_recall']
    
    # Look for files with F1 metrics first
    file_pattern = "*.csv"
    f1_files = list(model_dir.glob(file_pattern))
    
    if not f1_files:
        print(f"❌ No F1 metrics files found in {model_dir}")
        print(f"💡 Please run evaluate_qa_f1.py first to add F1 and exact match metrics")
        return
    
    if args.multi or args.metric == 'all':
        # Analyze all metrics in subplots
        analyze_multiple_metrics(
            model_dir=model_dir,
            task_type="question_answering",
            metrics=qa_metrics,
            file_pattern=file_pattern,
            combine_all_files=False
        )
    else:
        # Analyze single metric
        analyze_task_variations(
            model_dir=model_dir,
            task_type="question_answering",
            metric_name=args.metric,
            file_pattern=file_pattern,
            combine_all_files=False
        )


if __name__ == "__main__":
    main() 