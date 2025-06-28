#!/usr/bin/env python3

import argparse
from pathlib import Path
import sys

# Add the current directory to the path to import shared_analysis
sys.path.append(str(Path(__file__).parent))
from .shared_analysis import analyze_task_variations, analyze_multiple_metrics

def main():
    parser = argparse.ArgumentParser(description='Analyze summarization results')
    parser.add_argument('results_file', help='Path to summarization results CSV file')
    parser.add_argument('--metric', '-m', 
                       choices=['bleu', 'rouge1', 'rouge2', 'rougeL', 'sacrebleu', 'all'],
                       default='bleu',
                       help='Metric to analyze (default: bleu)')
    parser.add_argument('--multi', action='store_true',
                       help='Analyze all available summarization metrics in subplots')
    parser.add_argument('--gold_field', default='highlights',
                       help='Field name inside gold_updates for the gold summary (default: highlights)')
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent / "results"
    results_file = base_dir / args.results_file
    if not results_file.exists():
        print(f"Results file not found: {results_file}")
        return

    # Use the parent directory as the 'model_dir' for compatibility
    model_dir = results_file.parent

    # Define all summarization metrics
    summarization_metrics = ['bleu', 'rouge1', 'rouge2', 'rougeL', 'sacrebleu', 'bertscore']

    if args.multi or args.metric == 'all':
        analyze_multiple_metrics(
            model_dir=model_dir,
            task_type="summarization",
            metrics=summarization_metrics,
            file_pattern=results_file.name,  # Only this file
            combine_all_files=True
        )
    else:
        analyze_task_variations(
            model_dir=model_dir,
            task_type="summarization",
            metric_name=args.metric,
            file_pattern=results_file.name,  # Only this file
            combine_all_files=True
        )

if __name__ == "__main__":
    main() 