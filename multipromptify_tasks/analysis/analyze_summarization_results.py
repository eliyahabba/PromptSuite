#!/usr/bin/env python3

import argparse
from pathlib import Path
import sys

# Add the current directory to the path to import shared_analysis
sys.path.append(str(Path(__file__).parent))
from shared_analysis import analyze_task_variations, analyze_multiple_metrics

def main():
    parser = argparse.ArgumentParser(description='Analyze summarization results')
    parser.add_argument('--metric', '-m',
                       choices=['bleu', 'rouge1', 'rouge2', 'rougeL', 'sacrebleu', 'all'],
                       default='bleu',
                       help='Metric to analyze (default: bleu)')
    parser.add_argument('--multi', action='store_true',
                       help='Analyze all available summarization metrics in subplots')
    parser.add_argument('--gold_field', default='highlights',
                       help='Field name inside gold_updates for the gold summary (default: highlights)')
    parser.add_argument("--model", default="gpt_4o_mini",
                       help="Model name to analyze (default: gpt_4o_mini)")
    args = parser.parse_args()

    results_dir = Path(__file__).parent.parent /"tasks_data" / "results" / "summarization"
    model_dir = results_dir / args.model
    if not results_dir.exists():
        print(f"Results file not found: {results_dir}")
        return

    # Use the parent directory as the 'model_dir' for compatibility

    # Define all summarization metrics
    summarization_metrics = ['bleu', 'rouge1', 'rouge2', 'rougeL', 'sacrebleu', 'bertscore']

    if args.multi or args.metric == 'all':
        analyze_multiple_metrics(
            model_dir=model_dir,
            task_type="summarization",
            metrics=summarization_metrics,
            file_pattern="summarization_*.csv",  # Assuming summarization files follow this pattern
            combine_all_files=True
        )
    else:
        analyze_task_variations(
            model_dir=model_dir,
            task_type="summarization",
            metric_name=args.metric,
            file_pattern="summarization_*.csv",  # Assuming summarization files follow this pattern
            combine_all_files=True
        )

if __name__ == "__main__":
    main() 