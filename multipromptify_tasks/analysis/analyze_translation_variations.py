#!/usr/bin/env python3

import argparse
from pathlib import Path
import sys

# Add the current directory to the path to import shared_analysis
sys.path.append(str(Path(__file__).parent))
from shared_analysis import analyze_task_variations, analyze_multiple_metrics


def main():
    parser = argparse.ArgumentParser(description='Analyze translation variation performance')
    parser.add_argument('model', help='Model directory name (e.g., gpt_4o_mini)')
    parser.add_argument('--metric', '-m', 
                       choices=['bleu', 'rouge1', 'rouge2', 'rougeL', 'sacrebleu', 'all'],
                       default='bleu',
                       help='Metric to analyze (default: bleu)')
    parser.add_argument('--multi', action='store_true',
                       help='Analyze all available translation metrics in subplots')
    args = parser.parse_args()
    
    base_dir = Path(__file__).parent.parent / "tasks_data" / "results" / "translation"
    model_dir = base_dir / args.model
    
    if not model_dir.exists():
        print(f"Model directory not found: {model_dir}")
        return
    
    # Define all translation metrics
    translation_metrics = ['bleu', 'rouge1', 'rouge2', 'rougeL', 'sacrebleu']
    
    if args.multi or args.metric == 'all':
        # Analyze all metrics in subplots, combining all translation files
        analyze_multiple_metrics(
            model_dir=model_dir,
            task_type="translation",
            metrics=translation_metrics,
            file_pattern="*.csv",
            combine_all_files=True
        )
    else:
        # Analyze single metric, combining all translation files
        analyze_task_variations(
            model_dir=model_dir,
            task_type="translation",
            metric_name=args.metric,
            file_pattern="*.csv",
            combine_all_files=True
        )


if __name__ == "__main__":
    main() 