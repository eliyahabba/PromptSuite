#!/usr/bin/env python3
"""
Sentiment Analysis Results Analyzer
Analyzes sentiment analysis variation performance using the shared analysis framework.
"""

import argparse
from pathlib import Path
import sys

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from multipromptify_tasks.analysis.shared_analysis import analyze_task_variations, analyze_multiple_metrics


def analyze_sentiment_variations(model_name: str = "gpt_4o_mini"):
    """
    Analyze sentiment analysis variation performance with multiple metrics.
    
    Args:
        model_name: Name of the model directory to analyze
    """
    # Define paths
    results_dir = Path(__file__).parent.parent /"tasks_data" / "results" / "sentiment"
    model_dir = results_dir / model_name
    
    if not model_dir.exists():
        print(f"❌ Model directory not found: {model_dir}")
        print(f"Available models in {results_dir}:")
        if results_dir.exists():
            for d in results_dir.iterdir():
                if d.is_dir():
                    print(f"  - {d.name}")
        return
    
    print(f"🔍 Analyzing sentiment analysis results for model: {model_name}")
    print(f"📁 Results directory: {model_dir}")
    print("=" * 80)
    
    # Analyze sentiment variations using multiple metrics
    metrics_to_analyze = ['mae', 'mse']
    
    print("📊 Analyzing multiple metrics for sentiment analysis...")
    analyze_multiple_metrics(
        model_dir=model_dir,
        task_type="sentiment",
        metrics=metrics_to_analyze,
        file_pattern="sentiment_*.csv",
        subject_column=None,  # No subject column for sentiment analysis
        combine_all_files=True  # Combine all sentiment datasets for analysis
    )
    
    print("\n" + "="*80)
    print("📈 Individual metric analysis:")
    print("="*80)
    
    # Also analyze each metric individually for detailed view
    for metric in metrics_to_analyze:
        print(f"\n🎯 Analyzing {metric.upper()} metric:")
        print("-" * 50)
        
        analyze_task_variations(
            model_dir=model_dir,
            task_type="sentiment",
            metric_name=metric,
            file_pattern="sentiment_*.csv",
            subject_column=None,
            combine_all_files=True
        )


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(description="Analyze sentiment analysis variation results")
    parser.add_argument("--model", default="gpt_4o_mini", 
                       help="Model name to analyze (default: gpt_4o_mini)")
    
    args = parser.parse_args()
    
    analyze_sentiment_variations(model_name=args.model)


if __name__ == "__main__":
    main() 