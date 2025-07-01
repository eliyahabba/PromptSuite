#!/usr/bin/env python3
"""
Question Answering Results Analysis
Analyzes QA task variations for exact match, word F1, and other metrics.
"""

import argparse
from pathlib import Path
from shared_analysis import analyze_qa_variations, analyze_qa_multiple_metrics, analyze_qa_word_f1


def main():
    """Main function to analyze QA results."""
    parser = argparse.ArgumentParser(description="Analyze Question Answering task results")
    parser.add_argument("--model_dir", required=True, 
                        help="Path to model results directory containing QA CSV files")
    parser.add_argument("--metric", choices=["accuracy", "word_f1", "all"], default="all",
                        help="Which metric to analyze (default: all)")
    
    args = parser.parse_args()
    
    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        print(f"❌ Model directory not found: {model_dir}")
        return
    
    print(f"🔍 Analyzing Question Answering results in: {model_dir}")
    print("=" * 60)
    
    if args.metric == "accuracy":
        print("📊 Analyzing exact match accuracy...")
        analyze_qa_variations(model_dir)
    elif args.metric == "word_f1":
        print("📊 Analyzing word-level F1 scores...")
        analyze_qa_word_f1(model_dir)
    else:  # all
        print("📊 Analyzing all QA metrics...")
        analyze_qa_multiple_metrics(model_dir)


if __name__ == "__main__":
    main() 