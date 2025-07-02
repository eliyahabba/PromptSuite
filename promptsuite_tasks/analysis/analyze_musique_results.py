#!/usr/bin/env python3
"""
MuSiQue Analysis Results Analyzer
Analyzes MuSiQue multi-hop question answering variation performance using the shared analysis framework.
"""

import argparse
from pathlib import Path
import sys

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from promptsuite_tasks.analysis.shared_analysis import analyze_task_variations, analyze_multiple_metrics


def analyze_musique_variations(model_name: str = "gpt_4o_mini"):
    """
    Analyze MuSiQue multi-hop question answering variation performance with multiple metrics.
    
    Args:
        model_name: Name of the model directory to analyze
    """
    # Define paths
    results_dir = Path(__file__).parent.parent / "tasks_data" / "results" / "musique"
    model_dir = results_dir / model_name
    
    if not model_dir.exists():
        print(f"❌ Model directory not found: {model_dir}")
        print(f"Available models in {results_dir}:")
        if results_dir.exists():
            for d in results_dir.iterdir():
                if d.is_dir():
                    print(f"  - {d.name}")
        return
    
    print(f"🔍 Analyzing MuSiQue multi-hop QA results for model: {model_name}")
    print(f"📁 Results directory: {model_dir}")
    print("=" * 80)
    
    # Analyze MuSiQue variations using multiple metrics
    metrics_to_analyze = ['is_correct', 'word_f1', 'word_precision', 'word_recall', 'bleu', 'rouge1', 'rouge2', 'rougeL']
    
    print("📊 Analyzing multiple metrics for MuSiQue multi-hop question answering...")
    analyze_multiple_metrics(
        model_dir=model_dir,
        task_type="musique",
        metrics=metrics_to_analyze,
        file_pattern="musique_*.csv",
        subject_column=None,  # No subject column for MuSiQue
        combine_all_files=True  # Combine all MuSiQue datasets for analysis
    )
    
    print("\n" + "="*80)
    print("📈 Individual metric analysis:")
    print("="*80)
    
    # Analyze key metrics individually for detailed view
    key_metrics = ['is_correct', 'word_f1', 'bleu', 'rouge1']
    
    for metric in key_metrics:
        print(f"\n🎯 Analyzing {metric.upper()} metric:")
        print("-" * 50)
        
        analyze_task_variations(
            model_dir=model_dir,
            task_type="musique",
            metric_name=metric,
            file_pattern="musique_*.csv",
            subject_column=None,
            combine_all_files=True
        )


def analyze_musique_exact_match(model_name: str = "gpt_4o_mini"):
    """
    Analyze MuSiQue exact match accuracy specifically.
    
    Args:
        model_name: Name of the model directory to analyze
    """
    results_dir = Path(__file__).parent.parent / "tasks_data" / "results" / "musique"
    model_dir = results_dir / model_name
    
    if not model_dir.exists():
        print(f"❌ Model directory not found: {model_dir}")
        return
    
    print(f"🎯 Analyzing MuSiQue exact match accuracy for model: {model_name}")
    print("=" * 60)
    
    analyze_task_variations(
        model_dir=model_dir,
        task_type="musique",
        metric_name="is_correct",
        file_pattern="musique_*.csv",
        subject_column=None,
        combine_all_files=True
    )


def analyze_musique_word_f1(model_name: str = "gpt_4o_mini"):
    """
    Analyze MuSiQue word-level F1 scores specifically.
    
    Args:
        model_name: Name of the model directory to analyze
    """
    results_dir = Path(__file__).parent.parent / "tasks_data" / "results" / "musique"
    model_dir = results_dir / model_name
    
    if not model_dir.exists():
        print(f"❌ Model directory not found: {model_dir}")
        return
    
    print(f"📊 Analyzing MuSiQue word-level F1 scores for model: {model_name}")
    print("=" * 60)
    
    analyze_task_variations(
        model_dir=model_dir,
        task_type="musique",
        metric_name="word_f1",
        file_pattern="musique_*.csv",
        subject_column=None,
        combine_all_files=True
    )


def analyze_musique_text_generation_metrics(model_name: str = "gpt_4o_mini"):
    """
    Analyze MuSiQue text generation metrics (BLEU, ROUGE).
    
    Args:
        model_name: Name of the model directory to analyze
    """
    results_dir = Path(__file__).parent.parent / "tasks_data" / "results" / "musique"
    model_dir = results_dir / model_name
    
    if not model_dir.exists():
        print(f"❌ Model directory not found: {model_dir}")
        return
    
    print(f"📝 Analyzing MuSiQue text generation metrics for model: {model_name}")
    print("=" * 60)
    
    # Analyze text generation metrics
    text_gen_metrics = ['bleu', 'rouge1', 'rouge2', 'rougeL']
    
    analyze_multiple_metrics(
        model_dir=model_dir,
        task_type="musique",
        metrics=text_gen_metrics,
        file_pattern="musique_*.csv",
        subject_column=None,
        combine_all_files=True
    )


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(description="Analyze MuSiQue multi-hop question answering variation results")
    parser.add_argument("--model", default="gpt_4o_mini", 
                       help="Model name to analyze (default: gpt_4o_mini)")
    parser.add_argument("--analysis-type", choices=["all", "exact-match", "word-f1", "text-gen"], 
                       default="all",
                       help="Type of analysis to perform (default: all)")
    
    args = parser.parse_args()
    
    if args.analysis_type == "all":
        analyze_musique_variations(model_name=args.model)
    elif args.analysis_type == "exact-match":
        analyze_musique_exact_match(model_name=args.model)
    elif args.analysis_type == "word-f1":
        analyze_musique_word_f1(model_name=args.model)
    elif args.analysis_type == "text-gen":
        analyze_musique_text_generation_metrics(model_name=args.model)


if __name__ == "__main__":
    main() 