#!/usr/bin/env python3
"""
Code Generation Results Analysis
Analyze code generation task results and create visualizations.
"""

import argparse
from pathlib import Path
import sys

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from promptsuite_tasks.analysis.shared_analysis import (
    analyze_code_generation_variations, 
    analyze_code_generation_multiple_metrics,
    analyze_code_generation_functional_correctness,
    analyze_code_generation_pass_at_1
)


def main():
    """Main function to analyze code generation results."""
    parser = argparse.ArgumentParser(description="Analyze code generation task results")
    parser.add_argument("--model", default="gpt_4o_mini", 
                       help="Model directory name (e.g., gpt_4o_mini, llama_3_3_70b)")
    parser.add_argument("--metric", choices=["is_correct", "functionally_correct", "syntactically_correct", "pass_at_1", "multi"], 
                       default="functionally_correct",
                       help="Metric to analyze (default: functionally_correct)")
    parser.add_argument("--results_dir", 
                       default=str(Path(__file__).parent.parent / "tasks_data" / "results" / "code_generation"),
                       help="Directory containing code generation results")
    
    args = parser.parse_args()
    
    # Get model directory
    results_dir = Path(args.results_dir)
    model_dir = results_dir / args.model
    
    if not model_dir.exists():
        print(f"❌ Model directory not found: {model_dir}")
        print(f"Available models in {results_dir}:")
        if results_dir.exists():
            for d in results_dir.iterdir():
                if d.is_dir():
                    print(f"  - {d.name}")
        return
    
    print(f"📊 Analyzing code generation results for model: {args.model}")
    print(f"📁 Results directory: {model_dir}")
    
    if args.metric == "multi":
        print("📈 Analyzing multiple metrics...")
        analyze_code_generation_multiple_metrics(model_dir)
    elif args.metric == "functionally_correct":
        print("📈 Analyzing functional correctness...")
        analyze_code_generation_functional_correctness(model_dir)
    elif args.metric == "pass_at_1":
        print("📈 Analyzing pass@1 scores...")
        analyze_code_generation_pass_at_1(model_dir)
    elif args.metric == "syntactically_correct":
        print("📈 Analyzing syntactic correctness...")
        # Use the general analyze function with syntactically_correct metric
        from promptsuite_tasks.analysis.shared_analysis import analyze_task_variations
        analyze_task_variations(
            model_dir=model_dir,
            task_type="code_generation",
            metric_name="syntactically_correct",
            file_pattern="*.csv",
            subject_column=None,
            combine_all_files=True
        )
    else:
        print(f"📈 Analyzing metric: {args.metric}")
        analyze_code_generation_variations(model_dir)


if __name__ == "__main__":
    main() 