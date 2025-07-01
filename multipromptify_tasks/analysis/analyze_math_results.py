#!/usr/bin/env python3
"""
Math Problems Results Analysis
Analyzes results from math problem solving tasks to understand variation performance.
"""

import argparse
from pathlib import Path
import sys

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from multipromptify_tasks.analysis.shared_analysis import analyze_math_variations


def main():
    """Main function to analyze math problem solving results."""
    parser = argparse.ArgumentParser(description="Analyze math problem solving results")
    parser.add_argument("--model", type=str, required=True,
                        help="Model directory name (e.g., gpt_4o_mini)")
    parser.add_argument("--results_dir", type=str,
                        default=str(Path(__file__).parent.parent / "tasks_data" / "results" / "math"),
                        help="Path to results directory")
    
    args = parser.parse_args()
    
    # Validate model directory
    results_dir = Path(args.results_dir)
    model_dir = results_dir / args.model
    
    if not results_dir.exists():
        print(f"❌ Results directory not found: {results_dir}")
        return
    
    if not model_dir.exists():
        print(f"❌ Model directory not found: {model_dir}")
        print(f"Available models in {results_dir}:")
        for d in results_dir.iterdir():
            if d.is_dir():
                print(f"  - {d.name}")
        return
    
    # Check for CSV files
    csv_files = list(model_dir.glob("*.csv"))
    if not csv_files:
        print(f"❌ No CSV files found in {model_dir}")
        return
    
    print(f"🧮 Analyzing math problem solving results for model: {args.model}")
    print(f"📁 Results directory: {model_dir}")
    print(f"📊 Found {len(csv_files)} CSV files")
    print("=" * 60)
    
    # Run the analysis
    analyze_math_variations(model_dir)


if __name__ == "__main__":
    main() 