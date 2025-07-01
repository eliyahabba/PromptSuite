#!/usr/bin/env python3
"""
Add Metrics to CSV
General script for adding new metrics to existing result CSV files.
"""

import pandas as pd
import argparse
from pathlib import Path
from typing import Callable, Any
import sys

# Add current directory to path for local imports
current_dir = Path(__file__).parent.parent
sys.path.insert(0, str(current_dir))

from .shared_metrics import calculate_bertscore_metrics


def add_bertscore_to_csv(csv_path: Path, prediction_col: str = 'model_response', 
                        reference_col: str = 'gold_answer', lang: str = "en") -> None:
    """
    Add BERTScore column to an existing CSV file.
    
    Args:
        csv_path: Path to the CSV file
        prediction_col: Column name containing model predictions
        reference_col: Column name containing reference texts
        lang: Language code for BERTScore
    """
    print(f"Loading: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Check required columns
    if prediction_col not in df.columns or reference_col not in df.columns:
        raise ValueError(f"CSV must contain '{prediction_col}' and '{reference_col}' columns")
    
    # Handle empty values
    predictions = df[prediction_col].astype(str).fillna("").tolist()
    references = df[reference_col].astype(str).fillna("").tolist()
    
    print("Calculating BERTScore for all rows...")
    bertscore_f1 = calculate_bertscore_metrics(predictions, references, lang)
    
    # Add the BERTScore column
    df['bertscore'] = bertscore_f1
    
    # Overwrite the original CSV
    print(f"Updating original file with BERTScore column: {csv_path}")
    df.to_csv(csv_path, index=False)
    print("Done!")


def add_metric_to_csv(csv_path: Path, metric_function: Callable, metric_name: str, 
                     prediction_col: str = 'model_response', reference_col: str = 'gold_answer',
                     **kwargs) -> None:
    """
    General function to add any metric to a CSV file.
    
    Args:
        csv_path: Path to the CSV file
        metric_function: Function that calculates the metric
        metric_name: Name of the new metric column
        prediction_col: Column name containing model predictions
        reference_col: Column name containing reference texts
        **kwargs: Additional arguments for the metric function
    """
    print(f"Loading: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Check required columns
    if prediction_col not in df.columns or reference_col not in df.columns:
        raise ValueError(f"CSV must contain '{prediction_col}' and '{reference_col}' columns")
    
    # Handle empty values
    predictions = df[prediction_col].astype(str).fillna("").tolist()
    references = df[reference_col].astype(str).fillna("").tolist()
    
    print(f"Calculating {metric_name} for all rows...")
    metric_scores = metric_function(predictions, references, **kwargs)
    
    # Add the metric column
    df[metric_name] = metric_scores
    
    # Overwrite the original CSV
    print(f"Updating original file with {metric_name} column: {csv_path}")
    df.to_csv(csv_path, index=False)
    print("Done!")


def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(description="Add metrics to existing CSV result files")
    csv_path = Path("results/gpt_4o_mini/summarization_cnn_dailymail_variations.csv")
    parser.add_argument("csv_path", help="Path to the CSV file", default=csv_path, type=str)
    parser.add_argument("--metric", choices=["bertscore"], default="bertscore",
                       help="Metric to add (default: bertscore)")
    parser.add_argument("--prediction_col", default="model_response",
                       help="Column name containing model predictions (default: model_response)")
    parser.add_argument("--reference_col", default="gold_answer",
                       help="Column name containing reference texts (default: gold_answer)")
    parser.add_argument("--lang", default="en",
                       help="Language code for BERTScore (default: en)")
    
    args = parser.parse_args()
    
    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}")
        return
    
    if args.metric == "bertscore":
        add_bertscore_to_csv(
            csv_path=csv_path,
            prediction_col=args.prediction_col,
            reference_col=args.reference_col,
            lang=args.lang
        )
    else:
        print(f"Error: Unsupported metric: {args.metric}")


if __name__ == "__main__":
    main() 