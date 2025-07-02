#!/usr/bin/env python3
"""
Code Generation Evaluation Analysis
Analyze code generation evaluation results (pass@k metrics) and create visualizations.
This script analyzes the output from evaluate_code_generation.py
"""

import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys
from typing import Dict, List, Any

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def load_evaluation_results(evaluation_file: Path) -> Dict[str, Any]:
    """Load evaluation results from JSON file."""
    try:
        with open(evaluation_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Error loading evaluation file {evaluation_file}: {e}")
        return {}


def extract_variation_data_from_sample_key(sample_key: str) -> tuple:
    """Extract row_index and variation_index from sample key like '0_1'."""
    parts = sample_key.split('_')
    if len(parts) >= 2:
        return int(parts[0]), int(parts[1])
    return 0, 0


def create_variation_analysis_dataframe(evaluation_results: Dict[str, Any]) -> pd.DataFrame:
    """Convert evaluation results to DataFrame for analysis by variation."""
    sample_results = evaluation_results.get('sample_results', {})
    
    rows = []
    for sample_key, metrics in sample_results.items():
        row_idx, var_idx = extract_variation_data_from_sample_key(sample_key)
        
        row_data = {
            'sample_key': sample_key,
            'original_row_index': row_idx,
            'variation_index': var_idx,
        }
        
        # Add all pass@k metrics
        for metric_name, value in metrics.items():
            if metric_name.startswith('pass@'):
                row_data[metric_name] = value
        
        rows.append(row_data)
    
    return pd.DataFrame(rows)


def analyze_pass_at_k_by_variation(df: pd.DataFrame, model_name: str) -> None:
    """Create separate plots for each pass@k metric showing variation performance."""
    # Find all pass@k columns
    pass_columns = [col for col in df.columns if col.startswith('pass@')]
    
    if not pass_columns:
        print("❌ No pass@k metrics found in the data")
        return
    
    print(f"📊 Found pass@k metrics: {pass_columns}")
    
    # Create subplots - 2x2 grid for up to 4 metrics
    n_metrics = len(pass_columns)
    n_cols = 2
    n_rows = (n_metrics + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 6*n_rows))
    if n_metrics == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes if isinstance(axes, np.ndarray) else [axes]
    else:
        axes = axes.flatten()
    
    # Colors for different variations
    colors = plt.cm.Set1(np.linspace(0, 1, df['variation_index'].nunique()))
    
    for i, metric_col in enumerate(pass_columns):
        ax = axes[i]
        
        # Calculate average performance by variation
        variation_stats = df.groupby('variation_index')[metric_col].agg(['mean', 'std', 'count']).reset_index()
        variation_stats.columns = ['variation_index', 'mean', 'std', 'count']
        
        # Create scatter plot with error bars
        x = variation_stats['variation_index']
        y = variation_stats['mean']
        yerr = variation_stats['std']
        
        # Scatter plot
        scatter = ax.scatter(x, y, c=colors[:len(x)], s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
        
        # Error bars (if we have multiple samples per variation)
        mask = variation_stats['count'] > 1
        if mask.any():
            ax.errorbar(x[mask], y[mask], yerr=yerr[mask], fmt='none', 
                       capsize=3, capthick=1, ecolor='gray', alpha=0.6)
        
        # Formatting
        ax.set_xlabel('Variation Index')
        ax.set_ylabel(f'{metric_col} Score')
        ax.set_title(f'{metric_col.upper()} Performance by Variation')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
        
        # Add trend line
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        ax.plot(x, p(x), "r--", alpha=0.8, linewidth=1)
        
        # Add statistics text
        best_var = variation_stats.loc[variation_stats['mean'].idxmax(), 'variation_index']
        worst_var = variation_stats.loc[variation_stats['mean'].idxmin(), 'variation_index']
        best_score = variation_stats['mean'].max()
        worst_score = variation_stats['mean'].min()
        
        stats_text = f'Best: Var {best_var} ({best_score:.3f})\nWorst: Var {worst_var} ({worst_score:.3f})'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Hide empty subplots
    for i in range(n_metrics, len(axes)):
        axes[i].set_visible(False)
    
    # Overall title
    total_samples = len(df)
    total_variations = df['variation_index'].nunique()
    plt.suptitle(f'Code Generation Pass@k Analysis by Variation\n'
                f'Model: {model_name} | {total_samples} samples across {total_variations} variations', 
                fontsize=14)
    plt.tight_layout()
    
    # Save the plot
    figures_dir = Path(__file__).parent.parent / "tasks_data" / "figures" / "code_generation"
    figures_dir.mkdir(parents=True, exist_ok=True)
    output_filename = f'{model_name}_code_generation_pass_at_k_variation_analysis.png'
    plt.savefig(figures_dir / output_filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print detailed statistics
    print(f"\n=== CODE GENERATION PASS@K ANALYSIS ===")
    print(f"Model: {model_name}")
    print(f"Total samples: {total_samples}")
    print(f"Total variations: {total_variations}")
    
    for metric_col in pass_columns:
        print(f"\n{metric_col.upper()} Statistics:")
        variation_stats = df.groupby('variation_index')[metric_col].agg(['mean', 'std', 'count']).reset_index()
        
        overall_mean = df[metric_col].mean()
        best_variation = variation_stats.loc[variation_stats['mean'].idxmax()]
        worst_variation = variation_stats.loc[variation_stats['mean'].idxmin()]
        
        print(f"  Overall average: {overall_mean:.4f}")
        print(f"  Best variation: {int(best_variation['variation_index'])} (score: {best_variation['mean']:.4f})")
        print(f"  Worst variation: {int(worst_variation['variation_index'])} (score: {worst_variation['mean']:.4f})")
        print(f"  Performance range: {best_variation['mean'] - worst_variation['mean']:.4f}")
        
        # Show top 3 and bottom 3 variations
        sorted_variations = variation_stats.sort_values('mean', ascending=False)
        print(f"  Top 3 variations:")
        for _, row in sorted_variations.head(3).iterrows():
            print(f"    Variation {int(row['variation_index'])}: {row['mean']:.4f} ± {row['std']:.4f}")
        
        print(f"  Bottom 3 variations:")
        for _, row in sorted_variations.tail(3).iterrows():
            print(f"    Variation {int(row['variation_index'])}: {row['mean']:.4f} ± {row['std']:.4f}")


def analyze_evaluation_results(model_dir: Path) -> None:
    """Analyze all evaluation results in the model directory."""
    # Find evaluation files
    evaluation_files = list(model_dir.glob("*_evaluation.json"))
    
    if not evaluation_files:
        print(f"❌ No evaluation files (*_evaluation.json) found in {model_dir}")
        print("Make sure to run evaluate_code_generation.py first")
        return
    
    print(f"📂 Found {len(evaluation_files)} evaluation files:")
    for f in evaluation_files:
        print(f"  - {f.name}")
    
    # Combine all evaluation results
    all_samples = []
    
    for eval_file in evaluation_files:
        print(f"\n📊 Processing: {eval_file.name}")
        
        eval_results = load_evaluation_results(eval_file)
        if not eval_results:
            continue
        
        # Convert to DataFrame
        df = create_variation_analysis_dataframe(eval_results)
        if df.empty:
            print(f"⚠️ No data found in {eval_file.name}")
            continue
        
        # Add dataset identifier
        dataset_name = eval_file.stem.replace('_evaluation', '')
        df['dataset'] = dataset_name
        
        all_samples.append(df)
        
        print(f"  ✅ Loaded {len(df)} samples from {dataset_name}")
    
    if not all_samples:
        print("❌ No valid evaluation data found")
        return
    
    # Combine all datasets
    combined_df = pd.concat(all_samples, ignore_index=True)
    print(f"\n📊 Combined analysis: {len(combined_df)} total samples")
    
    # Analyze pass@k by variation
    model_name = model_dir.name
    analyze_pass_at_k_by_variation(combined_df, model_name)


def main():
    """Main function to analyze code generation evaluation results."""
    parser = argparse.ArgumentParser(description="Analyze code generation evaluation results (pass@k metrics)")
    parser.add_argument("--model", default="gpt_4o_mini", 
                       help="Model directory name (e.g., gpt_4o_mini, llama_3_3_70b)")
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
    
    print(f"📊 Analyzing code generation evaluation results for model: {args.model}")
    print(f"📁 Results directory: {model_dir}")
    
    analyze_evaluation_results(model_dir)


if __name__ == "__main__":
    main() 