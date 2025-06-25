#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import sys

def analyze_mmlu_variations(model_dir):
    all_data = []
    
    for csv_file in model_dir.glob("*.csv"):
        df = pd.read_csv(csv_file)
        df['subject'] = csv_file.stem
        all_data.append(df)
    
    if not all_data:
        print(f"No CSV files found in {model_dir}")
        return
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    question_counts = combined_df.groupby(['subject', 'original_row_index']).size().reset_index(name='count')
    total_variations = combined_df['variation_index'].nunique()
    
    shared_questions = question_counts[question_counts['count'] == total_variations]
    shared_questions = shared_questions[['subject', 'original_row_index']]
    
    filtered_df = pd.merge(combined_df, shared_questions, on=['subject', 'original_row_index'], how='inner')
    
    variation_scores = filtered_df.groupby('variation_index')['is_correct'].agg(['mean', 'count']).reset_index()
    variation_scores.columns = ['variation_index', 'average_score', 'question_count']
    
    total_questions = len(shared_questions)
    
    plt.figure(figsize=(12, 6))
    plt.scatter(variation_scores['variation_index'], variation_scores['average_score'], 
                s=50, alpha=0.7)
    
    plt.xlabel('Variation Index')
    plt.ylabel('Average Score')
    plt.title(f'MMLU Performance by Variation Index\nModel: {model_dir.name}\nTotal Questions: {total_questions}')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(model_dir / 'variation_performance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Total variations: {len(variation_scores)}")
    print(f"Total shared questions: {total_questions}")
    print(f"Average score across all variations: {variation_scores['average_score'].mean():.3f}")
    print(f"Best variation: {variation_scores.loc[variation_scores['average_score'].idxmax(), 'variation_index']} "
          f"(score: {variation_scores['average_score'].max():.3f})")
    print(f"Worst variation: {variation_scores.loc[variation_scores['average_score'].idxmin(), 'variation_index']} "
          f"(score: {variation_scores['average_score'].min():.3f})")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help='Model directory name (e.g., gpt_4o_mini)')
    args = parser.parse_args()
    
    base_dir = Path(__file__).parent / "results" / "mmlu"
    model_dir = base_dir / args.model
    
    if not model_dir.exists():
        print(f"Model directory not found: {model_dir}")
        return
    
    analyze_mmlu_variations(model_dir)

if __name__ == "__main__":
    main() 