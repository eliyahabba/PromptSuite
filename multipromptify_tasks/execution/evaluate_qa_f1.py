#!/usr/bin/env python3
"""
Question Answering F1 Evaluation Script
Evaluates QA (SQuAD) question answering results and calculates word-level F1, precision, recall, and exact match metrics.
This should be run after generating all QA responses.

Example usage:
python evaluate_qa_f1.py                                   # Uses default model (gpt_4o_mini)
python evaluate_qa_f1.py --model llama_3_3_70b            # Uses specific model
python evaluate_qa_f1.py --results_file path/to/file.csv  # Uses specific file
python evaluate_qa_f1.py --results_dir path/to/directory  # Uses specific directory
"""

import argparse
import json
import os
import re
import string
from pathlib import Path
from typing import Dict, List, Any, Set
from collections import Counter
import pandas as pd


def normalize_answer(text: str) -> str:
    """Normalize answer text for comparison."""
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text


def get_word_tokens(text: str) -> List[str]:
    """Get word tokens from text."""
    normalized = normalize_answer(text)
    if not normalized:
        return []
    return normalized.split()


def calculate_exact_match(prediction: str, gold: str) -> bool:
    """
    Calculate exact match between prediction and gold answer.
    
    Args:
        prediction: Model's predicted answer
        gold: Gold standard answer
        
    Returns:
        Boolean indicating exact match
    """
    pred_normalized = normalize_answer(prediction)
    gold_normalized = normalize_answer(gold)
    
    return pred_normalized == gold_normalized


def calculate_word_f1(prediction: str, gold: str) -> Dict[str, float]:
    """
    Calculate word-level F1, precision, and recall between prediction and gold answer.
    
    Args:
        prediction: Model's predicted answer
        gold: Gold standard answer
        
    Returns:
        Dictionary with f1, precision, and recall scores
    """
    pred_tokens = get_word_tokens(prediction)
    gold_tokens = get_word_tokens(gold)
    
    if not pred_tokens and not gold_tokens:
        return {"word_f1": 1.0, "word_precision": 1.0, "word_recall": 1.0}
    
    if not pred_tokens:
        return {"word_f1": 0.0, "word_precision": 0.0, "word_recall": 0.0}
    
    if not gold_tokens:
        return {"word_f1": 0.0, "word_precision": 0.0, "word_recall": 0.0}
    
    # Count word overlaps
    pred_counter = Counter(pred_tokens)
    gold_counter = Counter(gold_tokens)
    
    # Calculate intersection (common words with minimum count)
    intersection = sum((pred_counter & gold_counter).values())
    
    # Calculate precision and recall
    precision = intersection / len(pred_tokens) if pred_tokens else 0.0
    recall = intersection / len(gold_tokens) if gold_tokens else 0.0
    
    # Calculate F1
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    
    return {
        "word_f1": f1,
        "word_precision": precision,
        "word_recall": recall
    }


def load_qa_results(results_file: str) -> List[Dict[str, Any]]:
    """Load QA results from CSV file."""
    try:
        df = pd.read_csv(results_file, encoding='utf-8')
        results = df.to_dict('records')
        print(f"✅ Loaded {len(results)} results from {results_file}")
        return results
    except Exception as e:
        print(f"❌ Error loading results: {e}")
        return []


def evaluate_qa_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Evaluate QA results and add F1 and exact match metrics.
    
    Args:
        results: List of result dictionaries
        
    Returns:
        List of results with added F1 and exact match metrics
    """
    print(f"🔄 Evaluating {len(results)} QA results...")
    
    evaluated_results = []
    
    for i, result in enumerate(results):
        if i % 100 == 0:
            progress_pct = (i / len(results)) * 100
            print(f"   ⏳ Progress: {i}/{len(results)} ({progress_pct:.1f}%)")
        
        # Get model response and gold answer
        model_response = result.get('model_response', '')
        gold_answer = result.get('gold_answer', '')
        
        # Calculate exact match
        exact_match = calculate_exact_match(model_response, gold_answer)
        
        # Calculate word-level F1 metrics
        f1_metrics = calculate_word_f1(model_response, gold_answer)
        
        # Create new result with all metrics
        new_result = result.copy()
        new_result['exact_match'] = exact_match
        new_result.update(f1_metrics)
        
        evaluated_results.append(new_result)
    
    print(f"✅ Completed evaluation of {len(evaluated_results)} results")
    return evaluated_results


def save_results_to_csv(results: List[Dict[str, Any]], output_file: str):
    """Save results to CSV file."""
    try:
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"💾 Results saved to CSV: {output_file}")
    except Exception as e:
        print(f"❌ Error saving CSV: {e}")


def save_results_to_json(results: List[Dict[str, Any]], output_file: str):
    """Save results to JSON file."""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"💾 Results saved to JSON: {output_file}")
    except Exception as e:
        print(f"❌ Error saving JSON: {e}")


def calculate_overall_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calculate overall metrics across all results."""
    if not results:
        return {}
    
    # Calculate averages
    exact_match_scores = [r.get('exact_match', False) for r in results]
    word_f1_scores = [r.get('word_f1', 0.0) for r in results]
    word_precision_scores = [r.get('word_precision', 0.0) for r in results]
    word_recall_scores = [r.get('word_recall', 0.0) for r in results]
    
    overall_metrics = {
        'avg_exact_match': sum(exact_match_scores) / len(exact_match_scores),
        'avg_word_f1': sum(word_f1_scores) / len(word_f1_scores),
        'avg_word_precision': sum(word_precision_scores) / len(word_precision_scores),
        'avg_word_recall': sum(word_recall_scores) / len(word_recall_scores),
        'total_samples': len(results)
    }
    
    return overall_metrics


def print_evaluation_summary(overall_metrics: Dict[str, float]):
    """Print evaluation summary."""
    print(f"\n📊 QA F1 & Exact Match Evaluation Results:")
    print(f"   Total samples evaluated: {overall_metrics.get('total_samples', 0)}")
    print(f"   🎯 Average Exact Match: {overall_metrics.get('avg_exact_match', 0.0):.4f} ({overall_metrics.get('avg_exact_match', 0.0)*100:.2f}%)")
    print(f"   🎯 Average Word F1: {overall_metrics.get('avg_word_f1', 0.0):.4f} ({overall_metrics.get('avg_word_f1', 0.0)*100:.2f}%)")
    print(f"   🎯 Average Word Precision: {overall_metrics.get('avg_word_precision', 0.0):.4f} ({overall_metrics.get('avg_word_precision', 0.0)*100:.2f}%)")
    print(f"   🎯 Average Word Recall: {overall_metrics.get('avg_word_recall', 0.0):.4f} ({overall_metrics.get('avg_word_recall', 0.0)*100:.2f}%)")


def main():
    """Main function for QA F1 and exact match evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate QA results with word-level F1 and exact match metrics")
    
    parser.add_argument("--model", type=str, default="gpt_4o_mini",
                        help="Model name for results directory (default: gpt_4o_mini)")
    parser.add_argument("--results_file", type=str,
                        help="Path to specific results CSV file (overrides model-based path)")
    parser.add_argument("--results_dir", type=str,
                        help="Directory containing results CSV files (overrides model-based path)")

    
    args = parser.parse_args()
    
    # Get the current script directory and build relative paths
    script_dir = Path(__file__).parent
    tasks_data_dir = script_dir.parent / "tasks_data"
    
    # Build model-specific paths
    model_results_dir = tasks_data_dir / "results" / "qa" / args.model
    default_results_file = model_results_dir / "question_answering_squad_variations.csv"
    
    # Find results files
    results_files = []
    
    if args.results_file:
        # User specified a custom file
        results_file_path = Path(args.results_file)
        if results_file_path.exists():
            results_files.append(results_file_path)
            print(f"🎯 Using specified results file: {results_file_path}")
        else:
            print(f"❌ Results file not found: {results_file_path}")
            return
    elif args.results_dir:
        # User specified a custom directory
        results_dir = Path(args.results_dir)
        if results_dir.exists():
            results_files = list(results_dir.glob("*.csv"))
            if not results_files:
                print(f"❌ No CSV files found in: {results_dir}")
                return
            print(f"🎯 Using specified results directory: {results_dir}")
        else:
            print(f"❌ Results directory not found: {results_dir}")
            return
    else:
        # Use model-based default paths
        if default_results_file.exists():
            results_files.append(default_results_file)
            print(f"🎯 Using default results file for model '{args.model}': {default_results_file}")
        elif model_results_dir.exists():
            results_files = list(model_results_dir.glob("*.csv"))
            if not results_files:
                print(f"❌ No CSV files found in model directory: {model_results_dir}")
                return
            print(f"🎯 Using model directory '{args.model}': {model_results_dir}")
        else:
            print(f"❌ Model results directory not found: {model_results_dir}")
            print(f"💡 Available models in results directory:")
            qa_results_dir = tasks_data_dir / "results" / "qa"
            if qa_results_dir.exists():
                for model_dir in qa_results_dir.iterdir():
                    if model_dir.is_dir():
                        print(f"   - {model_dir.name}")
            return
    
    print(f"🔍 Found {len(results_files)} results files to evaluate")
    
    # Process each results file
    for results_file in results_files:
        print(f"\n📂 Processing: {results_file.name}")
        
        # Load results
        results = load_qa_results(str(results_file))
        if not results:
            continue
        
        # Evaluate results
        evaluated_results = evaluate_qa_results(results)
        
        # Calculate overall metrics
        overall_metrics = calculate_overall_metrics(evaluated_results)
        
        # Save results back to original files (overwrite)
        base_name = str(results_file).replace('.csv', '')
        
        # Save JSON (overwrite original JSON file)
        json_output_file = f"{base_name}.json"
        save_results_to_json(evaluated_results, json_output_file)
        
        # Save CSV (overwrite original CSV file)
        save_results_to_csv(evaluated_results, str(results_file))
        
        # Print summary
        print_evaluation_summary(overall_metrics)


if __name__ == "__main__":
    main() 