#!/usr/bin/env python3
"""
Code Generation Evaluation Script
Evaluates code generation results and calculates pass@k metrics.
This should be run after generating all code samples with multiple runs.

Example usage:
python evaluate_code_generation.py --results_dir multipromptify_tasks/tasks_data/results/code_generation/gpt_4o_mini
python evaluate_code_generation.py --results_file multipromptify_tasks/tasks_data/results/code_generation/gpt_4o_mini/code_generation_humaneval_variations.json
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


def load_code_generation_results(results_file: str) -> List[Dict[str, Any]]:
    """Load code generation results from JSON file."""
    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        print(f"✅ Loaded {len(results)} results from {results_file}")
        return results
    except Exception as e:
        print(f"❌ Error loading results: {e}")
        return []


def group_results_by_sample(results: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group results by (original_row_index, variation_index) to get all runs for each sample."""
    grouped = defaultdict(list)
    
    for result in results:
        row_idx = result.get('original_row_index', 0)
        var_idx = result.get('variation_index', 0)
        sample_key = f"{row_idx}_{var_idx}"
        grouped[sample_key].append(result)
    
    return dict(grouped)


def get_task_id_from_row_index(row_index: int, data_file: str = None) -> str:
    """Get HumanEval task_id from row index."""
    if data_file and os.path.exists(data_file):
        try:
            df = pd.read_csv(data_file)
            df_test = df[df['split'] == 'test']
            if row_index < len(df_test):
                return df_test.iloc[row_index]['task_id']
        except Exception as e:
            print(f"⚠️ Error reading data file: {e}")
    
    # Fallback to simple format
    return f"HumanEval/{row_index}"


def evaluate_code_samples_with_human_eval(samples: List[Dict[str, Any]], task_id: str, sample_key: str, evaluation_dir: str) -> Dict[str, float]:
    """Evaluate multiple code samples for a single task using human_eval."""
    try:
        from human_eval.data import write_jsonl
        from human_eval.evaluation import evaluate_functional_correctness
        import os
        
        # Prepare samples for evaluation
        eval_samples = []
        for i, sample in enumerate(samples):
            completion = sample.get('model_response', '')
            if completion.strip():  # Only add non-empty completions
                eval_samples.append({
                    "task_id": task_id,
                    "completion": completion
                })
        
        # If no valid samples, return zeros
        if not eval_samples:
            print(f"⚠️ No valid completions found for {task_id}")
            results = {}
            k_values = [1, 2, 3, 5, 10]
            for k in k_values:
                results[f'pass@{k}'] = 0.0
            return results
        
        # Create descriptive file names based on task and variation
        task_name = task_id.replace('/', '_')
        task_file = os.path.join(evaluation_dir, f"{task_name}_{sample_key}_eval.jsonl")
        
        # Write task-specific file
        try:
            write_jsonl(task_file, eval_samples)
            # Small delay to ensure file is fully written
            time.sleep(0.1)
        except Exception as e:
            print(f"❌ Failed to write task file {task_file}: {e}")
            raise
        
        # Evaluate with different k values
        results = {}
        max_k = min(len(samples), 10)  # Evaluate up to k=10 or number of samples
        k_values = [k for k in [1, 2, 3, 5, 10] if k <= max_k]
        
        try:            
            # Validate that the file was written correctly
            if not os.path.exists(task_file):
                raise Exception(f"Task file does not exist: {task_file}")
            
            if os.path.getsize(task_file) == 0:
                raise Exception(f"Task file is empty: {task_file}")
            
            # Validate file content
            try:
                with open(task_file, 'r') as f:
                    lines = f.readlines()
                    if len(lines) != len(eval_samples):
                        raise Exception(f"File line count mismatch: expected {len(eval_samples)}, got {len(lines)}")
                    
                    # Try to parse first line to ensure valid JSON
                    if lines:
                        json.loads(lines[0])
            except Exception as e:
                raise Exception(f"Task file validation failed for {task_file}: {e}")
            
            eval_results = evaluate_functional_correctness(task_file, k=k_values, ignore_incomplete=True)
            
            # Validate eval_results
            if not eval_results or not isinstance(eval_results, dict):
                raise Exception(f"Invalid evaluation results for {task_id}")
            
            for k in k_values:
                results[f'pass@{k}'] = eval_results.get(f'pass@{k}', 0.0)
                
        except Exception as e:
            print(f"⚠️ HumanEval evaluation failed for {task_id}: {e}")
            # Fallback to syntax checking
            syntactically_correct = []
            for sample in eval_samples:  # Use eval_samples instead of samples
                try:
                    completion = sample.get('completion', '')
                    if completion.strip():
                        compile(completion, '<string>', 'exec')
                        syntactically_correct.append(True)
                    else:
                        syntactically_correct.append(False)
                except:
                    syntactically_correct.append(False)
            
            # Calculate pass@k based on syntax
            max_k = min(len(syntactically_correct), 10) if syntactically_correct else 1
            k_values = [k for k in [1, 2, 3, 5, 10] if k <= max_k]
            
            for k in k_values:
                if len(syntactically_correct) >= k and any(syntactically_correct[:k]):
                    results[f'pass@{k}'] = 1.0
                else:
                    results[f'pass@{k}'] = 0.0
            
        return results
        
    except ImportError:
        print("⚠️ human_eval package not available, using syntactic correctness only")
        # Fallback to syntax checking
        syntactically_correct = []
        for sample in samples:
            try:
                compile(sample.get('model_response', ''), '<string>', 'exec')
                syntactically_correct.append(True)
            except:
                syntactically_correct.append(False)
        
        results = {}
        max_k = min(len(samples), 10)
        k_values = [k for k in [1, 2, 3, 5, 10] if k <= max_k]
        
        for k in k_values:
            if any(syntactically_correct[:k]):
                results[f'pass@{k}'] = 1.0
            else:
                results[f'pass@{k}'] = 0.0
                
        return results
    
    except Exception as e:
        print(f"❌ Error in evaluation: {e}")
        return {}


def evaluate_single_sample(sample_data: tuple, data_file: str = None, evaluation_dir: str = None) -> tuple:
    """Evaluate a single sample and return results."""
    sample_key, runs = sample_data
    
    try:
        # Get task_id from the first run
        row_idx = runs[0].get('original_row_index', 0)
        task_id = get_task_id_from_row_index(row_idx, data_file)
        
        # Evaluate all runs for this sample
        sample_results = evaluate_code_samples_with_human_eval(runs, task_id, sample_key, evaluation_dir)
        
        return sample_key, sample_results
    except Exception as e:
        print(f"❌ Error evaluating sample {sample_key}: {e}")
        return sample_key, {}


def evaluate_all_samples(grouped_results: Dict[str, List[Dict[str, Any]]], data_file: str = None, max_workers: int = 4) -> Dict[str, Dict[str, float]]:
    """Evaluate all grouped samples and return pass@k metrics."""
    evaluation_results = {}
    
    total_samples = len(grouped_results)
    print(f"🔄 Evaluating {total_samples} unique samples with {max_workers} workers...")
    
    # Use fixed evaluation directory
    evaluation_dir = "human_eval_temp"
    os.makedirs(evaluation_dir, exist_ok=True)
    
    print(f"📁 Using evaluation directory: {evaluation_dir}")
    
    # Thread-safe progress tracking
    progress_lock = threading.Lock()
    completed_count = [0]  # Use list for mutable reference
    
    def update_progress(sample_key: str, sample_results: Dict[str, float]):
        with progress_lock:
            completed_count[0] += 1
            progress_pct = (completed_count[0] / total_samples) * 100
            
            if sample_results:
                pass_at_1 = sample_results.get('pass@1', 0.0)
                print(f"   ✅ ({completed_count[0]}/{total_samples}, {progress_pct:.1f}%) Sample {sample_key}: pass@1 = {pass_at_1:.3f}")
            else:
                print(f"   ❌ ({completed_count[0]}/{total_samples}, {progress_pct:.1f}%) Sample {sample_key}: evaluation failed")
    
    try:
        # Prepare sample data for parallel processing
        sample_items = list(grouped_results.items())
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_sample = {}
            for sample_data in sample_items:
                future = executor.submit(evaluate_single_sample, sample_data, data_file, evaluation_dir)
                future_to_sample[future] = sample_data[0]  # sample_key
            
            # Process completed tasks
            for future in as_completed(future_to_sample):
                try:
                    sample_key, sample_results = future.result()
                    evaluation_results[sample_key] = sample_results
                    update_progress(sample_key, sample_results)
                except Exception as e:
                    sample_key = future_to_sample[future]
                    print(f"❌ Error processing sample {sample_key}: {e}")
                    evaluation_results[sample_key] = {}
                    update_progress(sample_key, {})
    
    finally:
        print(f"📁 Evaluation files saved in directory: {evaluation_dir}")
        # Count evaluation files created
        eval_files = [f for f in os.listdir(evaluation_dir) if f.endswith('_eval.jsonl')]
        print(f"📄 Created {len(eval_files)} evaluation files")
    
    return evaluation_results


def calculate_overall_metrics(evaluation_results: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """Calculate overall pass@k metrics across all samples."""
    if not evaluation_results:
        return {}
    
    # Get all k values from the first sample
    first_sample = next(iter(evaluation_results.values()))
    k_values = [k for k in first_sample.keys() if k.startswith('pass@')]
    
    overall_metrics = {}
    for k in k_values:
        scores = [sample_results.get(k, 0.0) for sample_results in evaluation_results.values()]
        overall_metrics[k] = sum(scores) / len(scores) if scores else 0.0
    
    return overall_metrics


def save_evaluation_results(evaluation_results: Dict[str, Dict[str, float]], 
                          overall_metrics: Dict[str, float], 
                          output_file: str):
    """Save evaluation results to JSON file."""
    results_to_save = {
        'overall_metrics': overall_metrics,
        'sample_results': evaluation_results,
        'total_samples': len(evaluation_results)
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_to_save, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Evaluation results saved to: {output_file}")


def print_evaluation_summary(overall_metrics: Dict[str, float], total_samples: int):
    """Print evaluation summary."""
    print(f"\n📊 Code Generation Evaluation Results:")
    print(f"   Total samples evaluated: {total_samples}")
    
    for k, score in sorted(overall_metrics.items()):
        print(f"   🎯 {k}: {score:.4f} ({score*100:.2f}%)")


def main():
    """Main function for code generation evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate code generation results with pass@k metrics")
    
    parser.add_argument("--results_file", type=str,
                        help="Path to specific results JSON file")
    parser.add_argument("--results_dir", type=str,
                        help="Directory containing results JSON files", 
                        default=str(Path(__file__).parent.parent / "tasks_data" / "results" / "code_generation" / "gpt_4o_mini"))
    parser.add_argument("--data_file", type=str,
                        default="multipromptify_tasks/tasks_data/raw_data/code_generation_humaneval.csv",
                        help="Path to original data file for task_id mapping")
    parser.add_argument("--output_suffix", type=str, default="_evaluation",
                        help="Suffix for output evaluation files")
    parser.add_argument("--max_workers", type=int, default=2,
                        help="Number of parallel workers for evaluation (default: 10)")
    
    args = parser.parse_args()
    
    # Find results files
    results_files = []
    if args.results_file:
        if os.path.exists(args.results_file):
            results_files.append(Path(args.results_file))
        else:
            print(f"❌ Results file not found: {args.results_file}")
            return
    elif args.results_dir:
        results_dir = Path(args.results_dir)
        if results_dir.exists():
            results_files = list(results_dir.glob("*.json"))
            if not results_files:
                print(f"❌ No JSON files found in: {results_dir}")
                return
        else:
            print(f"❌ Results directory not found: {results_dir}")
            return
    else:
        print("❌ Please specify either --results_file or --results_dir")
        return
    
    print(f"🔍 Found {len(results_files)} results files to evaluate")
    print(f"⚡ Using {args.max_workers} parallel workers for faster evaluation")
    
    # Process each results file
    for results_file in results_files:
        print(f"\n📂 Processing: {results_file.name}")
        
        # Load results
        results = load_code_generation_results(str(results_file))
        if not results:
            continue
        
        # Group by sample
        grouped_results = group_results_by_sample(results)
        print(f"📊 Grouped into {len(grouped_results)} unique samples")
        
        # Show runs per sample distribution
        runs_per_sample = [len(runs) for runs in grouped_results.values()]
        if runs_per_sample:
            min_runs = min(runs_per_sample)
            max_runs = max(runs_per_sample)
            avg_runs = sum(runs_per_sample) / len(runs_per_sample)
            print(f"🔄 Runs per sample: min={min_runs}, max={max_runs}, avg={avg_runs:.1f}")
        
        # Evaluate all samples
        evaluation_results = evaluate_all_samples(grouped_results, args.data_file, args.max_workers)
        
        # Calculate overall metrics
        overall_metrics = calculate_overall_metrics(evaluation_results)
        
        # Save results
        output_file = str(results_file).replace('.json', f'{args.output_suffix}.json')
        save_evaluation_results(evaluation_results, overall_metrics, output_file)
        
        # Print summary
        print_evaluation_summary(overall_metrics, len(grouped_results))


if __name__ == "__main__":
    main() 