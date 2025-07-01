#!/usr/bin/env python3
"""
Code Generation Batch Runner
Automatically runs language model on code generation variation files.

Example usage:
python run_code_generation_batch.py --batch_size 5 --max_retries 5
python run_code_generation_batch.py --model llama_3_3_70b --max_tokens 512
python run_code_generation_batch.py --no_resume  # Start fresh
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, Callable

from multipromptify_tasks.constants import MODEL_SHORT_NAMES
from multipromptify_tasks.execution.batch_runner_base import BatchRunnerBase
from multipromptify_tasks.execution.run_language_model import get_model_name
from multipromptify_tasks.execution.shared_metrics import calculate_code_generation_correctness_and_metrics


class CodeGenerationBatchRunner(BatchRunnerBase):
    """Batch runner for code generation tasks."""

    def __init__(self):
        super().__init__(
            task_name="Code Generation",
            data_dir_name="code_generation",
            file_pattern="code_generation_*_variations.json"
        )

    def extract_identifier_from_filename(self, filename: str) -> str:
        """Extract dataset name from code generation filename."""
        if filename.startswith('code_generation_') and filename.endswith('_variations.json'):
            return filename[16:-16]  # Remove 'code_generation_' (16 chars) and '_variations.json' (16 chars)
        return filename

    def get_metrics_function(self) -> Optional[Callable]:
        """Return the metrics function for code generation tasks."""
        return calculate_code_generation_correctness_and_metrics

    def create_metrics_function_with_gold_field(self, gold_field: str) -> Optional[Callable]:
        """Create a metrics function that uses the specified gold_field for code generation."""
        def code_generation_metrics_with_field(variation: Dict[str, Any], model_response: str) -> tuple:
            return calculate_code_generation_correctness_and_metrics(variation, model_response, gold_field)
        return code_generation_metrics_with_field

    def create_result_dict(self, identifier: str, status: str, duration: float,
                           variations_processed: int = None, output_file: str = None,
                           error: str = None) -> Dict[str, Any]:
        """Create a result dictionary with code generation-specific fields."""
        result = {
            "dataset": identifier,  # Code generation uses 'dataset' instead of 'identifier'
            "status": status,
            "duration": duration
        }

        if variations_processed is not None:
            result["variations_processed"] = variations_processed
        if output_file is not None:
            result["output_file"] = output_file
        if error is not None:
            result["error"] = error

        return result


def print_code_generation_summary(results_dir: Path, model_short: str) -> None:
    """Print code generation accuracy and metrics summary."""
    model_dir = results_dir / model_short
    if not model_dir.exists():
        print("📊 No code generation data available")
        return

    # Find all JSON result files
    json_files = list(model_dir.glob("*.json"))

    if not json_files:
        print("📊 No code generation data available")
        return

    total_responses = 0
    all_results = []

    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                dataset_results = json.load(f)
            all_results.extend(dataset_results)
            total_responses += len(dataset_results)

        except Exception as e:
            print(f"⚠️  Error reading {json_file}: {e}")

    if total_responses == 0:
        print("📊 No code generation data available")
        return

    print(f"\n📊 Code Generation Results Summary:")
    print(f"   Total responses: {total_responses}")
    
    # Check if we have multiple runs per sample
    run_numbers = [result.get('run_number', 1) for result in all_results]
    unique_runs = set(run_numbers)
    if len(unique_runs) > 1:
        max_runs = max(unique_runs)
        print(f"   🔄 Multiple runs detected: {max_runs} runs per sample")
        
        # Show distribution by run number
        for run_num in sorted(unique_runs):
            run_results = [r for r in all_results if r.get('run_number', 1) == run_num]
            print(f"   📊 Run {run_num}: {len(run_results)} responses")
    
    print(f"   💾 Code generation completed - evaluation should be done separately")
    print(f"   📝 Use a separate evaluation script to calculate pass@k metrics")

    # Calculate text generation metrics if available
    if all_results and any('bleu' in result for result in all_results):
        metrics = ['bleu', 'rouge1', 'rouge2', 'rougeL', 'sacrebleu']
        print(f"\n📈 Text Generation Metrics:")
        
        for metric in metrics:
            scores = [result.get(metric, 0.0) for result in all_results if metric in result]
            if scores:
                avg_score = sum(scores) / len(scores)
                if metric == 'sacrebleu':
                    print(f"   {metric.upper()}: {avg_score:.2f}")
                else:
                    print(f"   {metric.upper()}: {avg_score:.4f}")

    # Show code length statistics
    code_lengths = [result.get('code_length', 0) for result in all_results if 'code_length' in result]
    canonical_lengths = [result.get('canonical_length', 0) for result in all_results if 'canonical_length' in result]
    
    if code_lengths and canonical_lengths:
        avg_generated_length = sum(code_lengths) / len(code_lengths)
        avg_canonical_length = sum(canonical_lengths) / len(canonical_lengths)
        print(f"\n📏 Code Length Statistics:")
        print(f"   Average generated code length: {avg_generated_length:.1f} chars")
        print(f"   Average canonical code length: {avg_canonical_length:.1f} chars")


def main():
    """Main function to run language model on all code generation files."""
    runner = CodeGenerationBatchRunner()

    parser = argparse.ArgumentParser(description="Run language model on all code generation variations")

    # Setup common arguments
    default_data_dir = str(Path(__file__).parent.parent / "tasks_data" / "generated_data" / "code_generation")
    runner.setup_common_args(parser, default_data_dir)

    # Add code generation-specific arguments
    parser.add_argument("--datasets", nargs="+",
                        help="Run only specific datasets (e.g., --datasets humaneval)")
    parser.add_argument("--list_datasets", action="store_true",
                        help="List available datasets and exit")
    parser.add_argument("--runs_per_sample", type=int, default=5,
                        help="Number of times to run each sample/variation (default: 1)")
    
    # Add gold_field with code generation-specific default
    runner.add_gold_field_with_default(parser, "canonical_solution", "Field name in gold_updates containing the canonical solution (default: 'canonical_solution')")

    args = parser.parse_args()

    # Handle list datasets option
    if args.list_datasets:
        code_generation_dir = Path(args.code_generation_dir).resolve()
        if not code_generation_dir.exists():
            print(f"❌ Code generation directory not found: {code_generation_dir}")
            return

        code_generation_files = runner.find_variation_files(code_generation_dir)
        if not code_generation_files:
            print(f"❌ No code generation variation files found in: {code_generation_dir}")
            print("   Expected files matching pattern: code_generation_*_variations.json")
            return

        datasets = [runner.extract_identifier_from_filename(f.name) for f in code_generation_files]
        datasets.sort()

        print(f"💻 Available code generation datasets ({len(datasets)}):")
        for i, dataset in enumerate(datasets, 1):
            print(f"   {i:2d}. {dataset}")
        return

    # Get the full model name based on platform and model key
    try:
        full_model_name = get_model_name(args.platform, args.model)
    except ValueError as e:
        print(f"❌ {e}")
        return

    # Find and filter code generation files
    code_generation_dir = Path(args.code_generation_dir).resolve()
    if not code_generation_dir.exists():
        print(f"❌ Code generation directory not found: {code_generation_dir}")
        return

    code_generation_files = runner.find_variation_files(code_generation_dir)
    if not code_generation_files:
        print(f"❌ No code generation variation files found in: {code_generation_dir}")
        return

    # Filter datasets if specified
    if args.datasets:
        datasets_to_include = set(args.datasets)
        code_generation_files = [f for f in code_generation_files
                      if runner.extract_identifier_from_filename(f.name) in datasets_to_include]
        if not code_generation_files:
            print(f"❌ No files found for specified datasets: {args.datasets}")
            return

    # Print header and process files
    runner.print_header(args, full_model_name, code_generation_files)

    # Process files
    results = []
    total_start_time = time.time()

    for i, code_generation_file in enumerate(code_generation_files, 1):
        dataset = runner.extract_identifier_from_filename(code_generation_file.name)
        print(f"\n💻 Processing dataset {i}/{len(code_generation_files)}: {dataset}")

        result = runner.run_language_model_on_file(code_generation_file, args)
        results.append(result)

        # Print result
        if result["status"] == "success":
            variations_count = result.get("variations_processed", "unknown")
            print(f"✅ {dataset} completed in {result['duration']:.1f}s ({variations_count} variations)")
        else:
            print(f"❌ {dataset} failed: {result.get('error', 'Unknown error')} in {result['duration']:.1f}s")

        # Show progress
        if i < len(code_generation_files):
            runner.print_progress_summary(results, i, len(code_generation_files))

    # Save summary and print final results
    total_duration = time.time() - total_start_time
    model_short = MODEL_SHORT_NAMES.get(full_model_name, full_model_name.replace(" ", "_"))
    results_dir = Path(__file__).parent.parent / "tasks_data" / "results" / "code_generation"

    # Print simple summary
    successful = len([r for r in results if r["status"] == "success"])
    print(f"\n🎉 Code Generation Processing Completed!")
    print(f"   ✅ Successful: {successful}/{len(results)}")
    print(f"   ⏱️  Total time: {total_duration:.1f}s")

    # Print accuracy summary
    print_code_generation_summary(results_dir, model_short)


if __name__ == "__main__":
    main() 