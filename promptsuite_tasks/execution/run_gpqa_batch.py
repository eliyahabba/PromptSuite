#!/usr/bin/env python3
"""
GPQA Batch Runner
Automatically runs language model on GPQA variation files.

Example usage:
python run_gpqa_batch.py --batch_size 5 --max_retries 5
python run_gpqa_batch.py --model llama_3_3_70b --max_tokens 512
python run_gpqa_batch.py --no_resume  # Start fresh
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, Callable

from promptsuite_tasks.constants import MODEL_SHORT_NAMES
from promptsuite_tasks.execution.batch_runner_base import BatchRunnerBase
from promptsuite_tasks.execution.run_language_model import get_model_name
from promptsuite_tasks.execution.shared_metrics import calculate_gpqa_correctness_and_metrics


class GPQABatchRunner(BatchRunnerBase):
    """Batch runner for GPQA tasks."""

    def __init__(self):
        super().__init__(
            task_name="GPQA",
            data_dir_name="gpqa",
            file_pattern="gpqa_*_variations.json"
        )

    def extract_identifier_from_filename(self, filename: str) -> str:
        """Extract dataset name from GPQA filename."""
        if filename.startswith('gpqa_') and filename.endswith('_variations.json'):
            return filename[5:-16]  # Remove 'gpqa_' (5 chars) and '_variations.json' (16 chars)
        return filename

    def get_display_name(self, identifier: str) -> str:
        """Convert identifier to display name."""
        return identifier.replace('_', ' ').title()

    def get_metrics_function(self) -> Optional[Callable]:
        """Return the metrics function for GPQA tasks."""
        return calculate_gpqa_correctness_and_metrics

    def create_metrics_function_with_gold_field(self, gold_field: str) -> Optional[Callable]:
        """Create a metrics function that uses the specified gold_field for GPQA."""
        def gpqa_metrics_with_field(variation: dict, model_response: str) -> tuple:
            return calculate_gpqa_correctness_and_metrics(variation, model_response, gold_field)
        return gpqa_metrics_with_field

    def create_result_dict(self, identifier: str, status: str, duration: float,
                           variations_processed: int = None, output_file: str = None,
                           error: str = None) -> Dict[str, Any]:
        """Create a result dictionary with GPQA-specific fields."""
        result = {
            "dataset": identifier,  # GPQA uses 'dataset' instead of 'identifier'
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


def print_gpqa_accuracy_summary(results_dir: Path, model_short: str) -> None:
    """Print GPQA accuracy summary."""
    model_dir = results_dir / model_short
    if not model_dir.exists():
        print("📊 No accuracy data available")
        return

    # Find all JSON result files
    json_files = list(model_dir.glob("*.json"))

    if not json_files:
        print("📊 No accuracy data available")
        return

    total_responses = 0
    total_correct = 0
    dataset_accuracies = {}

    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                dataset_results = json.load(f)

            dataset_name = json_file.stem
            dataset_total = len(dataset_results)
            dataset_correct = sum(1 for result in dataset_results if result.get('is_correct', False))
            dataset_accuracy = (dataset_correct / dataset_total * 100) if dataset_total > 0 else 0.0

            total_responses += dataset_total
            total_correct += dataset_correct
            dataset_accuracies[dataset_name] = {
                "accuracy": dataset_accuracy,
                "correct": dataset_correct,
                "total": dataset_total
            }
        except Exception as e:
            print(f"⚠️  Error reading {json_file}: {e}")

    if total_responses == 0:
        print("📊 No accuracy data available")
        return

    overall_accuracy = (total_correct / total_responses * 100)

    print(f"\n📊 GPQA Results Summary:")
    print(f"   Total datasets: {len(dataset_accuracies)}")
    print(f"   Total responses: {total_responses}")
    print(f"   ✅ Total correct: {total_correct}")
    print(f"   📈 Overall accuracy: {overall_accuracy:.2f}%")

    # Show dataset performance
    if dataset_accuracies:
        sorted_datasets = sorted(dataset_accuracies.items(), key=lambda x: x[1]['accuracy'], reverse=True)

        print(f"\n🏆 Dataset performance:")
        for i, (dataset, data) in enumerate(sorted_datasets, 1):
            print(f"   {i}. {dataset}: {data['accuracy']:.2f}% ({data['correct']}/{data['total']})")


def main():
    """Main function to run language model on all GPQA files."""
    runner = GPQABatchRunner()

    parser = argparse.ArgumentParser(description="Run language model on GPQA variations")

    # Setup common arguments
    default_data_dir = str(Path(__file__).parent.parent / "tasks_data" / "generated_data" / "gpqa")
    runner.setup_common_args(parser, default_data_dir)

    # Add GPQA-specific arguments
    parser.add_argument("--datasets", nargs="+",
                        help="Run only specific datasets (e.g., --datasets diamond)")
    parser.add_argument("--list_datasets", action="store_true",
                        help="List available datasets and exit")
    
    # Add gold_field with GPQA-specific default
    runner.add_gold_field_with_default(parser, "Correct Answer", "Field name in gold_updates containing the correct answer (default: 'Correct Answer')")

    args = parser.parse_args()

    # Handle list datasets option
    if args.list_datasets:
        gpqa_dir = Path(args.gpqa_dir).resolve()
        if not gpqa_dir.exists():
            print(f"❌ GPQA directory not found: {gpqa_dir}")
            return

        gpqa_files = runner.find_variation_files(gpqa_dir)
        if not gpqa_files:
            print(f"❌ No GPQA variation files found in: {gpqa_dir}")
            print("   Expected files matching pattern: gpqa_*_variations.json")
            return

        datasets = [runner.extract_identifier_from_filename(f.name) for f in gpqa_files]
        datasets.sort()

        print(f"🧪 Available GPQA datasets ({len(datasets)}):")
        for i, dataset in enumerate(datasets, 1):
            print(f"   {i:2d}. {dataset}")
        return

    # Get the full model name based on platform and model key
    try:
        full_model_name = get_model_name(args.platform, args.model)
    except ValueError as e:
        print(f"❌ {e}")
        return

    # Find and filter GPQA files
    gpqa_dir = Path(args.gpqa_dir).resolve()
    if not gpqa_dir.exists():
        print(f"❌ GPQA directory not found: {gpqa_dir}")
        return

    gpqa_files = runner.find_variation_files(gpqa_dir)
    if not gpqa_files:
        print(f"❌ No GPQA variation files found in: {gpqa_dir}")
        return

    # Filter datasets if specified
    if args.datasets:
        datasets_to_include = set(args.datasets)
        gpqa_files = [f for f in gpqa_files
                      if runner.extract_identifier_from_filename(f.name) in datasets_to_include]
        if not gpqa_files:
            print(f"❌ No files found for specified datasets: {args.datasets}")
            return

    # Print header and process files
    runner.print_header(args, full_model_name, gpqa_files)

    # Process files
    results = []
    total_start_time = time.time()

    for i, gpqa_file in enumerate(gpqa_files, 1):
        dataset = runner.extract_identifier_from_filename(gpqa_file.name)
        print(f"\n🧪 Processing dataset {i}/{len(gpqa_files)}: {dataset}")

        result = runner.run_language_model_on_file(gpqa_file, args)
        results.append(result)

        # Print result
        if result["status"] == "success":
            variations_count = result.get("variations_processed", "unknown")
            print(f"✅ {dataset} completed in {result['duration']:.1f}s ({variations_count} variations)")
        else:
            print(f"❌ {dataset} failed: {result.get('error', 'Unknown error')} in {result['duration']:.1f}s")

        # Show progress
        if i < len(gpqa_files):
            runner.print_progress_summary(results, i, len(gpqa_files))

    # Save summary and print final results
    total_duration = time.time() - total_start_time
    model_short = MODEL_SHORT_NAMES.get(full_model_name, full_model_name.replace(" ", "_"))
    results_dir = Path(__file__).parent.parent / "tasks_data" / "results" / "gpqa"

    # Print simple summary
    successful = len([r for r in results if r["status"] == "success"])
    print(f"\n🎉 GPQA Processing Completed!")
    print(f"   ✅ Successful: {successful}/{len(results)}")
    print(f"   ⏱️  Total time: {total_duration:.1f}s")

    # Print accuracy summary
    print_gpqa_accuracy_summary(results_dir, model_short)


if __name__ == "__main__":
    main() 