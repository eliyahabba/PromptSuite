#!/usr/bin/env python3
"""
Question Answering Batch Runner
Automatically runs language model on all QA variation files.

Example usage:
python run_qa_batch.py --batch_size 5 --max_retries 5
python run_qa_batch.py --model llama_3_3_70b --max_tokens 512
python run_qa_batch.py --no_resume  # Start fresh
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, Callable

from promptsuite_tasks.constants import MODEL_SHORT_NAMES
from promptsuite_tasks.execution.batch_runner_base import BatchRunnerBase
from promptsuite_tasks.execution.run_language_model import get_model_name
from promptsuite_tasks.execution.shared_metrics import calculate_qa_correctness_and_metrics


class QABatchRunner(BatchRunnerBase):
    """Batch runner for Question Answering tasks."""

    def __init__(self):
        super().__init__(
            task_name="Question Answering",
            data_dir_name="qa",
            file_pattern="question_answering_*_variations.json"
        )

    def extract_identifier_from_filename(self, filename: str) -> str:
        """Extract dataset name from QA filename."""
        if filename.startswith('question_answering_') and filename.endswith('_variations.json'):
            return filename[19:-16]  # Remove 'question_answering_' (19 chars) and '_variations.json' (16 chars)
        return filename

    def get_metrics_function(self) -> Optional[Callable]:
        """Return the metrics function for QA tasks."""
        return calculate_qa_correctness_and_metrics

    def create_metrics_function_with_gold_field(self, gold_field: str) -> Optional[Callable]:
        """Create a metrics function that uses the specified gold_field for QA."""
        def qa_metrics_with_field(variation: Dict[str, Any], model_response: str) -> tuple:
            return calculate_qa_correctness_and_metrics(variation, model_response, gold_field)
        return qa_metrics_with_field

    def create_result_dict(self, identifier: str, status: str, duration: float,
                           variations_processed: int = None, output_file: str = None,
                           error: str = None) -> Dict[str, Any]:
        """Create a result dictionary with QA-specific fields."""
        result = {
            "dataset": identifier,  # QA uses 'dataset' instead of 'identifier'
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


def print_qa_accuracy_summary(results_dir: Path, model_short: str) -> None:
    """Print QA accuracy and metrics summary."""
    model_dir = results_dir / model_short
    if not model_dir.exists():
        print("📊 No QA data available")
        return

    # Find all JSON result files
    json_files = list(model_dir.glob("*.json"))

    if not json_files:
        print("📊 No QA data available")
        return

    total_responses = 0
    total_correct = 0
    all_results = []

    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                dataset_results = json.load(f)

            dataset_name = json_file.stem
            dataset_total = len(dataset_results)
            dataset_correct = sum(1 for result in dataset_results if result.get('is_correct', False))

            total_responses += dataset_total
            total_correct += dataset_correct
            all_results.extend(dataset_results)

        except Exception as e:
            print(f"⚠️  Error reading {json_file}: {e}")

    if total_responses == 0:
        print("📊 No QA data available")
        return

    overall_accuracy = (total_correct / total_responses * 100)

    print(f"\n📊 Question Answering Results Summary:")
    print(f"   Total responses: {total_responses}")
    print(f"   ✅ Total correct: {total_correct}")
    print(f"   📈 Overall accuracy: {overall_accuracy:.2f}%")

    # Calculate metrics if available
    if all_results and any('word_f1' in result for result in all_results):
        metrics = ['word_f1', 'word_precision', 'word_recall', 'bleu', 'rouge1', 'rouge2', 'rougeL']
        print(f"\n📈 QA Metrics:")
        
        for metric in metrics:
            scores = [result.get(metric, 0.0) for result in all_results if metric in result]
            if scores:
                avg_score = sum(scores) / len(scores)
                if metric.startswith('word_'):
                    print(f"   {metric.upper()}: {avg_score:.4f}")
                elif metric in ['bleu', 'rouge1', 'rouge2', 'rougeL']:
                    print(f"   {metric.upper()}: {avg_score:.4f}")


def main():
    """Main function to run language model on all QA files."""
    runner = QABatchRunner()

    parser = argparse.ArgumentParser(description="Run language model on all Question Answering variations")

    # Setup common arguments
    default_data_dir = str(Path(__file__).parent.parent / "tasks_data" / "generated_data" / "qa")
    runner.setup_common_args(parser, default_data_dir)

    # Add QA-specific arguments
    parser.add_argument("--datasets", nargs="+",
                        help="Run only specific datasets (e.g., --datasets squad)")
    parser.add_argument("--exclude", nargs="+",
                        help="Exclude specific datasets (e.g., --exclude squad)")
    parser.add_argument("--list_datasets", action="store_true",
                        help="List available datasets and exit")
    
    # Add gold_field with QA-specific default
    runner.add_gold_field_with_default(parser, "answer", "Field name in gold_updates containing the correct answer (default: 'answer')")

    args = parser.parse_args()

    # Handle list datasets option
    if args.list_datasets:
        qa_dir = Path(args.qa_dir).resolve()
        if not qa_dir.exists():
            print(f"❌ QA directory not found: {qa_dir}")
            return

        qa_files = runner.find_variation_files(qa_dir)
        if not qa_files:
            print(f"❌ No QA variation files found in: {qa_dir}")
            print("   Expected files matching pattern: question_answering_*_variations.json")
            return

        datasets = [runner.extract_identifier_from_filename(f.name) for f in qa_files]
        datasets.sort()

        print(f"📚 Available QA datasets ({len(datasets)}):")
        for i, dataset in enumerate(datasets, 1):
            print(f"   {i:2d}. {dataset}")
        return

    # Get the full model name based on platform and model key
    try:
        full_model_name = get_model_name(args.platform, args.model)
    except ValueError as e:
        print(f"❌ {e}")
        return

    # Find and filter QA files
    qa_dir = Path(args.qa_dir).resolve()
    if not qa_dir.exists():
        print(f"❌ QA directory not found: {qa_dir}")
        return

    qa_files = runner.find_variation_files(qa_dir)
    if not qa_files:
        print(f"❌ No QA variation files found in: {qa_dir}")
        return

    # Filter datasets if specified
    if args.datasets:
        datasets_to_include = set(args.datasets)
        qa_files = [f for f in qa_files
                    if runner.extract_identifier_from_filename(f.name) in datasets_to_include]
        if not qa_files:
            print(f"❌ No files found for specified datasets: {args.datasets}")
            return

    if args.exclude:
        datasets_to_exclude = set(args.exclude)
        qa_files = [f for f in qa_files
                    if runner.extract_identifier_from_filename(f.name) not in datasets_to_exclude]
        if not qa_files:
            print(f"❌ All files excluded by --exclude: {args.exclude}")
            return

    # Print header and process files
    runner.print_header(args, full_model_name, qa_files)

    # Process files
    results = []
    total_start_time = time.time()

    for i, qa_file in enumerate(qa_files, 1):
        dataset = runner.extract_identifier_from_filename(qa_file.name)
        print(f"\n📚 Processing dataset {i}/{len(qa_files)}: {dataset}")

        result = runner.run_language_model_on_file(qa_file, args)
        results.append(result)

        # Print result
        if result["status"] == "success":
            variations_count = result.get("variations_processed", "unknown")
            print(f"✅ {dataset} completed in {result['duration']:.1f}s ({variations_count} variations)")
        else:
            print(f"❌ {dataset} failed: {result.get('error', 'Unknown error')} in {result['duration']:.1f}s")

        # Show progress
        if i < len(qa_files):
            runner.print_progress_summary(results, i, len(qa_files))

    # Save summary and print final results
    total_duration = time.time() - total_start_time
    model_short = MODEL_SHORT_NAMES.get(full_model_name, full_model_name.replace(" ", "_"))
    results_dir = Path(__file__).parent.parent / "tasks_data" / "results" / "question_answering"

    # Print simple summary
    successful = len([r for r in results if r["status"] == "success"])
    print(f"\n🎉 QA Processing Completed!")
    print(f"   ✅ Successful: {successful}/{len(results)}")
    print(f"   ⏱️  Total time: {total_duration:.1f}s")

    # Print accuracy summary
    print_qa_accuracy_summary(results_dir, model_short)


if __name__ == "__main__":
    main() 