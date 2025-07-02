#!/usr/bin/env python3
"""
Sentiment Analysis Batch Runner
Automatically runs language model on sentiment analysis variation files.

Example usage:
python run_sentiment_batch.py --batch_size 5 --max_retries 5
python run_sentiment_batch.py --model llama_3_3_70b --max_tokens 512
python run_sentiment_batch.py --no_resume  # Start fresh
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, Callable

from promptsuite_tasks.constants import MODEL_SHORT_NAMES
from promptsuite_tasks.execution.batch_runner_base import BatchRunnerBase
from promptsuite_tasks.execution.run_language_model import get_model_name
from promptsuite_tasks.execution.shared_metrics import calculate_sentiment_correctness_and_metrics


class SentimentBatchRunner(BatchRunnerBase):
    """Batch runner for sentiment analysis tasks."""

    def __init__(self):
        super().__init__(
            task_name="Sentiment Analysis",
            data_dir_name="sentiment",
            file_pattern="sentiment_*_variations.json"
            #gold is the "label"

        )

    def extract_identifier_from_filename(self, filename: str) -> str:
        """Extract dataset name from sentiment filename."""
        if filename.startswith('sentiment_') and filename.endswith('_variations.json'):
            return filename[10:-16]  # Remove 'sentiment_' (10 chars) and '_variations.json' (16 chars)
        return filename

    def get_display_name(self, identifier: str) -> str:
        """Convert identifier to display name."""
        return identifier.upper()

    def get_metrics_function(self) -> Optional[Callable]:
        """Return the metrics function for sentiment analysis tasks."""
        return calculate_sentiment_correctness_and_metrics

    def create_metrics_function_with_gold_field(self, gold_field: str) -> Optional[Callable]:
        """Create a metrics function that uses the specified gold_field for sentiment analysis."""
        def sentiment_metrics_with_field(variation: dict, model_response: str) -> tuple:
            return calculate_sentiment_correctness_and_metrics(variation, model_response, gold_field)
        return sentiment_metrics_with_field

    def create_result_dict(self, identifier: str, status: str, duration: float,
                           variations_processed: int = None, output_file: str = None,
                           error: str = None) -> Dict[str, Any]:
        """Create a result dictionary with sentiment-specific fields."""
        result = {
            "dataset": identifier,  # Sentiment uses 'dataset' instead of 'identifier'
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


def print_sentiment_accuracy_summary(results_dir: Path, model_short: str) -> None:
    """Print sentiment analysis accuracy and regression metrics summary."""
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
    all_mse_scores = []
    all_mae_scores = []
    dataset_accuracies = {}

    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                dataset_results = json.load(f)

            dataset_name = json_file.stem
            dataset_total = len(dataset_results)
            dataset_correct = sum(1 for result in dataset_results if result.get('is_correct', False))
            dataset_accuracy = (dataset_correct / dataset_total * 100) if dataset_total > 0 else 0.0

            # Collect regression metrics
            dataset_mse = [result.get('mse', 0.0) for result in dataset_results if 'mse' in result]
            dataset_mae = [result.get('mae', 0.0) for result in dataset_results if 'mae' in result]
            
            all_mse_scores.extend(dataset_mse)
            all_mae_scores.extend(dataset_mae)

            total_responses += dataset_total
            total_correct += dataset_correct
            dataset_accuracies[dataset_name] = {
                "accuracy": dataset_accuracy,
                "correct": dataset_correct,
                "total": dataset_total,
                "avg_mse": sum(dataset_mse) / len(dataset_mse) if dataset_mse else 0.0,
                "avg_mae": sum(dataset_mae) / len(dataset_mae) if dataset_mae else 0.0
            }
        except Exception as e:
            print(f"⚠️  Error reading {json_file}: {e}")

    if total_responses == 0:
        print("📊 No accuracy data available")
        return

    overall_accuracy = (total_correct / total_responses * 100)
    overall_mse = sum(all_mse_scores) / len(all_mse_scores) if all_mse_scores else 0.0
    overall_mae = sum(all_mae_scores) / len(all_mae_scores) if all_mae_scores else 0.0

    print(f"\n📊 Sentiment Analysis Results Summary:")
    print(f"   Total datasets: {len(dataset_accuracies)}")
    print(f"   Total responses: {total_responses}")
    print(f"   ✅ Total correct (±0.2 tolerance): {total_correct}")
    print(f"   📈 Overall accuracy: {overall_accuracy:.2f}%")
    print(f"   📉 Overall MSE: {overall_mse:.4f}")
    print(f"   📏 Overall MAE: {overall_mae:.4f}")

    # Show dataset performance
    if dataset_accuracies:
        sorted_datasets = sorted(dataset_accuracies.items(), key=lambda x: x[1]['accuracy'], reverse=True)

        print(f"\n🏆 Dataset performance:")
        for i, (dataset, data) in enumerate(sorted_datasets, 1):
            print(f"   {i}. {dataset}: {data['accuracy']:.2f}% (MSE: {data['avg_mse']:.4f}, MAE: {data['avg_mae']:.4f})")


def main():
    """Main function to run language model on all sentiment analysis files."""
    runner = SentimentBatchRunner()

    parser = argparse.ArgumentParser(description="Run language model on sentiment analysis variations")

    # Setup common arguments
    default_data_dir = str(Path(__file__).parent.parent / "tasks_data" / "generated_data" / "sentiment")
    runner.setup_common_args(parser, default_data_dir)

    # Add sentiment-specific arguments
    parser.add_argument("--datasets", nargs="+",
                        help="Run only specific datasets (e.g., --datasets sst)")
    parser.add_argument("--list_datasets", action="store_true",
                        help="List available datasets and exit")
    
    # Add gold_field with sentiment-specific default
    runner.add_gold_field_with_default(parser, "label", "Field name in gold_updates containing the sentiment score (default: 'label')")

    args = parser.parse_args()

    # Handle list datasets option
    if args.list_datasets:
        sentiment_dir = Path(args.sentiment_dir).resolve()
        if not sentiment_dir.exists():
            print(f"❌ Sentiment directory not found: {sentiment_dir}")
            return

        sentiment_files = runner.find_variation_files(sentiment_dir)
        if not sentiment_files:
            print(f"❌ No sentiment variation files found in: {sentiment_dir}")
            print("   Expected files matching pattern: sentiment_*_variations.json")
            return

        datasets = [runner.extract_identifier_from_filename(f.name) for f in sentiment_files]
        datasets.sort()

        print(f"📊 Available sentiment datasets ({len(datasets)}):")
        for i, dataset in enumerate(datasets, 1):
            print(f"   {i:2d}. {dataset}")
        return

    # Get the full model name based on platform and model key
    try:
        full_model_name = get_model_name(args.platform, args.model)
    except ValueError as e:
        print(f"❌ {e}")
        return

    # Find and filter sentiment files
    sentiment_dir = Path(args.sentiment_dir).resolve()
    if not sentiment_dir.exists():
        print(f"❌ Sentiment directory not found: {sentiment_dir}")
        return

    sentiment_files = runner.find_variation_files(sentiment_dir)
    if not sentiment_files:
        print(f"❌ No sentiment variation files found in: {sentiment_dir}")
        return

    # Filter datasets if specified
    if args.datasets:
        datasets_to_include = set(args.datasets)
        sentiment_files = [f for f in sentiment_files
                          if runner.extract_identifier_from_filename(f.name) in datasets_to_include]
        if not sentiment_files:
            print(f"❌ No files found for specified datasets: {args.datasets}")
            return

    # Print header and process files
    runner.print_header(args, full_model_name, sentiment_files)

    # Process files
    results = []
    total_start_time = time.time()

    for i, sentiment_file in enumerate(sentiment_files, 1):
        dataset = runner.extract_identifier_from_filename(sentiment_file.name)
        print(f"\n📊 Processing dataset {i}/{len(sentiment_files)}: {dataset}")

        result = runner.run_language_model_on_file(sentiment_file, args)
        results.append(result)

        # Print result
        if result["status"] == "success":
            variations_count = result.get("variations_processed", "unknown")
            print(f"✅ {dataset} completed in {result['duration']:.1f}s ({variations_count} variations)")
        else:
            print(f"❌ {dataset} failed: {result.get('error', 'Unknown error')} in {result['duration']:.1f}s")

        # Show progress
        if i < len(sentiment_files):
            runner.print_progress_summary(results, i, len(sentiment_files))

    # Save summary and print final results
    total_duration = time.time() - total_start_time
    model_short = MODEL_SHORT_NAMES.get(full_model_name, full_model_name.replace(" ", "_"))
    results_dir = Path(__file__).parent.parent/  "tasks_data" / "results" / "sentiment"

    # Print simple summary
    successful = len([r for r in results if r["status"] == "success"])
    print(f"\n🎉 Sentiment Analysis Processing Completed!")
    print(f"   ✅ Successful: {successful}/{len(results)}")
    print(f"   ⏱️  Total time: {total_duration:.1f}s")

    # Print accuracy summary
    print_sentiment_accuracy_summary(results_dir, model_short)


if __name__ == "__main__":
    main() 