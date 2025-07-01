#!/usr/bin/env python3
"""
MuSiQue Batch Runner
Automatically runs language model on MuSiQue multi-hop question answering variation files.

Example usage:
python run_musique_batch.py --batch_size 5 --max_retries 5
python run_musique_batch.py --model llama_3_3_70b --max_tokens 512
python run_musique_batch.py --no_resume  # Start fresh
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, Callable

from multipromptify_tasks.constants import MODEL_SHORT_NAMES
from multipromptify_tasks.execution.batch_runner_base import BatchRunnerBase, get_model_name
from multipromptify_tasks.execution.shared_metrics import calculate_musique_correctness_and_metrics


class MuSiQueBatchRunner(BatchRunnerBase):
    """Batch runner for MuSiQue tasks."""

    def __init__(self):
        super().__init__(
            task_name="MuSiQue",
            data_dir_name="musique",
            file_pattern="musique*_variations.json"
        )

    def extract_identifier_from_filename(self, filename: str) -> str:
        """Extract identifier from MuSiQue filename."""
        if filename.startswith('musique_') and filename.endswith('_variations.json'):
            return filename[8:-16]  # Remove 'musique_' (8 chars) and '_variations.json' (16 chars)
        return filename

    def get_metrics_function(self) -> Optional[Callable]:
        """Return the metrics function for MuSiQue multi-hop question answering tasks."""
        return calculate_musique_correctness_and_metrics

    def create_metrics_function_with_gold_field(self, gold_field: str) -> Optional[Callable]:
        """Create a metrics function that uses the specified gold_field for MuSiQue."""
        def musique_metrics_with_field(variation: Dict[str, Any], model_response: str) -> tuple:
            return calculate_musique_correctness_and_metrics(variation, model_response, gold_field)
        return musique_metrics_with_field

    def create_result_dict(self, identifier: str, status: str, duration: float,
                           variations_processed: int = None, output_file: str = None,
                           error: str = None) -> Dict[str, Any]:
        """Create a result dictionary with MuSiQue-specific fields."""
        result = {
            "task": identifier,  # MuSiQue uses 'task' instead of 'identifier'
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


def print_musique_accuracy_summary(results_dir: Path, model_short: str) -> None:
    """Print MuSiQue accuracy and metrics summary."""
    model_dir = results_dir / model_short
    if not model_dir.exists():
        print("📊 No MuSiQue data available")
        return

    # Find all JSON result files
    json_files = list(model_dir.glob("*.json"))

    if not json_files:
        print("📊 No MuSiQue data available")
        return

    total_responses = 0
    total_exact_match = 0
    all_results = []

    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                task_results = json.load(f)

            task_total = len(task_results)
            task_exact_match = sum(1 for result in task_results if result.get('is_correct', False))

            total_responses += task_total
            total_exact_match += task_exact_match
            all_results.extend(task_results)

        except Exception as e:
            print(f"⚠️  Error reading {json_file}: {e}")

    if total_responses == 0:
        print("📊 No MuSiQue data available")
        return

    overall_exact_match = (total_exact_match / total_responses * 100)

    print(f"\n📊 MuSiQue Results Summary:")
    print(f"   Total responses: {total_responses}")
    print(f"   ✅ Exact matches: {total_exact_match}")
    print(f"   📈 Exact match accuracy: {overall_exact_match:.2f}%")

    # Calculate additional metrics if available
    if all_results and any('word_f1' in result for result in all_results):
        metrics = ['word_f1', 'word_precision', 'word_recall', 'bleu', 'rouge1', 'rouge2', 'rougeL']
        print(f"\n📈 MuSiQue Detailed Metrics:")
        
        for metric in metrics:
            scores = [result.get(metric, 0.0) for result in all_results if metric in result]
            if scores:
                avg_score = sum(scores) / len(scores)
                if metric.startswith('word_'):
                    print(f"   {metric.replace('_', ' ').title()}: {avg_score:.4f}")
                elif metric in ['bleu', 'rouge1', 'rouge2', 'rougeL']:
                    print(f"   {metric.upper()}: {avg_score:.4f}")


def main():
    """Main function to run language model on MuSiQue files."""
    runner = MuSiQueBatchRunner()

    parser = argparse.ArgumentParser(description="Run language model on MuSiQue multi-hop question answering variations")

    # Setup common arguments
    default_data_dir = str(Path(__file__).parent.parent / "tasks_data" / "generated_data" / "musique")
    runner.setup_common_args(parser, default_data_dir)

    # Add MuSiQue-specific arguments
    parser.add_argument("--tasks", nargs="+",
                        help="Run only specific tasks (if multiple task files exist)")
    parser.add_argument("--list_tasks", action="store_true",
                        help="List available tasks and exit")
    
    # Add gold_field with MuSiQue-specific default
    runner.add_gold_field_with_default(parser, "answer", "Field name in gold_updates containing the correct answer (default: 'answer')")

    args = parser.parse_args()

    # Handle list tasks option
    if args.list_tasks:
        musique_dir = Path(args.musique_dir).resolve()
        if not musique_dir.exists():
            print(f"❌ MuSiQue directory not found: {musique_dir}")
            return

        musique_files = runner.find_variation_files(musique_dir)
        if not musique_files:
            print(f"❌ No MuSiQue variation files found in: {musique_dir}")
            print("   Expected files matching pattern: musique_*_variations.json")
            return

        tasks = [runner.extract_identifier_from_filename(f.name) for f in musique_files]
        tasks.sort()

        print(f"📚 Available MuSiQue tasks ({len(tasks)}):")
        for i, task in enumerate(tasks, 1):
            print(f"   {i:2d}. {task}")
        return

    # Get the full model name based on platform and model key
    try:
        full_model_name = get_model_name(args.platform, args.model)
    except ValueError as e:
        print(f"❌ {e}")
        return

    # Find and filter MuSiQue files
    musique_dir = Path(args.musique_dir).resolve()
    if not musique_dir.exists():
        print(f"❌ MuSiQue directory not found: {musique_dir}")
        return

    musique_files = runner.find_variation_files(musique_dir)
    if not musique_files:
        print(f"❌ No MuSiQue variation files found in: {musique_dir}")
        return

    # Filter tasks if specified
    if args.tasks:
        tasks_to_include = set(args.tasks)
        musique_files = [f for f in musique_files
                        if runner.extract_identifier_from_filename(f.name) in tasks_to_include]
        if not musique_files:
            print(f"❌ No files found for specified tasks: {args.tasks}")
            return

    # Print header and process files
    runner.print_header(args, full_model_name, musique_files)

    # Process files
    results = []
    total_start_time = time.time()

    for i, musique_file in enumerate(musique_files, 1):
        task = runner.extract_identifier_from_filename(musique_file.name)
        print(f"\n📚 Processing task {i}/{len(musique_files)}: {task}")

        result = runner.run_language_model_on_file(musique_file, args)
        results.append(result)

        # Print result
        if result["status"] == "success":
            variations_count = result.get("variations_processed", "unknown")
            print(f"✅ {task} completed in {result['duration']:.1f}s ({variations_count} variations)")
        else:
            print(f"❌ {task} failed: {result.get('error', 'Unknown error')} in {result['duration']:.1f}s")

        # Show progress
        if i < len(musique_files):
            runner.print_progress_summary(results, i, len(musique_files))

    # Save summary and print final results
    total_duration = time.time() - total_start_time
    model_short = MODEL_SHORT_NAMES.get(full_model_name, full_model_name.replace(" ", "_"))
    results_dir = Path(__file__).parent.parent / "tasks_data" / "results" / "musique"

    # Print simple summary
    successful = len([r for r in results if r["status"] == "success"])
    print(f"\n🎉 MuSiQue Processing Completed!")
    print(f"   ✅ Successful: {successful}/{len(results)}")
    print(f"   ⏱️  Total time: {total_duration:.1f}s")

    # Print accuracy summary
    print_musique_accuracy_summary(results_dir, model_short)


if __name__ == "__main__":
    main() 