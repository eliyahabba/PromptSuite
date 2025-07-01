#!/usr/bin/env python3
"""
MMLU Batch Runner
Automatically runs language model on all MMLU subject variation files.

Example usage:
python run_mmlu_batch.py --batch_size 5 --max_retries 5
python run_mmlu_batch.py --model llama_3_3_70b --max_tokens 512
python run_mmlu_batch.py --no_resume  # Start fresh
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, Callable

from multipromptify_tasks.constants import MODEL_SHORT_NAMES
from multipromptify_tasks.execution.batch_runner_base import BatchRunnerBase
from multipromptify_tasks.execution.run_language_model import get_model_name
from multipromptify_tasks.execution.shared_metrics import calculate_mmlu_correctness_and_metrics


class MMLUBatchRunner(BatchRunnerBase):
    """Batch runner for MMLU tasks."""

    def __init__(self):
        super().__init__(
            task_name="MMLU",
            data_dir_name="mmlu",
            file_pattern="mmlu_*_variations.json"
        )

    def extract_identifier_from_filename(self, filename: str) -> str:
        """Extract subject name from MMLU filename."""
        if filename.startswith('mmlu_') and filename.endswith('_variations.json'):
            return filename[5:-16]  # Remove 'mmlu_' (5 chars) and '_variations.json' (16 chars)
        return filename

    def get_metrics_function(self) -> Optional[Callable]:
        """Return the metrics function for MMLU multiple choice tasks."""
        return calculate_mmlu_correctness_and_metrics

    def create_metrics_function_with_gold_field(self, gold_field: str) -> Optional[Callable]:
        """Create a metrics function that uses the specified gold_field for MMLU."""
        def mmlu_metrics_with_field(variation: Dict[str, Any], model_response: str) -> tuple:
            return calculate_mmlu_correctness_and_metrics(variation, model_response, gold_field)
        return mmlu_metrics_with_field

    def create_result_dict(self, identifier: str, status: str, duration: float,
                           variations_processed: int = None, output_file: str = None,
                           error: str = None) -> Dict[str, Any]:
        """Create a result dictionary with MMLU-specific fields."""
        result = {
            "subject": identifier,  # MMLU uses 'subject' instead of 'identifier'
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


def print_mmlu_accuracy_summary(results_dir: Path, model_short: str) -> None:
    """Print MMLU accuracy summary."""
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
    subject_accuracies = {}

    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                subject_results = json.load(f)

            subject_name = json_file.stem
            subject_total = len(subject_results)
            subject_correct = sum(1 for result in subject_results if result.get('is_correct', False))
            subject_accuracy = (subject_correct / subject_total * 100) if subject_total > 0 else 0.0

            total_responses += subject_total
            total_correct += subject_correct
            subject_accuracies[subject_name] = {
                "accuracy": subject_accuracy,
                "correct": subject_correct,
                "total": subject_total
            }
        except Exception as e:
            print(f"⚠️  Error reading {json_file}: {e}")

    if total_responses == 0:
        print("📊 No accuracy data available")
        return

    overall_accuracy = (total_correct / total_responses * 100)

    print(f"\n📊 MMLU Results Summary:")
    print(f"   Total subjects: {len(subject_accuracies)}")
    print(f"   Total responses: {total_responses}")
    print(f"   ✅ Total correct: {total_correct}")
    print(f"   📈 Overall accuracy: {overall_accuracy:.2f}%")

    # Show top and bottom performing subjects
    if subject_accuracies:
        sorted_subjects = sorted(subject_accuracies.items(), key=lambda x: x[1]['accuracy'], reverse=True)

        print(f"\n🏆 Top performing subjects:")
        for i, (subject, data) in enumerate(sorted_subjects[:3], 1):
            print(f"   {i}. {subject}: {data['accuracy']:.2f}% ({data['correct']}/{data['total']})")

        if len(sorted_subjects) > 3:
            print(f"\n📉 Bottom performing subjects:")
            for i, (subject, data) in enumerate(sorted_subjects[-3:], 1):
                print(f"   {i}. {subject}: {data['accuracy']:.2f}% ({data['correct']}/{data['total']})")


def main():
    """Main function to run language model on all MMLU files."""
    runner = MMLUBatchRunner()

    parser = argparse.ArgumentParser(description="Run language model on all MMLU subject variations")

    # Setup common arguments
    default_data_dir = str(Path(__file__).parent.parent / "tasks_data" / "generated_data" / "mmlu")
    runner.setup_common_args(parser, default_data_dir)

    # Add MMLU-specific arguments
    parser.add_argument("--subjects", nargs="+",
                        help="Run only specific subjects (e.g., --subjects anatomy chemistry)")
    parser.add_argument("--exclude", nargs="+",
                        help="Exclude specific subjects (e.g., --exclude anatomy chemistry)")
    parser.add_argument("--list_subjects", action="store_true",
                        help="List available subjects and exit")
    
    # Add gold_field with MMLU-specific default
    runner.add_gold_field_with_default(parser, "answer", "Field name in gold_updates containing the correct answer index (default: 'answer')")

    args = parser.parse_args()

    # Handle list subjects option
    if args.list_subjects:
        mmlu_dir = Path(args.mmlu_dir).resolve()
        if not mmlu_dir.exists():
            print(f"❌ MMLU directory not found: {mmlu_dir}")
            return

        mmlu_files = runner.find_variation_files(mmlu_dir)
        if not mmlu_files:
            print(f"❌ No MMLU variation files found in: {mmlu_dir}")
            print("   Expected files matching pattern: mmlu_*_variations.json")
            return

        subjects = [runner.extract_identifier_from_filename(f.name) for f in mmlu_files]
        subjects.sort()

        print(f"📚 Available MMLU subjects ({len(subjects)}):")
        for i, subject in enumerate(subjects, 1):
            print(f"   {i:2d}. {subject}")
        return

    # Get the full model name based on platform and model key
    try:
        full_model_name = get_model_name(args.platform, args.model)
    except ValueError as e:
        print(f"❌ {e}")
        return

    # Find and filter MMLU files
    mmlu_dir = Path(args.mmlu_dir).resolve()
    if not mmlu_dir.exists():
        print(f"❌ MMLU directory not found: {mmlu_dir}")
        return

    mmlu_files = runner.find_variation_files(mmlu_dir)
    if not mmlu_files:
        print(f"❌ No MMLU variation files found in: {mmlu_dir}")
        return

    # Filter subjects if specified
    if args.subjects:
        subjects_to_include = set(args.subjects)
        mmlu_files = [f for f in mmlu_files
                      if runner.extract_identifier_from_filename(f.name) in subjects_to_include]
        if not mmlu_files:
            print(f"❌ No files found for specified subjects: {args.subjects}")
            return

    if args.exclude:
        subjects_to_exclude = set(args.exclude)
        mmlu_files = [f for f in mmlu_files
                      if runner.extract_identifier_from_filename(f.name) not in subjects_to_exclude]
        if not mmlu_files:
            print(f"❌ All files excluded by --exclude: {args.exclude}")
            return

    # Print header and process files
    runner.print_header(args, full_model_name, mmlu_files)

    # Process files
    results = []
    total_start_time = time.time()

    for i, mmlu_file in enumerate(mmlu_files, 1):
        subject = runner.extract_identifier_from_filename(mmlu_file.name)
        print(f"\n📚 Processing subject {i}/{len(mmlu_files)}: {subject}")

        result = runner.run_language_model_on_file(mmlu_file, args)
        results.append(result)

        # Print result
        if result["status"] == "success":
            variations_count = result.get("variations_processed", "unknown")
            print(f"✅ {subject} completed in {result['duration']:.1f}s ({variations_count} variations)")
        else:
            print(f"❌ {subject} failed: {result.get('error', 'Unknown error')} in {result['duration']:.1f}s")

        # Show progress
        if i < len(mmlu_files):
            runner.print_progress_summary(results, i, len(mmlu_files))

    # Save summary and print final results
    total_duration = time.time() - total_start_time
    model_short = MODEL_SHORT_NAMES.get(full_model_name, full_model_name.replace(" ", "_"))
    results_dir = Path(__file__).parent.parent / "tasks_data" / "results" / "mmlu"

    # Print simple summary
    successful = len([r for r in results if r["status"] == "success"])
    print(f"\n🎉 MMLU Processing Completed!")
    print(f"   ✅ Successful: {successful}/{len(results)}")
    print(f"   ⏱️  Total time: {total_duration:.1f}s")

    # Print accuracy summary
    print_mmlu_accuracy_summary(results_dir, model_short)


if __name__ == "__main__":
    main()
