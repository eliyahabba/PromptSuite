#!/usr/bin/env python3
"""
Translation Batch Runner
Automatically runs language model on all translation language pair variation files.

Example usage:
python run_translation_batch.py --batch_size 5 --max_retries 5
python run_translation_batch.py --model llama_3_3_70b --max_tokens 512
python run_translation_batch.py --no_resume  # Start fresh
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable

from multipromptify_tasks.constants import MODEL_SHORT_NAMES
from multipromptify_tasks.execution.batch_runner_base import BatchRunnerBase
from multipromptify_tasks.execution.run_language_model import get_model_name
from multipromptify_tasks.execution.shared_metrics import calculate_translation_correctness_and_metrics


class TranslationBatchRunner(BatchRunnerBase):
    """Batch runner for translation tasks."""
    
    def __init__(self):
        super().__init__(
            task_name="Translation",
            data_dir_name="translation",
            file_pattern="translation_*_variations.json"
        )
    
    def find_variation_files(self, data_dir: Path) -> List[Path]:
        """Find all translation variation files, excluding WMT14."""
        files = list(data_dir.glob(self.file_pattern))
        # Exclude the original WMT14 file to focus on language pairs
        files = [f for f in files if not f.name.startswith("translation_wmt14")]
        return sorted(files)
    
    def extract_identifier_from_filename(self, filename: str) -> str:
        """Extract language pair from translation filename."""
        if filename.startswith('translation_') and filename.endswith('_variations.json'):
            return filename[12:-16]  # Remove 'translation_' (12 chars) and '_variations.json' (16 chars)
        return filename
    
    def get_display_name(self, identifier: str) -> str:
        """Convert language pair identifier to display name."""
        # Convert 'en-de' to 'English → German'
        lang_map = {
            'en': 'English', 'de': 'German', 'fr': 'French', 'es': 'Spanish',
            'it': 'Italian', 'pt': 'Portuguese', 'ru': 'Russian', 'zh': 'Chinese',
            'ja': 'Japanese', 'ko': 'Korean', 'ar': 'Arabic', 'hi': 'Hindi',
            'cs': 'Czech'
        }
        
        if '-' in identifier:
            source, target = identifier.split('-', 1)
            source_name = lang_map.get(source, source.upper())
            target_name = lang_map.get(target, target.upper())
            return f"{source_name} → {target_name}"
        return identifier
    
    def get_metrics_function(self) -> Optional[Callable]:
        """Return the metrics function for translation tasks."""
        return calculate_translation_correctness_and_metrics

    def create_metrics_function_with_gold_field(self, gold_field: str) -> Optional[Callable]:
        """Create a metrics function that uses the specified gold_field for translation."""
        def translation_metrics_with_field(variation: Dict[str, Any], model_response: str) -> tuple:
            return calculate_translation_correctness_and_metrics(variation, model_response, gold_field)
        return translation_metrics_with_field


def print_translation_summary(results_dir: Path, model_short: str) -> None:
    """Print translation accuracy and metrics summary."""
    model_dir = results_dir / model_short
    if not model_dir.exists():
        print("📊 No translation data available")
        return

    # Find all JSON result files
    json_files = list(model_dir.glob("*.json"))
    
    if not json_files:
        print("📊 No translation data available")
        return

    total_responses = 0
    total_correct = 0
    all_results = []

    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                pair_results = json.load(f)
            all_results.extend(pair_results)
            
            pair_total = len(pair_results)
            pair_correct = sum(1 for result in pair_results if result.get('is_correct', False))
            
            total_responses += pair_total
            total_correct += pair_correct
            
        except Exception as e:
            print(f"⚠️  Error reading {json_file}: {e}")

    if total_responses == 0:
        print("📊 No translation data available")
        return

    overall_accuracy = (total_correct / total_responses * 100)

    print(f"\n📊 Translation Results Summary:")
    print(f"   Total responses: {total_responses}")
    print(f"   ✅ Total correct: {total_correct}")
    print(f"   📈 Overall accuracy: {overall_accuracy:.2f}%")

    # Calculate metrics if available
    if all_results and any('bleu' in result for result in all_results):
        metrics = ['bleu', 'rouge1', 'rouge2', 'rougeL', 'sacrebleu']
        print(f"\n📈 Translation Metrics:")
        
        for metric in metrics:
            scores = [result.get(metric, 0.0) for result in all_results if metric in result]
            if scores:
                avg_score = sum(scores) / len(scores)
                if metric == 'sacrebleu':
                    print(f"   {metric.upper()}: {avg_score:.2f}")
                else:
                    print(f"   {metric.upper()}: {avg_score:.4f}")


def main():
    """Main function to run language model on all translation files."""
    runner = TranslationBatchRunner()
    
    parser = argparse.ArgumentParser(description="Run language model on all translation language pair variations")
    
    # Setup common arguments
    default_data_dir = str(Path(__file__).parent.parent / "tasks_data" / "generated_data" / "translation")
    runner.setup_common_args(parser, default_data_dir)
    
    # Add translation-specific arguments
    parser.add_argument("--pairs", nargs="+",
                        help="Run only specific language pairs (e.g., --pairs en-de fr-en)")
    parser.add_argument("--list_pairs", action="store_true",
                        help="List available language pairs and exit")
    
    # Add gold_field with translation-specific default (None for auto-detect)
    runner.add_gold_field_with_default(parser, None, "Field name in gold_updates containing the translation (default: auto-detect language codes)")

    args = parser.parse_args()

    # Handle list pairs option
    if args.list_pairs:
        translation_dir = Path(args.translation_dir).resolve()
        if not translation_dir.exists():
            print(f"❌ Translation directory not found: {translation_dir}")
            return

        translation_files = runner.find_variation_files(translation_dir)
        if not translation_files:
            print(f"❌ No translation variation files found in: {translation_dir}")
            return

        pairs = [runner.extract_identifier_from_filename(f.name) for f in translation_files]
        pairs.sort()

        print(f"🌍 Available translation pairs ({len(pairs)}):")
        for i, pair in enumerate(pairs, 1):
            display_name = runner.get_display_name(pair)
            print(f"   {i:2d}. {display_name} ({pair})")
        return

    # Get the full model name based on platform and model key
    try:
        full_model_name = get_model_name(args.platform, args.model)
    except ValueError as e:
        print(f"❌ {e}")
        return

    # Find and filter translation files
    translation_dir = Path(args.translation_dir).resolve()
    if not translation_dir.exists():
        print(f"❌ Translation directory not found: {translation_dir}")
        return

    translation_files = runner.find_variation_files(translation_dir)
    if not translation_files:
        print(f"❌ No translation variation files found in: {translation_dir}")
        return

    # Filter pairs if specified
    if args.pairs:
        pairs_to_include = set(args.pairs)
        translation_files = [f for f in translation_files
                           if runner.extract_identifier_from_filename(f.name) in pairs_to_include]
        if not translation_files:
            print(f"❌ No files found for specified pairs: {args.pairs}")
            return


    # Print header and process files
    runner.print_header(args, full_model_name, translation_files)

    # Process files
    results = []
    total_start_time = time.time()

    for i, translation_file in enumerate(translation_files, 1):
        pair = runner.extract_identifier_from_filename(translation_file.name)
        display_name = runner.get_display_name(pair)
        print(f"\n🌍 Processing pair {i}/{len(translation_files)}: {display_name}")

        result = runner.run_language_model_on_file(translation_file, args)
        results.append(result)

        # Print result
        if result["status"] == "success":
            variations_count = result.get("variations_processed", "unknown")
            print(f"✅ {display_name} completed in {result['duration']:.1f}s ({variations_count} variations)")
        else:
            print(f"❌ {display_name} failed: {result.get('error', 'Unknown error')} in {result['duration']:.1f}s")

        # Show progress
        if i < len(translation_files):
            runner.print_progress_summary(results, i, len(translation_files))

    # Save summary and print final results
    total_duration = time.time() - total_start_time
    model_short = MODEL_SHORT_NAMES.get(full_model_name, full_model_name.replace(" ", "_"))
    results_dir = Path(__file__).parent.parent / "tasks_data" / "results" / "translation"

    # Print simple summary
    successful = len([r for r in results if r["status"] == "success"])
    print(f"\n🎉 Translation Processing Completed!")
    print(f"   ✅ Successful: {successful}/{len(results)}")
    print(f"   ⏱️  Total time: {total_duration:.1f}s")

    # Print accuracy summary
    print_translation_summary(results_dir, model_short)


if __name__ == "__main__":
    main()
