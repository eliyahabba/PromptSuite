#!/usr/bin/env python3
"""
Language Model Runner
Loads variation files and runs language models on prompt variations.

Features:
- Rate limit handling with exponential backoff
- Batch processing with intermediate saves
- Resume functionality
- Support for multiple platforms (TogetherAI, OpenAI)
- CSV and JSON output formats

Example usage:
python run_language_model.py --batch_size 5 --max_retries 5 --retry_sleep 90
python run_language_model.py --no_resume  # Start fresh
"""

import argparse
import sys
from pathlib import Path

# Add the project root to the path to import multipromptify
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from multipromptify_tasks.constants import (
    LM_DEFAULT_MAX_TOKENS, LM_DEFAULT_PLATFORM, LM_DEFAULT_TEMPERATURE,
    LM_DEFAULT_PARALLEL_WORKERS,
    PLATFORMS, MODEL_SHORT_NAMES
)
from multipromptify_tasks.execution.shared_metrics import (
    calculate_summarization_metrics, calculate_mmlu_correctness_and_metrics,
    calculate_translation_correctness_and_metrics
)
from multipromptify_tasks.execution.batch_runner_base import (
    load_variations_file, filter_variations_by_rows_and_variations,
    run_model_on_variations, get_model_name, load_existing_results
)


def main():
    """Main function to run the language model on variation files."""
    parser = argparse.ArgumentParser(description="Run language model on prompt variations")
    parser.add_argument("--input_folder", help="Input folder containing variation files",
                        default=str(Path(__file__).parent / "tasks_data" / "generated_data" / "data"))
    parser.add_argument("--input_file", help="Input JSON file with variations (e.g., mmlu_local_variations.json)",
                        default="mmlu_local_variations.json")
    parser.add_argument("--platform", choices=list(PLATFORMS.keys()), default=LM_DEFAULT_PLATFORM,
                        help="Platform to use (TogetherAI or OpenAI)")
    parser.add_argument("--model", default="default",
                        help="Model key to use (e.g., 'default', 'gpt_4o_mini', 'llama_3_3_70b')")
    parser.add_argument("--max_tokens", type=int, default=LM_DEFAULT_MAX_TOKENS,
                        help="Maximum tokens for model response")
    parser.add_argument("--temperature", type=float, default=LM_DEFAULT_TEMPERATURE,
                        help="Temperature for model response (default: 0.0)")
    parser.add_argument("--rows", type=int, default=None,
                        help="Maximum number of rows to process (None = all rows)")
    parser.add_argument("--variations", type=int, default=None,
                        help="Maximum variations per row to process (None = all variations)")
    parser.add_argument("--max_retries", type=int, default=3,
                        help="Maximum number of retries for rate limit errors (default: 3)")
    parser.add_argument("--retry_sleep", type=int, default=60,
                        help="Base sleep time in seconds for rate limit retries (default: 60)")
    parser.add_argument("--batch_size", type=int, default=10,
                        help="Number of variations to process before saving intermediate results (default: 10)")
    parser.add_argument("--no_resume", action="store_true",
                        help="Don't resume from existing results file (start fresh)")
    parser.add_argument("--parallel_workers", type=int, default=LM_DEFAULT_PARALLEL_WORKERS,
                        help=f"Number of parallel workers for model calls (1=sequential, default: {LM_DEFAULT_PARALLEL_WORKERS})")
    parser.add_argument("--gold_field", type=str,
                        help="Field name in gold_updates containing the gold answer/label (auto-detected by file type if not specified)")

    args = parser.parse_args()

    # Get the full model name based on platform and model key
    try:
        full_model_name = get_model_name(args.platform, args.model)
    except ValueError as e:
        print(f"❌ {e}")
        return

    input_file = Path(args.input_folder).resolve() / args.input_file
    # Create output filename in main results directory
    input_path = Path(input_file)
    main_dir = Path(__file__).parent.parent  # Go to project root
    results_dir = main_dir / "tasks_data" / "results"

    # Create subdirectory based on input file location
    if "mmlu" in str(input_path):
        results_dir = results_dir / "mmlu"
    elif "translation" in str(input_path):
        results_dir = results_dir / "translation"
    elif "sentiment" in str(input_path):
        results_dir = results_dir / "sentiment"
    elif "summarization" in str(input_path) or "summary" in str(input_path):
        results_dir = results_dir / "summarization"
    elif "qa" in str(input_path) or "question" in str(input_path):
        results_dir = results_dir / "question_answering"

    # Create model-specific subdirectory
    model_short = MODEL_SHORT_NAMES.get(full_model_name, "unknown")
    results_dir = results_dir / model_short
    results_dir.mkdir(parents=True, exist_ok=True)

    # Use original filename without model prefix
    output_file = results_dir / f"{input_path.stem}.json"

    # Auto-detect gold_field if not specified
    if not args.gold_field:
        input_filename = input_path.name.lower()
        if "mmlu" in input_filename:
            args.gold_field = "answer"
        elif "sentiment" in input_filename:
            args.gold_field = "label"
        elif "summarization" in input_filename or "summary" in input_filename:
            args.gold_field = "highlights"
        elif "qa" in input_filename or "question" in input_filename:
            args.gold_field = "answer"
        # For translation, leave as None for auto-detection

    print("🤖 MultiPromptify Language Model Runner")
    print("=" * 50)
    print(f"Input file: {input_file}")
    print(f"Platform: {args.platform}")
    print(f"Model: {full_model_name}")
    print(f"Output file: {output_file}")
    if args.gold_field:
        print(f"Gold field: {args.gold_field}")
    else:
        print("Gold field: auto-detect (translation)")
    print("=" * 50)

    # Load variations
    variations = load_variations_file(input_file)
    if not variations:
        return

    # Filter variations based on row and variation limits
    filtered_variations = filter_variations_by_rows_and_variations(
        variations,
        max_rows=args.rows,
        max_variations_per_row=args.variations
    )

    if not filtered_variations:
        print("❌ No variations to process after filtering")
        return

    # Determine metrics function based on input file type and gold_field
    metrics_function = None
    input_filename = input_path.name.lower()
    
    if "mmlu" in input_filename:
        if args.gold_field:
            def mmlu_metrics_with_field(variation, response):
                return calculate_mmlu_correctness_and_metrics(variation, response, args.gold_field)
            metrics_function = mmlu_metrics_with_field
        else:
            metrics_function = calculate_mmlu_correctness_and_metrics
        print("📊 Using MMLU correctness metrics")
    elif "translation" in input_filename:
        if args.gold_field:
            def translation_metrics_with_field(variation, response):
                return calculate_translation_correctness_and_metrics(variation, response, args.gold_field)
            metrics_function = translation_metrics_with_field
        else:
            metrics_function = calculate_translation_correctness_and_metrics
        print("📊 Using translation metrics (BLEU, ROUGE, SacreBlEU)")
    elif "sentiment" in input_filename:
        if args.gold_field:
            def sentiment_metrics_with_field(variation, response):
                from multipromptify_tasks.execution.shared_metrics import calculate_sentiment_correctness_and_metrics
                return calculate_sentiment_correctness_and_metrics(variation, response, args.gold_field)
            metrics_function = sentiment_metrics_with_field
        else:
            from multipromptify_tasks.execution.shared_metrics import calculate_sentiment_correctness_and_metrics
            metrics_function = calculate_sentiment_correctness_and_metrics
        print("📊 Using sentiment analysis metrics")
    elif "summarization" in input_filename or "summary" in input_filename:
        if args.gold_field:
            def summarization_metrics_with_field(variation, response):
                return calculate_summarization_metrics(variation, response, args.gold_field)
            metrics_function = summarization_metrics_with_field
        else:
            metrics_function = calculate_summarization_metrics
        print("📊 Using summarization metrics (BLEU, ROUGE, SacreBlEU)")
    elif "qa" in input_filename or "question" in input_filename:
        if args.gold_field:
            def qa_metrics_with_field(variation, response):
                from multipromptify_tasks.execution.shared_metrics import calculate_qa_correctness_and_metrics
                return calculate_qa_correctness_and_metrics(variation, response, args.gold_field)
            metrics_function = qa_metrics_with_field
        else:
            from multipromptify_tasks.execution.shared_metrics import calculate_qa_correctness_and_metrics
            metrics_function = calculate_qa_correctness_and_metrics
        print("📊 Using question answering metrics")
    else:
        print("📊 Using default correctness checking")

    # Run model on filtered variations
    run_model_on_variations(
        filtered_variations, full_model_name, args.max_tokens, args.platform, str(output_file),
        temperature=args.temperature, max_retries=args.max_retries, retry_sleep=args.retry_sleep,
        batch_size=args.batch_size, resume=not args.no_resume,
        parallel_workers=args.parallel_workers,
        metrics_function=metrics_function
    )

    print("\n✅ Processing completed!")

    # Load final results and print simple summary
    final_results = load_existing_results(str(output_file))
    if final_results:
        total = len(final_results)
        correct = sum(1 for result in final_results if result.get('is_correct', False))
        accuracy = (correct / total) * 100 if total > 0 else 0.0
        
        print(f"\n📊 Results Summary:")
        print(f"   Total responses: {total}")
        print(f"   ✅ Correct: {correct}")
        print(f"   📈 Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    main()