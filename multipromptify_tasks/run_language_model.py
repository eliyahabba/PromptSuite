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

import os
import json
import csv
import sys
import argparse
import time
import re
from pathlib import Path
from typing import List, Dict, Any

# Add the project root to the path to import multipromptify
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from multipromptify.shared.model_client import get_model_response
from multipromptify_tasks.constants import (
    DEFAULT_MAX_TOKENS, DEFAULT_PLATFORM, PLATFORMS, MODELS, MODEL_SHORT_NAMES
)


def load_variations_file(file_path: str) -> List[Dict[str, Any]]:
    """Load variations from a JSON file."""
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            variations = json.load(f)
        print(f"✅ Loaded {len(variations)} variations from {file_path}")
        return variations
    except Exception as e:
        print(f"❌ Error loading file {file_path}: {e}")
        return []


def filter_variations_by_rows_and_variations(variations: List[Dict[str, Any]], 
                                           max_rows: int = None, 
                                           max_variations_per_row: int = None) -> List[Dict[str, Any]]:
    """
    Filter variations based on row and variation limits.
    
    Args:
        variations: List of all variations
        max_rows: Maximum number of rows to process (None = all rows)
        max_variations_per_row: Maximum variations per row (None = all variations)
        
    Returns:
        Filtered list of variations
    """
    if max_rows is None and max_variations_per_row is None:
        return variations
    
    # Group variations by original row index
    row_groups = {}
    for variation in variations:
        row_idx = variation.get('original_row_index', 0)
        if row_idx not in row_groups:
            row_groups[row_idx] = []
        row_groups[row_idx].append(variation)
    
    # Sort rows and limit them
    sorted_rows = sorted(row_groups.keys())
    if max_rows is not None:
        sorted_rows = sorted_rows[:max_rows]
    
    # Filter variations
    filtered_variations = []
    for row_idx in sorted_rows:
        row_variations = row_groups[row_idx]
        if max_variations_per_row is not None:
            row_variations = row_variations[:max_variations_per_row]
        filtered_variations.extend(row_variations)
    
    print(f"🔍 Filtered to {len(filtered_variations)} variations from {len(row_groups)} rows")
    return filtered_variations


def is_rate_limit_error(error_message: str) -> bool:
    """Check if the error is a rate limit error."""
    rate_limit_indicators = [
        "rate limit",
        "Error code: 429",
        "model_rate_limit",
        "maximum rate limit",
        "queries per minute"
    ]
    error_lower = error_message.lower()
    return any(indicator in error_lower for indicator in rate_limit_indicators)


def get_model_response_with_retry(conversation: List[Dict[str, Any]], 
                                model_name: str, 
                                max_tokens: int, 
                                platform: str,
                                max_retries: int = 3,
                                base_sleep_time: int = 60) -> str:
    """
    Get model response with retry logic for rate limit errors.
    
    Args:
        conversation: The conversation to send to the model
        model_name: Name of the model to use
        max_tokens: Maximum tokens for response
        platform: Platform to use
        max_retries: Maximum number of retries for rate limit errors
        base_sleep_time: Base sleep time in seconds for rate limit errors
        
    Returns:
        Model response string
        
    Raises:
        Exception: If all retries are exhausted or non-rate-limit error occurs
    """
    for attempt in range(max_retries + 1):
        try:
            return get_model_response(conversation, model_name, max_tokens=max_tokens, platform=platform)
        except Exception as e:
            error_message = str(e)
            
            # Check if this is a rate limit error
            if is_rate_limit_error(error_message):
                if attempt < max_retries:
                    sleep_time = base_sleep_time * (attempt + 1)
                    print(f"⏳ Rate limit hit (attempt {attempt + 1}/{max_retries + 1}). Sleeping {sleep_time}s...")
                    time.sleep(sleep_time)
                    continue
                else:
                    print(f"❌ Rate limit error persists after {max_retries} retries")
                    raise e
            else:
                # Non-rate-limit error, don't retry
                raise e
    
    # This should never be reached, but just in case
    raise Exception("Unexpected error in retry logic")


def load_existing_results(output_file: str) -> List[Dict[str, Any]]:
    """Load existing results from output file if it exists."""
    if not os.path.exists(output_file):
        return []
    
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        print(f"📂 Loaded {len(results)} existing results from {output_file}")
        return results
    except Exception as e:
        print(f"⚠️  Error loading existing results: {e}")
        return []


def get_processed_variation_indices(results: List[Dict[str, Any]]) -> set:
    """Get set of (original_row_index, variation_index) tuples that have already been processed."""
    processed = set()
    for result in results:
        if 'variation_index' in result:
            # Use combination of original_row_index and variation_index as unique identifier
            row_idx = result.get('original_row_index', 0)
            var_idx = result['variation_index']
            processed.add((row_idx, var_idx))
    return processed


def save_batch_results(results: List[Dict[str, Any]], output_file: str) -> None:
    """Save results to JSON file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Also save CSV
    csv_file = str(output_file).replace('.json', '.csv')
    save_results_as_csv(results, csv_file)


def get_model_name(platform: str, model_key: str) -> str:
    """Get the full model name based on platform and model key."""
    if platform not in MODELS:
        raise ValueError(f"Unsupported platform: {platform}. Supported platforms: {list(MODELS.keys())}")
    
    platform_models = MODELS[platform]
    if model_key not in platform_models:
        raise ValueError(f"Unsupported model '{model_key}' for platform '{platform}'. Available models: {list(platform_models.keys())}")
    
    return platform_models[model_key]


def run_model_on_variations(variations: List[Dict[str, Any]], 
                           model_name: str,
                           max_tokens: int,
                           platform: str,
                           output_file: str,
                           max_retries: int = 3,
                           retry_sleep: int = 60,
                           batch_size: int = 10,
                           resume: bool = True,
                           ) -> None:
    """Run the language model on variations and save results."""
    print(f"🤖 Using model: {model_name}")
    print(f"📝 Max tokens: {max_tokens}")
    print(f"🌐 Platform: {platform}")
    print(f"📦 Batch size: {batch_size}")
    print(f"🔄 Resume mode: {resume}")
    if not resume:
        print(f"🆕 Starting fresh (not resuming)")
    
    # Load existing results if resume mode is enabled
    results = []
    processed_indices = set()
    if resume:
        results = load_existing_results(output_file)
        if results:
            print(f"🔄 Resuming from existing results file...")
            processed_indices = get_processed_variation_indices(results)
            if processed_indices:
                print(f"📋 Found {len(processed_indices)} already processed variations")
        else:
            print(f"📄 No existing results found, starting fresh...")
    
    # Filter out already processed variations
    variations_to_process = []
    for variation in variations:
        row_idx = variation.get('original_row_index', 0)
        variation_index = variation.get('variation_count')
        variation_key = (row_idx, variation_index)
        if variation_key not in processed_indices:
            variations_to_process.append(variation)
    
    if not variations_to_process:
        print("✅ All variations already processed!")
        return
    
    print(f"🔄 Processing {len(variations_to_process)} remaining variations (out of {len(variations)} total)")
    
    batch_count = 0
    
    for i, variation in enumerate(variations_to_process, 1):
        try:
            # Get conversation from variation
            conversation = variation.get('conversation', [])
            if not conversation:
                print(f"⚠️  Skipping variation {i}: No conversation found")
                continue
            
            print(f"Processing {i}/{len(variations_to_process)} (variation {variation.get('variation_count')})")
            
            # Run the model with conversation format, max_tokens, and platform (with retry logic)
            response = get_model_response_with_retry(
                conversation, model_name, max_tokens=max_tokens, platform=platform,
                max_retries=max_retries, base_sleep_time=retry_sleep
            )
            
            # Extract gold answer and check correctness
            gold_answer_text, is_correct = extract_gold_answer_and_check_correctness(variation, response)
            
            # Create result entry with model information
            result = {
                'variation_index': variation['variation_count'],
                'original_row_index': variation.get('original_row_index', 'Unknown'),
                'model_response': response,
                'model_name': model_name,  # Add model name to results
                'gold_answer': gold_answer_text,
                'is_correct': is_correct,
                'conversation': conversation,
                'template_config': variation.get('template_config', {})
            }
            
            results.append(result)
            
        except Exception as e:
            print(f"❌ Error processing variation {i}: {e}")
            # Extract gold answer even for errors
            gold_answer_text, is_correct = extract_gold_answer_and_check_correctness(variation, f"ERROR: {str(e)}")
            
            # Add error entry with model information
            results.append({
                'variation_index': variation.get('variation_count', i),
                'original_row_index': variation.get('original_row_index', 'Unknown'),
                'model_response': f"ERROR: {str(e)}",
                'model_name': model_name,  # Add model name to results
                'gold_answer': gold_answer_text,
                'is_correct': is_correct,
                'conversation': variation.get('conversation', []),
                'template_config': variation.get('template_config', {})
            })
        
        # Save batch results
        if len(results) % batch_size == 0 or i == len(variations_to_process):
            batch_count += 1
            progress_pct = (i / len(variations_to_process)) * 100
            print(f"💾 Saving batch {batch_count} ({len(results)} total results, {progress_pct:.1f}% complete)...")
            save_batch_results(results, output_file)
            print(f"✅ Batch {batch_count} saved successfully")
    
    # Final save
    print(f"💾 Results saved to: {output_file}")
    csv_file = str(output_file).replace('.json', '.csv')
    print(f"📊 CSV saved to: {csv_file}")
    print(f"📊 Total processed: {len(results)} variations")


def save_results_as_csv(results: List[Dict[str, Any]], csv_file: str) -> None:
    """Save results as CSV with minimal information in logical column order."""
    # Define columns in logical order
    columns = [
        'variation_index',
        'original_row_index', 
        'model_name',
        'model_response',
        'gold_answer',
        'is_correct'
    ]
    
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        
        for result in results:
            # Extract only the columns we want for CSV
            csv_row = {
                'variation_index': result.get('variation_index', ''),
                'original_row_index': result.get('original_row_index', ''),
                'model_name': result.get('model_name', ''),
                'model_response': result.get('model_response', ''),
                'gold_answer': result.get('gold_answer', ''),
                'is_correct': result.get('is_correct', False)
            }
            writer.writerow(csv_row)


def extract_gold_answer_and_check_correctness(variation: Dict[str, Any], model_response: str) -> tuple:
    """
    Extract the correct answer from choices and check if model response is correct.
    
    Args:
        variation: The variation dictionary containing gold_updates and configuration
        model_response: The model's response string
        
    Returns:
        tuple: (gold_answer_text, is_correct)
    """
    try:
        # Get the gold answer index
        gold_index = variation.get('gold_updates', {}).get('answer')
        if gold_index is None:
            return "No gold answer", False
        
        # Get the choices from field_values
        field_values = variation.get('configuration', {}).get('field_values', {})
        choices_str = field_values.get('choices', '')
        
        if not choices_str:
            return "No choices found", False
        
        # Split choices by lines and clean them
        choices = [choice.strip() for choice in choices_str.split('\n') if choice.strip()]
        
        # Get the correct answer by index
        try:
            gold_index_int = int(gold_index)
            if 0 <= gold_index_int < len(choices):
                gold_answer_text = choices[gold_index_int]
            else:
                return f"Invalid index {gold_index}", False
        except (ValueError, TypeError):
            return f"Invalid gold index: {gold_index}", False
        
        # Check if model response is correct
        # Clean the model response and compare
        model_response_clean = model_response.strip().lower()
        gold_answer_clean = gold_answer_text.strip().lower()
        
        # Check for exact match or if model response contains the gold answer
        is_correct = (model_response_clean == gold_answer_clean or 
                     gold_answer_clean in model_response_clean or
                     model_response_clean in gold_answer_clean)
        
        return gold_answer_text, is_correct
        
    except Exception as e:
        return f"Error extracting gold answer: {str(e)}", False


def calculate_accuracy_stats(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate accuracy statistics from results."""
    total = len(results)
    if total == 0:
        return {"total": 0, "correct": 0, "incorrect": 0, "accuracy": 0.0}
    
    correct = sum(1 for result in results if result.get('is_correct', False))
    incorrect = total - correct
    accuracy = (correct / total) * 100 if total > 0 else 0.0
    
    return {
        "total": total,
        "correct": correct,
        "incorrect": incorrect,
        "accuracy": accuracy
    }


def print_accuracy_summary(results: List[Dict[str, Any]]) -> None:
    """Print accuracy summary."""
    stats = calculate_accuracy_stats(results)
    
    print(f"\n📊 Accuracy Summary:")
    print(f"   Total responses: {stats['total']}")
    print(f"   ✅ Correct: {stats['correct']}")
    print(f"   ❌ Incorrect: {stats['incorrect']}")
    print(f"   📈 Accuracy: {stats['accuracy']:.2f}%")


def main():
    """Main function to run the language model on variation files."""
    parser = argparse.ArgumentParser(description="Run language model on prompt variations")
    parser.add_argument("--input_folder", help="Input folder containing variation files",
                        default=str(Path(__file__).parent / "data"))
    parser.add_argument("--input_file", help="Input JSON file with variations (e.g., mmlu_local_variations.json)", default="mmlu_local_variations.json")
    parser.add_argument("--platform", choices=list(PLATFORMS.keys()), default=DEFAULT_PLATFORM,
                       help="Platform to use (TogetherAI or OpenAI)")
    parser.add_argument("--model", default="default",
                       help="Model key to use (e.g., 'default', 'gpt_4o_mini', 'llama_3_3_70b')")
    parser.add_argument("--max_tokens", type=int, default=DEFAULT_MAX_TOKENS,
                       help="Maximum tokens for model response")
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
    # Get the main multipromptify_tasks directory
    main_dir = Path(__file__).parent
    results_dir = main_dir / "results"
    
    # Create subdirectory based on input file location
    if "mmlu" in str(input_path):
        results_dir = results_dir / "mmlu"
    
    # Create model-specific subdirectory
    model_short = MODEL_SHORT_NAMES.get(full_model_name, "unknown")
    results_dir = results_dir / model_short
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Use original filename without model prefix
    output_file = results_dir / f"{input_path.stem}.json"
    
    print("🤖 MultiPromptify Language Model Runner")
    print("=" * 50)
    print(f"Input file: {input_file}")
    print(f"Platform: {args.platform}")
    print(f"Model: {full_model_name}")
    print(f"Max tokens: {args.max_tokens}")
    if args.rows is not None:
        print(f"Max rows: {args.rows}")
    if args.variations is not None:
        print(f"Max variations per row: {args.variations}")
    print(f"Max retries for rate limits: {args.max_retries}")
    print(f"Retry sleep time: {args.retry_sleep} seconds")
    print(f"Batch size: {args.batch_size}")
    resume_mode = not args.no_resume
    print(f"Resume mode: {resume_mode}")
    if not resume_mode:
        print(f"Starting fresh (not resuming)")
    print(f"Output file: {output_file}")
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
    
    # Run model on filtered variations
    run_model_on_variations(
        filtered_variations, full_model_name, args.max_tokens, args.platform, str(output_file),
        max_retries=args.max_retries, retry_sleep=args.retry_sleep,
        batch_size=args.batch_size, resume=resume_mode
    )
    
    print("\n✅ Processing completed!")
    
    # Load final results and print accuracy summary
    final_results = load_existing_results(str(output_file))
    if final_results:
        print_accuracy_summary(final_results)


if __name__ == "__main__":
    main() 