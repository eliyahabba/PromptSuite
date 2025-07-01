#!/usr/bin/env python3
"""
GPQA Extracted Answer Evaluation Script

This script adds an extracted answer column to GPQA results using the provided
extract_answer function, and calculates accuracy based on the extracted answers.

Usage:
    python evaluate_gpqa_extracted_answers.py [--model MODEL_NAME] [--file SPECIFIC_FILE] [--input_dir INPUT_DIR]

Examples:
    # Process default model (gpt_4o_mini)
    python evaluate_gpqa_extracted_answers.py

    # Process specific model
    python evaluate_gpqa_extracted_answers.py --model llama_3_3_70b

    # Process specific file
    python evaluate_gpqa_extracted_answers.py --file /path/to/gpqa_results.csv

    # Process specific directory
    python evaluate_gpqa_extracted_answers.py --input_dir /path/to/results/dir
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from tqdm import tqdm


def extract_answer(text: str, original_row_index: int, variation_index: int) -> Optional[str]:
    """
    Extract answer choice from text by looking for valid choice patterns.

    Args:
        text: The full response text
        original_row_index: The original row index (for debugging)
        variation_index: The variation index (for debugging)

    Returns:
        The extracted answer choice (e.g., 'A', '3', 'ב')
    """
    # Split into lines
    lines = text.strip().split('\n')

    # Define valid choices (first 5 of each type)
    valid_choices = [
        '1', '2', '3', '4', '5',
        'A', 'B', 'C', 'D', 'E',
        'a', 'b', 'c', 'd', 'e',
        'α', 'β', 'γ', 'δ', 'ε',
        'I', 'II', 'III', 'IV', 'V'
    ]

    # Create pattern: optional non-letter chars + choice + dot + space
    # Examples: **A. , :1. , ##B. , etc.
    choices_pattern = '|'.join(re.escape(choice) for choice in valid_choices)
    pattern = rf'(?:^|[^a-zA-Z])({choices_pattern})\.(?:\s|$)'

    # Search from bottom to top
    for i in range(len(lines) - 1, -1, -1):
        line = lines[i]
        match = re.search(pattern, line)
        if match:
            return match.group(1)
    
    # If no match found, return None (uncomment for debugging)
    # print(f"Warning: No valid answer found in row {original_row_index}, variation {variation_index}.")
    return None


def normalize_answer(answer: str) -> str:
    """
    Normalize answer for comparison.
    
    Args:
        answer: The answer string to normalize
        
    Returns:
        Normalized answer string
    """
    if not answer:
        return ""
    
    # Extract just the choice part from formats like "3. 11", "D. 11", etc.
    answer_str = str(answer).strip()
    
    # If the answer contains a dot, take only the part before the dot
    if '.' in answer_str:
        choice_part = answer_str.split('.')[0].strip()
    else:
        choice_part = answer_str
    
    return choice_part


def calculate_extracted_accuracy(data: List[Dict]) -> Dict[str, float]:
    """
    Calculate accuracy based on extracted answers.
    
    Args:
        data: List of result dictionaries
        
    Returns:
        Dictionary with accuracy metrics
    """
    total_samples = len(data)
    correct_extracted = 0
    extracted_answers_found = 0
    valid_gold_answers = 0
    
    for item in data:
        extracted_answer = item.get('extracted_answer')
        parsed_gold_answer = item.get('parsed_gold_answer')
        
        # Count valid gold answers
        if parsed_gold_answer is not None and str(parsed_gold_answer).lower() != 'nan':
            valid_gold_answers += 1
        
        # Count extracted answers
        if extracted_answer is not None and str(extracted_answer).lower() != 'nan':
            extracted_answers_found += 1
            
            # Check if correct (using the pre-calculated field)
            if item.get('extracted_is_correct', False):
                correct_extracted += 1
    
    extraction_rate = extracted_answers_found / total_samples if total_samples > 0 else 0
    extracted_accuracy = correct_extracted / extracted_answers_found if extracted_answers_found > 0 else 0
    overall_accuracy = correct_extracted / total_samples if total_samples > 0 else 0
    
    return {
        'total_samples': total_samples,
        'valid_gold_answers': valid_gold_answers,
        'extracted_answers_found': extracted_answers_found,
        'extraction_rate': extraction_rate,
        'correct_extracted': correct_extracted,
        'extracted_accuracy': extracted_accuracy,
        'overall_accuracy': overall_accuracy
    }


def add_extracted_answers(data: List[Dict]) -> List[Dict]:
    """
    Add extracted answer column to the data.
    
    Args:
        data: List of result dictionaries
        
    Returns:
        List of dictionaries with added extracted_answer column
    """
    print("Extracting answers from model responses...")
    
    for item in tqdm(data, desc="Processing responses"):
        model_response = item.get('model_response', '')
        original_row_index = item.get('original_row_index', 0)
        variation_index = item.get('variation_index', 0)
        
        # Extract answer using the provided function
        extracted_answer = extract_answer(model_response, original_row_index, variation_index)
        item['extracted_answer'] = extracted_answer
        
        # Parse the gold answer using the same normalization logic
        gold_answer = item.get('gold_answer')
        if gold_answer is not None:
            item['parsed_gold_answer'] = normalize_answer(gold_answer)
        else:
            item['parsed_gold_answer'] = None
        
        # Calculate if extracted answer is correct
        # Both extracted_answer and parsed_gold_answer must not be None/nan
        if (extracted_answer is not None and 
            item['parsed_gold_answer'] is not None and 
            str(extracted_answer).lower() != 'nan' and 
            str(item['parsed_gold_answer']).lower() != 'nan'):
            item['extracted_is_correct'] = normalize_answer(extracted_answer) == item['parsed_gold_answer']
        else:
            item['extracted_is_correct'] = False
    
    return data


def find_gpqa_files(input_dir: Path) -> List[Path]:
    """
    Find all GPQA CSV files in the input directory, excluding already processed ones.
    
    Args:
        input_dir: Directory to search for files
        
    Returns:
        List of Path objects for GPQA files
    """
    all_files = list(input_dir.glob("gpqa_*.csv"))
    # Filter out files that already have extracted answers
    return [f for f in all_files if not f.name.endswith("_with_extracted.csv")]


def main():
    parser = argparse.ArgumentParser(
        description="Add extracted answer evaluation to GPQA results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split('Usage:')[1] if 'Usage:' in __doc__ else ""
    )
    
    parser.add_argument(
        "--model", 
        type=str, 
        default="gpt_4o_mini",
        help="Model name to process (default: gpt_4o_mini)"
    )
    
    parser.add_argument(
        "--file", 
        type=str,
        help="Specific CSV file to process (overrides model-based selection)"
    )
    
    parser.add_argument(
        "--input_dir", 
        type=str,
        help="Input directory containing GPQA results (overrides model-based selection)"
    )
    
    args = parser.parse_args()
    
    # Determine input file/directory
    if args.file:
        input_file = Path(args.file)
        if not input_file.exists():
            print(f"Error: File {input_file} not found")
            return
        files_to_process = [input_file]
        
    elif args.input_dir:
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            print(f"Error: Directory {input_dir} not found")
            return
        files_to_process = find_gpqa_files(input_dir)
        if not files_to_process:
            print(f"No GPQA CSV files found in {input_dir}")
            return
            
    else:
        # Use model-based path
        tasks_data_dir = Path(__file__).parent.parent / "tasks_data"
        model_dir = tasks_data_dir / "results" / "gpqa" / args.model
        
        if not model_dir.exists():
            available_models = [d.name for d in (tasks_data_dir / "results" / "gpqa").iterdir() if d.is_dir()]
            print(f"Error: Model directory {model_dir} not found")
            print(f"Available models: {', '.join(available_models)}")
            return
            
        files_to_process = find_gpqa_files(model_dir)
        if not files_to_process:
            print(f"No GPQA CSV files found in {model_dir}")
            return
    
    # Process each file
    for input_file in files_to_process:
        print(f"\nProcessing: {input_file}")
        
        # Load data
        print("Loading data...")
        try:
            df = pd.read_csv(input_file)
            data = df.to_dict('records')
            print(f"Loaded {len(data)} samples")
        except Exception as e:
            print(f"Error loading {input_file}: {e}")
            continue
        
        # Add extracted answers
        data_with_extracted = add_extracted_answers(data)
        
        # Calculate metrics
        metrics = calculate_extracted_accuracy(data_with_extracted)
        
        # Create output paths
        output_file_csv = input_file.parent / f"{input_file.stem}.csv"
        output_file_json = input_file.parent / f"{input_file.stem}.json"
        
        # Save results
        print("Saving results...")
        
        # Save as CSV
        df_output = pd.DataFrame(data_with_extracted)
        df_output.to_csv(output_file_csv, index=False)
        
        # Save as JSON
        with open(output_file_json, 'w', encoding='utf-8') as f:
            json.dump(data_with_extracted, f, indent=2, ensure_ascii=False)
        
        # Print summary
        print(f"\nEvaluation Summary for {input_file.name}:")
        print(f"Total samples: {metrics['total_samples']}")
        print(f"Valid gold answers: {metrics['valid_gold_answers']}")
        print(f"Extracted answers found: {metrics['extracted_answers_found']}")
        print(f"Extraction rate: {metrics['extraction_rate']:.2%}")
        print(f"Correct extracted answers: {metrics['correct_extracted']}")
        print(f"Extracted answer accuracy: {metrics['extracted_accuracy']:.2%}")
        print(f"Overall accuracy: {metrics['overall_accuracy']:.2%}")
        
        print(f"\nFiles saved:")
        print(f"CSV: {output_file_csv}")
        print(f"JSON: {output_file_json}")


if __name__ == "__main__":
    main() 