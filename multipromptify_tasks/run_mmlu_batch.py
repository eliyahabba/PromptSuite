#!/usr/bin/env python3
"""
MMLU Batch Runner
Automatically runs language model on all MMLU subject variation files.

Example usage:
python run_mmlu_batch.py --batch_size 5 --max_retries 5
python run_mmlu_batch.py --model llama_3_3_70b --max_tokens 512
python run_mmlu_batch.py --no_resume  # Start fresh
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Any
import time
import json

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from multipromptify_tasks.constants import (
    DEFAULT_MAX_TOKENS, DEFAULT_PLATFORM, PLATFORMS, MODELS, MODEL_SHORT_NAMES
)


def get_model_name(platform: str, model_key: str) -> str:
    """Get the full model name based on platform and model key."""
    if platform not in MODELS:
        raise ValueError(f"Unsupported platform: {platform}. Supported platforms: {list(MODELS.keys())}")
    
    platform_models = MODELS[platform]
    if model_key not in platform_models:
        raise ValueError(f"Unsupported model '{model_key}' for platform '{platform}'. Available models: {list(platform_models.keys())}")
    
    return platform_models[model_key]


def find_mmlu_files(mmlu_dir: Path) -> List[Path]:
    """Find all MMLU variation files in the mmlu directory."""
    pattern = "mmlu_*_variations.json"
    files = list(mmlu_dir.glob(pattern))
    return sorted(files)


def extract_subject_from_filename(filename: str) -> str:
    """Extract subject name from MMLU filename."""
    # Remove 'mmlu_' prefix and '_variations.json' suffix
    if filename.startswith('mmlu_') and filename.endswith('_variations.json'):
        return filename[5:-16]  # Remove 'mmlu_' (5 chars) and '_variations.json' (16 chars)
    return filename


def run_language_model_on_file(file_path: Path, args: argparse.Namespace) -> Dict[str, Any]:
    """Run the language model on a single MMLU file."""
    subject = extract_subject_from_filename(file_path.name)
    
    # Build command
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "run_language_model.py"),
        "--input_folder", str(file_path.parent),
        "--input_file", file_path.name,
        "--platform", args.platform,
        "--model", args.model,
        "--max_tokens", str(args.max_tokens),
        "--max_retries", str(args.max_retries),
        "--retry_sleep", str(args.retry_sleep),
        "--batch_size", str(args.batch_size)
    ]
    
    # Add optional flags
    if args.rows is not None:
        cmd.extend(["--rows", str(args.rows)])
    if args.variations is not None:
        cmd.extend(["--variations", str(args.variations)])
    if args.no_resume:
        cmd.append("--no_resume")
    
    print(f"🚀 Running: {' '.join(cmd[2:])}")  # Skip python and script path for cleaner output
    
    start_time = time.time()
    try:
        # Run with real-time output capture
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Capture output in real-time
        stdout_lines = []
        stderr_lines = []
        
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                line = output.strip()
                stdout_lines.append(line)
                # Print progress lines in real-time
                if any(keyword in line for keyword in [
                    "Processing", "Saving batch", "Rate limit", "Error", 
                    "Sleeping", "Resuming", "Loaded", "Filtered"
                ]):
                    print(f"   {line}")
        
        # Get return code
        returncode = process.poll()
        end_time = time.time()
        duration = end_time - start_time
        
        if returncode == 0:
            return {
                "subject": subject,
                "status": "success",
                "duration": duration,
                "stdout": "\n".join(stdout_lines),
                "stderr": "\n".join(stderr_lines)
            }
        else:
            return {
                "subject": subject,
                "status": "failed",
                "duration": duration,
                "returncode": returncode,
                "stdout": "\n".join(stdout_lines),
                "stderr": "\n".join(stderr_lines)
            }
            
    except subprocess.TimeoutExpired:
        process.kill()
        return {
            "subject": subject,
            "status": "timeout",
            "duration": 3600,
            "error": "Process timed out after 1 hour"
        }
    except Exception as e:
        return {
            "subject": subject,
            "status": "error",
            "duration": time.time() - start_time,
            "error": str(e)
        }


def save_batch_summary(results: List[Dict[str, Any]], output_dir: Path, model_short: str) -> None:
    """Save a summary of all batch results."""
    # Create model-specific directory for summary
    model_dir = output_dir / model_short
    model_dir.mkdir(parents=True, exist_ok=True)
    summary_file = model_dir / "mmlu_batch_summary.json"
    
    summary = {
        "total_subjects": len(results),
        "successful": len([r for r in results if r["status"] == "success"]),
        "failed": len([r for r in results if r["status"] == "failed"]),
        "errors": len([r for r in results if r["status"] == "error"]),
        "timeouts": len([r for r in results if r["status"] == "timeout"]),
        "total_duration": sum(r["duration"] for r in results),
        "model": model_short,
        "results": results
    }
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n📊 Batch summary saved to: {summary_file}")
    return summary


def print_progress_summary(results: List[Dict[str, Any]], current: int, total: int) -> None:
    """Print a progress summary."""
    successful = len([r for r in results if r["status"] == "success"])
    failed = len([r for r in results if r["status"] != "success"])
    
    print(f"\n📈 Progress: {current}/{total} subjects (✅{successful} ❌{failed})")
    
    if results:
        avg_duration = sum(r["duration"] for r in results) / len(results)
        remaining = total - current
        estimated_remaining = remaining * avg_duration
        print(f"   ⏱️  ETA: {estimated_remaining/60:.1f} minutes")


def calculate_mmlu_accuracy_stats(results_dir: Path, model_short: str) -> Dict[str, Any]:
    """Calculate accuracy statistics for all MMLU subjects."""
    model_dir = results_dir / model_short
    if not model_dir.exists():
        return {"total_subjects": 0, "total_responses": 0, "overall_accuracy": 0.0}
    
    # Find all JSON result files
    json_files = list(model_dir.glob("*.json"))
    if not json_files:
        return {"total_subjects": 0, "total_responses": 0, "overall_accuracy": 0.0}
    
    total_responses = 0
    total_correct = 0
    subject_accuracies = {}
    
    for json_file in json_files:
        if json_file.name == "mmlu_batch_summary.json":
            continue  # Skip the summary file
            
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
                "total": subject_total,
                "correct": subject_correct,
                "accuracy": subject_accuracy
            }
            
        except Exception as e:
            print(f"⚠️  Error reading {json_file}: {e}")
    
    overall_accuracy = (total_correct / total_responses * 100) if total_responses > 0 else 0.0
    
    return {
        "total_subjects": len(subject_accuracies),
        "total_responses": total_responses,
        "total_correct": total_correct,
        "overall_accuracy": overall_accuracy,
        "subject_accuracies": subject_accuracies
    }


def print_mmlu_accuracy_summary(results_dir: Path, model_short: str) -> None:
    """Print MMLU accuracy summary."""
    stats = calculate_mmlu_accuracy_stats(results_dir, model_short)
    
    if stats["total_subjects"] == 0:
        print("📊 No accuracy data available")
        return
    
    print(f"\n📊 MMLU Accuracy Summary:")
    print(f"   Total subjects: {stats['total_subjects']}")
    print(f"   Total responses: {stats['total_responses']}")
    print(f"   ✅ Total correct: {stats['total_correct']}")
    print(f"   📈 Overall accuracy: {stats['overall_accuracy']:.2f}%")
    
    # Show top and bottom performing subjects
    if stats['subject_accuracies']:
        sorted_subjects = sorted(
            stats['subject_accuracies'].items(),
            key=lambda x: x[1]['accuracy'],
            reverse=True
        )
        
        print(f"\n🏆 Top 5 subjects:")
        for i, (subject, data) in enumerate(sorted_subjects[:5], 1):
            print(f"   {i}. {subject}: {data['accuracy']:.2f}% ({data['correct']}/{data['total']})")
        
        if len(sorted_subjects) > 5:
            print(f"\n📉 Bottom 5 subjects:")
            for i, (subject, data) in enumerate(sorted_subjects[-5:], 1):
                print(f"   {i}. {subject}: {data['accuracy']:.2f}% ({data['correct']}/{data['total']})")


def main():
    """Main function to run language model on all MMLU files."""
    parser = argparse.ArgumentParser(description="Run language model on all MMLU subject variations")
    
    # Input/output options
    parser.add_argument("--mmlu_dir", 
                       default=str(Path(__file__).parent / "data" / "mmlu"),
                       help="Directory containing MMLU variation files")
    
    # Model options
    parser.add_argument("--platform", choices=list(PLATFORMS.keys()), default=DEFAULT_PLATFORM,
                       help="Platform to use (TogetherAI or OpenAI)")
    parser.add_argument("--model", default="default",
                       help="Model key to use (e.g., 'default', 'gpt_4o_mini', 'llama_3_3_70b')")
    parser.add_argument("--max_tokens", type=int, default=DEFAULT_MAX_TOKENS,
                       help="Maximum tokens for model response")
    
    # Processing options
    parser.add_argument("--rows", type=int, default=None,
                       help="Maximum number of rows to process per subject (None = all rows)")
    parser.add_argument("--variations", type=int, default=None,
                       help="Maximum variations per row to process (None = all variations)")
    
    # Retry and batch options
    parser.add_argument("--max_retries", type=int, default=3,
                       help="Maximum number of retries for rate limit errors (default: 3)")
    parser.add_argument("--retry_sleep", type=int, default=60,
                       help="Base sleep time in seconds for rate limit retries (default: 60)")
    parser.add_argument("--batch_size", type=int, default=10,
                       help="Number of variations to process before saving intermediate results (default: 10)")
    
    # Resume options
    parser.add_argument("--no_resume", action="store_true",
                       help="Don't resume from existing results files (start fresh on all subjects)")

    
    # Execution options
    parser.add_argument("--subjects", nargs="+",
                       help="Run only specific subjects (e.g., --subjects anatomy chemistry)")
    parser.add_argument("--exclude", nargs="+",
                       help="Exclude specific subjects (e.g., --exclude anatomy chemistry)")
    parser.add_argument("--dry_run", action="store_true",
                       help="Show what would be run without actually running it")
    parser.add_argument("--list_subjects", action="store_true",
                       help="List available subjects and exit")
    parser.add_argument("--accuracy_only", action="store_true",
                       help="Calculate and display accuracy statistics only (don't run model)")
    
    args = parser.parse_args()
    
    # Handle list subjects option
    if args.list_subjects:
        mmlu_dir = Path(args.mmlu_dir).resolve()
        if not mmlu_dir.exists():
            print(f"❌ MMLU directory not found: {mmlu_dir}")
            return
        
        mmlu_files = find_mmlu_files(mmlu_dir)
        if not mmlu_files:
            print(f"❌ No MMLU variation files found in: {mmlu_dir}")
            print("   Expected files matching pattern: mmlu_*_variations.json")
            print("\n💡 To generate MMLU variations, run:")
            print("   python multipromptify_tasks/tasks/mmlu_task.py --all")
            return
        
        subjects = [extract_subject_from_filename(f.name) for f in mmlu_files]
        subjects.sort()
        
        print(f"📚 Available MMLU subjects ({len(subjects)}):")
        for i, subject in enumerate(subjects, 1):
            print(f"   {i:2d}. {subject}")
        
        print(f"\n💡 To run all subjects:")
        print(f"   python {sys.argv[0]}")
        print(f"\n💡 To run specific subjects:")
        print(f"   python {sys.argv[0]} --subjects {' '.join(subjects[:3])}")
        return
    
    # Handle accuracy only option
    if args.accuracy_only:
        # Get the full model name based on platform and model key
        try:
            full_model_name = get_model_name(args.platform, args.model)
        except ValueError as e:
            print(f"❌ {e}")
            return
        
        model_short = MODEL_SHORT_NAMES.get(full_model_name, full_model_name.replace(" ", "_"))
        results_dir = Path(__file__).parent / "results" / "mmlu"
        
        print(f"📊 Calculating accuracy statistics for model: {full_model_name}")
        print(f"📁 Results directory: {results_dir}")
        
        print_mmlu_accuracy_summary(results_dir, model_short)
        return
    
    # Get the full model name based on platform and model key
    try:
        full_model_name = get_model_name(args.platform, args.model)
    except ValueError as e:
        print(f"❌ {e}")
        return
    
    # Find MMLU files
    mmlu_dir = Path(args.mmlu_dir).resolve()
    if not mmlu_dir.exists():
        print(f"❌ MMLU directory not found: {mmlu_dir}")
        return
    
    mmlu_files = find_mmlu_files(mmlu_dir)
    if not mmlu_files:
        print(f"❌ No MMLU variation files found in: {mmlu_dir}")
        print("   Expected files matching pattern: mmlu_*_variations.json")
        return
    
    # Filter subjects if specified
    if args.subjects:
        subjects_to_include = set(args.subjects)
        mmlu_files = [f for f in mmlu_files 
                     if extract_subject_from_filename(f.name) in subjects_to_include]
        if not mmlu_files:
            print(f"❌ No files found for specified subjects: {args.subjects}")
            return
    
    if args.exclude:
        subjects_to_exclude = set(args.exclude)
        mmlu_files = [f for f in mmlu_files 
                     if extract_subject_from_filename(f.name) not in subjects_to_exclude]
        if not mmlu_files:
            print(f"❌ All files excluded by --exclude: {args.exclude}")
            return
    
    print("🤖 MMLU Batch Language Model Runner")
    print("=" * 60)
    print(f"MMLU directory: {mmlu_dir}")
    print(f"Platform: {args.platform}")
    print(f"Model: {full_model_name}")
    print(f"Max tokens: {args.max_tokens}")
    if args.rows is not None:
        print(f"Max rows per subject: {args.rows}")
    if args.variations is not None:
        print(f"Max variations per row: {args.variations}")
    print(f"Max retries: {args.max_retries}")
    print(f"Retry sleep time: {args.retry_sleep} seconds")
    print(f"Batch size: {args.batch_size}")
    resume_mode = not args.no_resume
    print(f"Resume mode: {resume_mode}")
    print(f"Found {len(mmlu_files)} MMLU subjects to process:")
    
    for i, file in enumerate(mmlu_files, 1):
        subject = extract_subject_from_filename(file.name)
        print(f"  {i:2d}. {subject}")
    
    print("=" * 60)
    
    if args.dry_run:
        print("🏃‍♂️ DRY RUN - No actual processing will be performed")
        return
    
    # Process each file
    results = []
    total_start_time = time.time()
    
    for i, mmlu_file in enumerate(mmlu_files, 1):
        subject = extract_subject_from_filename(mmlu_file.name)
        
        print(f"\n📚 Processing subject {i}/{len(mmlu_files)}: {subject}")
        print("=" * 50)
        
        result = run_language_model_on_file(mmlu_file, args)
        results.append(result)
        
        # Print result
        if result["status"] == "success":
            print(f"✅ {subject} completed successfully in {result['duration']:.1f}s")
        elif result["status"] == "failed":
            print(f"❌ {subject} failed (exit code: {result['returncode']}) in {result['duration']:.1f}s")
            if result.get("stderr"):
                print(f"   Error: {result['stderr'][:200]}...")
        elif result["status"] == "timeout":
            print(f"⏰ {subject} timed out after {result['duration']:.1f}s")
        else:
            print(f"💥 {subject} error: {result.get('error', 'Unknown error')}")
        
        # Show progress
        if i < len(mmlu_files):  # Don't show progress after the last item
            print_progress_summary(results, i, len(mmlu_files))
    
    # Final summary
    total_duration = time.time() - total_start_time
    model_short = MODEL_SHORT_NAMES.get(full_model_name, full_model_name.replace(" ", "_"))
    
    # Save summary in results directory (same place as the actual results)
    results_dir = Path(__file__).parent / "results" / "mmlu"
    summary = save_batch_summary(results, results_dir, model_short)
    
    print(f"\n🎉 MMLU Batch Processing Completed!")
    print("=" * 60)
    print(f"📊 Results Summary:")
    print(f"   Total subjects: {summary['total_subjects']}")
    print(f"   ✅ Successful: {summary['successful']}")
    print(f"   ❌ Failed: {summary['failed']}")
    print(f"   💥 Errors: {summary['errors']}")
    print(f"   ⏰ Timeouts: {summary['timeouts']}")
    print(f"   ⏱️  Total time: {total_duration:.1f}s ({total_duration/60:.1f} minutes)")
    print(f"   ⏱️  Average per subject: {total_duration/len(mmlu_files):.1f}s")
    
    if summary['failed'] > 0 or summary['errors'] > 0 or summary['timeouts'] > 0:
        print(f"\n⚠️  Some subjects failed. Check the batch summary for details.")
        failed_subjects = [r['subject'] for r in results if r['status'] != 'success']
        print(f"   Failed subjects: {', '.join(failed_subjects)}")
    
    print(f"\n📄 Detailed summary saved to: {results_dir}/{model_short}/mmlu_batch_summary.json")
    print("✅ Batch processing completed!")

    # Calculate and print MMLU accuracy summary
    print_mmlu_accuracy_summary(results_dir, model_short)


if __name__ == "__main__":
    main() 