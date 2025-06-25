# MMLU Batch Processing System

This system allows you to process MMLU (Massive Multitask Language Understanding) datasets by subject, making it easier to manage and run language models on specific domains.

## Overview

The system consists of two main components:

1. **Subject-wise variation generation** (`mmlu_task.py`)
2. **Batch language model runner** (`run_mmlu_batch.py`)

## Quick Start

### 1. Generate MMLU Variations by Subject

First, generate prompt variations for all MMLU subjects:

```bash
# Generate variations for all subjects
python multipromptify_tasks/tasks/mmlu_task.py --all

# Or generate for a specific subject
python multipromptify_tasks/tasks/mmlu_task.py --subject anatomy

# List available subjects
python multipromptify_tasks/tasks/mmlu_task.py --list-subjects
```

This will create files in `multipromptify_tasks/data/mmlu/`:
- `mmlu_anatomy_variations.json`
- `mmlu_chemistry_variations.json`
- etc.

### 2. Run Language Model on All Subjects

```bash
# Run on all subjects with default settings
python multipromptify_tasks/run_mmlu_batch.py

# See what would be run (dry run)
python multipromptify_tasks/run_mmlu_batch.py --dry_run

# Run on specific subjects only
python multipromptify_tasks/run_mmlu_batch.py --subjects anatomy chemistry

# List available subjects
python multipromptify_tasks/run_mmlu_batch.py --list_subjects
```

## Available Subjects

The system currently supports these MMLU subjects:

1. anatomy
2. college_chemistry  
3. college_computer_science
4. college_mathematics
5. electrical_engineering
6. elementary_mathematics
7. global_facts
8. medical_genetics
9. professional_accounting
10. sociology

## Key Features

### Subject-wise Processing
- Each subject is processed independently
- Results are saved separately for each subject
- Easy to focus on specific domains

### Robust Error Handling
- Automatic retry for rate limit errors
- Exponential backoff
- Batch processing with intermediate saves
- Resume functionality

### Flexible Configuration
- Choose specific subjects or exclude subjects
- Control number of rows and variations processed
- Adjust batch size, retry attempts, and sleep times
- Support for different models and platforms

## Common Usage Patterns

### Testing
```bash
# Quick test on one subject
python multipromptify_tasks/run_mmlu_batch.py --subjects anatomy --rows 2 --variations 2

# Test with dry run first
python multipromptify_tasks/run_mmlu_batch.py --subjects anatomy --dry_run
```

### Production Runs
```bash
# High reliability settings
python multipromptify_tasks/run_mmlu_batch.py --max_retries 5 --retry_sleep 120 --batch_size 5

# Use different model
python multipromptify_tasks/run_mmlu_batch.py --model llama_3_3_70b --max_tokens 1500
```

### Resume and Error Recovery
```bash
# Resume from existing results (default behavior)
python multipromptify_tasks/run_mmlu_batch.py

# Start fresh, ignore existing results
python multipromptify_tasks/run_mmlu_batch.py --no_resume
```

## File Structure

```
multipromptify_tasks/
├── data/mmlu/
│   ├── mmlu_anatomy_variations.json           # Input: Generated variations
│   ├── mmlu_chemistry_variations.json         # Input: Generated variations
│   └── ...
└── results/mmlu/
    ├── gpt_4o_mini/
    │   ├── mmlu_anatomy_variations.json       # Output: Model responses
    │   ├── mmlu_anatomy_variations.csv
    │   └── mmlu_batch_summary.json
    └── llama_3_3_70b/
        ├── mmlu_anatomy_variations.json       # Output: Model responses
        ├── mmlu_anatomy_variations.csv
        └── mmlu_batch_summary.json
```

## Configuration Options

### Model Settings
- `--platform`: TogetherAI (default) or OpenAI
- `--model`: Model to use (default, llama_3_3_70b, gpt_4o_mini, etc.)
- `--max_tokens`: Maximum response length

### Processing Control
- `--rows N`: Process only first N rows per subject
- `--variations N`: Process only first N variations per row
- `--subjects A B C`: Process only specified subjects
- `--exclude A B C`: Exclude specified subjects

### Reliability Settings
- `--max_retries N`: Maximum retry attempts for rate limits (default: 3)
- `--retry_sleep N`: Base sleep time for retries in seconds (default: 60)
- `--batch_size N`: Save results every N variations (default: 10)

### Resume Options
- `--no_resume`: Start fresh, ignore existing results

## Getting Help

```bash
# Show all available options
python multipromptify_tasks/run_mmlu_batch.py --help

# List available subjects
python multipromptify_tasks/run_mmlu_batch.py --list_subjects
``` 