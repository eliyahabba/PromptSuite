# MultiPromptify Tasks

This directory contains task-specific scripts for running language models on various datasets.

## Scripts Overview

### 1. `run_language_model.py`
Runs language models on prompt variations with robust error handling and batch processing.

**Key Features:**
- Rate limit handling with exponential backoff
- Batch processing with intermediate saves
- Resume functionality
- Support for multiple platforms (TogetherAI, OpenAI)
- CSV and JSON output formats
- **NEW**: Model-specific output directories

**Example Usage:**
```bash
# Basic run with default settings
python run_language_model.py

# Custom model and settings
python run_language_model.py --model llama_3_3_70b --max_tokens 512 --batch_size 5

# Process only specific rows/variations
python run_language_model.py --rows 10 --variations 3

# Start fresh (don't resume)
python run_language_model.py --no_resume
```

### 2. `run_mmlu_batch.py`
Automatically runs language models on all MMLU subject variation files.

**Key Features:**
- Batch processing of all MMLU subjects
- Subject filtering (include/exclude specific subjects)
- **NEW**: Real-time progress tracking with detailed output
- **NEW**: Model-specific output directories
- Comprehensive batch summaries
- Dry run mode for testing

**Example Usage:**
```bash
# Run all subjects with default settings
python run_mmlu_batch.py

# Run specific subjects only
python run_mmlu_batch.py --subjects anatomy chemistry

# Exclude specific subjects
python run_mmlu_batch.py --exclude anatomy chemistry

# Custom model and settings
python run_mmlu_batch.py --model gpt_4o_mini --platform OpenAI --max_tokens 512

# Process limited data for testing
python run_mmlu_batch.py --rows 5 --variations 2 --batch_size 1

# Dry run to see what would be processed
python run_mmlu_batch.py --dry_run
```

## Directory Structure

### Output Organization
Results are now organized by model in separate directories:

```
multipromptify_tasks/
├── results/
│   └── mmlu/
│       ├── llama_3_3_70b/
│       │   ├── mmlu_anatomy_variations.json
│       │   ├── mmlu_anatomy_variations.csv
│       │   └── ...
│       ├── gpt_4o_mini/
│       │   ├── mmlu_anatomy_variations.json
│       │   ├── mmlu_anatomy_variations.csv
│       │   └── ...
│       └── ...
└── data/
    └── mmlu/
        ├── llama_3_3_70b/
        │   └── mmlu_batch_summary.json
        ├── gpt_4o_mini/
        │   └── mmlu_batch_summary.json
        └── ...
```

### Benefits of New Structure:
- **Clean Organization**: Each model has its own directory
- **Easy Comparison**: Compare results across models easily
- **No Filename Conflicts**: No need for model prefixes in filenames
- **Batch Summaries**: Each model gets its own batch summary

## Progress Tracking

### Real-time Progress
Both scripts now provide detailed real-time progress information:

- **Processing Status**: Shows current variation being processed
- **Batch Saves**: Indicates when intermediate results are saved
- **Rate Limit Handling**: Shows retry attempts and sleep times
- **Error Details**: Displays specific error messages
- **Resume Information**: Shows when resuming from existing results

### Example Progress Output:
```
🚀 Running: --input_folder /path/to/data --input_file mmlu_anatomy_variations.json
   ✅ Loaded 25 variations from /path/to/data/mmlu_anatomy_variations.json
   🔍 Filtered to 2 variations from 5 rows
   🔄 Processing 2 remaining variations (out of 2 total)
   Processing 1/2 (variation 1)
   💾 Saving batch 1 (1 total results, 50.0% complete)...
   Processing 2/2 (variation 13)
   💾 Saving batch 2 (2 total results, 100.0% complete)...
   ✅ Processing completed!
```

## Configuration

### Model Configuration
Models are configured in `constants.py`:

```python
MODELS = {
    "TogetherAI": {
        "default": "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        "llama_3_3_70b": "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
    },
    "OpenAI": {
        "default": "gpt-4o-mini",
        "gpt_4o_mini": "gpt-4o-mini",
        "gpt_4o": "gpt-4o",
    }
}
```

### Platform Configuration
Supported platforms:
- **TogetherAI**: For Llama models and other open models
- **OpenAI**: For GPT models

## Error Handling

### Rate Limit Handling
- Automatic retry with exponential backoff
- Configurable retry count and sleep times
- Only retries rate limit errors (not other errors)

### Resume Functionality
- Automatically resumes from where it left off
- Saves intermediate results every batch
- Can be disabled with `--no_resume` flag

### Batch Processing
- Processes variations in configurable batches
- Saves results after each batch
- Allows for safe interruption and resumption

## Examples

### Complete MMLU Processing
```bash
# Process all MMLU subjects with Llama model
python run_mmlu_batch.py --model llama_3_3_70b --batch_size 10

# Process with GPT model and custom settings
python run_mmlu_batch.py --model gpt_4o_mini --platform OpenAI --max_tokens 512 --batch_size 5
```

### Individual Subject Processing
```bash
# Process single subject
python run_language_model.py --input_file mmlu_anatomy_variations.json

# Process with limited data for testing
python run_language_model.py --input_file mmlu_anatomy_variations.json --rows 5 --variations 2
```

### Testing and Debugging
```bash
# Dry run to see what would be processed
python run_mmlu_batch.py --dry_run --subjects anatomy chemistry

# Start fresh without resuming
python run_mmlu_batch.py --no_resume --subjects anatomy
```

## Troubleshooting

### Common Issues
1. **Rate Limits**: Increase `--retry_sleep` and `--max_retries`
2. **Memory Issues**: Reduce `--batch_size`
3. **Long Processing**: Use `--rows` and `--variations` to limit data
4. **Resume Issues**: Use `--no_resume` to start fresh

### Output Files
- **JSON**: Full results with all metadata
- **CSV**: Simplified results for analysis
- **Batch Summary**: Overall processing statistics

## Performance Tips

1. **Batch Size**: Use larger batch sizes for faster processing (if memory allows)
2. **Rate Limits**: Monitor API rate limits and adjust retry settings
3. **Testing**: Use `--rows` and `--variations` for quick testing
4. **Resume**: Always use resume mode for long-running processes 