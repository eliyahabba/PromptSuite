# Execution Module

This module contains simplified batch runners for processing language model tasks.

## Structure

### Base Class
- `batch_runner_base.py` - Common functionality for all batch runners
  - Handles file loading, processing, and result saving
  - Provides common CLI arguments
  - Manages progress reporting and error handling

### Batch Runners
- `run_mmlu_batch.py` - MMLU subject batch processing
- `run_translation_batch.py` - Translation language pair batch processing  
- `run_language_model.py` - General language model runner

### Support Files
- `shared_metrics.py` - Metrics calculation functions
- `add_metrics_to_csv.py` - Utility for adding metrics to CSV files

## Key Simplifications

1. **Removed `accuracy_only` option** - Simplified command line interface
2. **Created base class** - Eliminated code duplication between batch runners
3. **Consolidated imports** - Cleaner dependency management
4. **Unified argument handling** - Common CLI arguments across runners

## Usage Examples

```bash
# Run MMLU batch processing
python run_mmlu_batch.py --batch_size 5 --max_retries 5

# Run translation batch processing  
python run_translation_batch.py --model llama_3_3_70b --max_tokens 512

# List available subjects/pairs
python run_mmlu_batch.py --list_subjects
python run_translation_batch.py --list_pairs

# Process specific items only
python run_mmlu_batch.py --subjects anatomy chemistry
python run_translation_batch.py --language_pairs cs-en de-en
```

## Benefits

- **Reduced complexity**: Removed ~300 lines of duplicated code
- **Better maintainability**: Changes to common functionality only need to be made once
- **Cleaner interface**: Simplified command line options
- **Consistent behavior**: All batch runners work the same way 