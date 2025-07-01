# Execution Scripts

This directory contains scripts for running language models on generated prompt variations.

## Scripts

### Batch Runners
- `run_sentiment_batch.py` - Run sentiment analysis tasks
- `run_mmlu_batch.py` - Run MMLU multiple choice tasks  
- `run_translation_batch.py` - Run translation tasks
- `run_language_model.py` - Generic runner for any task type

### Utilities
- `batch_runner_base.py` - Base class for all batch runners
- `shared_metrics.py` - Metrics calculation functions
- `add_metrics_to_csv.py` - Add metrics to existing CSV files

## Usage

### Basic Usage
```bash
# Run sentiment analysis with default settings
python run_sentiment_batch.py

# Run MMLU with specific model
python run_mmlu_batch.py --model llama_3_3_70b --platform TogetherAI

# Run translation with limited data
python run_translation_batch.py --rows 5 --variations 3
```

### Gold Field Configuration

**Great News**: Each batch runner now has smart defaults for the `--gold_field` parameter! You usually don't need to specify it manually.

**Default Values by Task:**
- **Sentiment Analysis**: `--gold_field label` (default)
- **MMLU**: `--gold_field answer` (default)
- **Translation**: Auto-detect language codes (default)
- **QA**: Auto-detect as `answer` in generic runner
- **Summarization**: Auto-detect as `highlights` in generic runner

```bash
# These all use the correct defaults automatically:
python run_sentiment_batch.py
python run_mmlu_batch.py  
python run_translation_batch.py

# Only specify --gold_field if your data uses different field names:
python run_sentiment_batch.py --gold_field sentiment_score
python run_mmlu_batch.py --gold_field correct_answer
python run_translation_batch.py --gold_field target_text
```

### Common Gold Field Examples

| Task Type | Typical gold_updates Structure | Default --gold_field | Override Example |
|-----------|-------------------------------|---------------------|------------------|
| Sentiment | `{"label": "0.7"}` | `label` ✅ | `--gold_field sentiment_score` |
| MMLU | `{"answer": "2"}` | `answer` ✅ | `--gold_field correct_answer` |
| Translation (EN→CS) | `{"cs": "Translated text"}` | auto-detect ✅ | `--gold_field cs` |
| Translation (CS→EN) | `{"en": "Translated text"}` | auto-detect ✅ | `--gold_field en` |
| QA | `{"answer": "The answer"}` | `answer` ✅ | `--gold_field response` |
| Summarization | `{"highlights": "Summary text"}` | `highlights` ✅ | `--gold_field summary` |

✅ = Works automatically without specifying --gold_field

**Auto-Detection in Generic Runner:**
The `run_language_model.py` script automatically detects the appropriate `gold_field` based on the input filename:
- Files with "mmlu" → `--gold_field answer`
- Files with "sentiment" → `--gold_field label`  
- Files with "summarization" or "summary" → `--gold_field highlights`
- Files with "qa" or "question" → `--gold_field answer`
- Files with "translation" → Auto-detect language codes

### Resume Functionality

All scripts support resume functionality by default:

```bash
# Resume from existing results (default)
python run_sentiment_batch.py

# Start fresh (ignore existing results)
python run_sentiment_batch.py --no_resume
```

### Parallel Processing

```bash
# Use 4 parallel workers
python run_sentiment_batch.py --parallel_workers 4

# Sequential processing
python run_sentiment_batch.py --parallel_workers 1
```

## Configuration

All scripts share common parameters:

- `--model`: Model to use (gpt_4o_mini, llama_3_3_70b, etc.)
- `--platform`: Platform (OpenAI, TogetherAI)
- `--max_tokens`: Maximum response tokens
- `--temperature`: Response temperature
- `--batch_size`: Results per save batch
- `--max_retries`: Retry attempts for failed requests
- `--gold_field`: Field name for correct answer in gold_updates
- `--rows`: Limit number of rows to process
- `--variations`: Limit variations per row
- `--no_resume`: Start fresh instead of resuming

## Output

Results are saved to `project_data/results/{task}/{model}/` in both JSON and CSV formats.

## Metrics

Each task type calculates appropriate metrics:

- **Sentiment**: MSE, MAE, accuracy (±0.2 tolerance)
- **MMLU**: Exact match accuracy
- **Translation**: BLEU, ROUGE, SacreBLEU
- **QA**: BLEU, ROUGE, SacreBLEU
- **Summarization**: BLEU, ROUGE, SacreBLEU 