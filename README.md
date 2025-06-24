# MultiPromptify

A tool that creates multi-prompt datasets from single-prompt datasets using templates with variation specifications.

## Overview

MultiPromptify transforms your single-prompt datasets into rich multi-prompt datasets by applying various types of variations specified in your templates. It supports HuggingFace-compatible datasets and provides both a command-line interface and a modern web UI.

## 📚 Documentation

- 📖 **[Complete API Guide](docs/api-guide.md)** - Python API reference and examples
- 🏗️ **[Developer Documentation](docs/dev/)** - For contributors and developers

## Installation

### From PyPI (Recommended)

```bash
pip install multipromptify
```

### From GitHub (Latest)

```bash
pip install git+https://github.com/ehabba/MultiPromptifyPipeline.git
```

### From Source

```bash
git clone https://github.com/ehabba/MultiPromptifyPipeline.git
cd MultiPromptifyPipeline
pip install -e .
```

## Quick Start
### Command Line Interface

```bash
multipromptify --template '{"instruction": "{instruction}: {text}", "text": ["paraphrase_with_llm"], "gold": "label"}' \
               --data data.csv --max-variations-per-row 50
```
### Streamlit Interface

Launch the modern Streamlit interface for an intuitive experience:

```bash
# If installed via pip
multipromptify-ui

# From project root (development)
python src/multipromptify/ui/main.py

# Alternative: using the runner script
python scripts/run_ui.py
```

The web UI provides:
- 📁 **Step 1**: Upload data or use sample datasets
- 🔧 **Step 2**: Build templates with smart suggestions
- ⚡ **Step 3**: Generate variations with real-time progress and export results


### Python API

```python
from multipromptify import MultiPromptifier
import pandas as pd

# Initialize
mp = MultiPromptifier()

# Load data
data = [{"question": "What is 2+2?", "answer": "4"}]
mp.load_dataframe(pd.DataFrame(data))

# Configure template
template = {
  'instruction': 'Please answer the following questions.',
  'prompt format': 'Q: {question}\nA: {answer}',
  'question': ['typos and noise'],
}
mp.set_template(template)

# Generate variations
mp.configure(max_rows=2, variations_per_field=3)
variations = mp.generate(verbose=True)

# Export results
mp.export("output.json", format="json")
```

## 📚 Core Concepts

### Templates
Templates control how prompts are structured and varied:

| Key | Description                       | Example                                                                                                               |
|-----|-----------------------------------|-----------------------------------------------------------------------------------------------------------------------|
| `instruction` | System prompt (optional) {placeholders}| `'You are a helpful assistant. Answer the following questions about {subject}.''` |
| `prompt format` | Main template with {placeholders} | `'Q: {question}\nA: {answer}'`                                                                                        |
| `gold` | Correct answer field | `'answer'` or `{'field': 'answer', 'type': 'index'}`                                                                  |
| `few_shot` | Few-shot configuration | `{'count': 2, 'format': 'shared_random_n', 'split': 'train'}`                                                                  |

### Variation Types

| Type                  | Description | Requires API Key |
|-----------------------|-------------|------------------|
| `paraphrase_with_llm` | AI-powered rephrasing | ✅ |
| `context`             | Adds background context | ✅ |
| `format_structure`    | Changes separators, casing, field connectors | ❌ |
| `typos and noise`     | Injects typos, capitalization changes, spacing, character swaps, and punctuation noise | ❌ |
| `shuffle`             | Reorders list items | ❌ |
| `enumerate`           | Adds numbering (1. 2. 3.) | ❌ |


This template demonstrates how to use all the main keys for maximum flexibility and clarity. You can import these keys from `multipromptify.core.template_keys` to avoid typos and ensure consistency.

## Template Format

Templates use Python f-string syntax with custom variation annotations:

```python
"{instruction:semantic}: {few_shot}\n Question: {question:paraphrase_with_llm}\n Options: {options:non-semantic}"
```

### System Prompt
- `instruction`: (optional) A general instruction that appears at the top of every prompt, before any few-shot or main question. You can use placeholders (e.g., `{subject}`) that will be filled from the data for each row.
- `prompt format`: The per-example template, usually containing the main question and placeholders for fields.

## Supported Variation Types

- `paraphrase_with_llm` - Paraphrasing variations (LLM-based)
- `format_structure` - Semantic-preserving format changes (e.g., separators, casing, field connectors)
- `typos and noise` - Injects typos, capitalization changes, spacing, character swaps, and punctuation noise
- `context` - Context-based variations
- `shuffle` - Shuffle options/elements (for multiple choice)
- `enumerate` - Enumerate list fields (e.g., 1. 2. 3. 4., A. B. C. D., roman numerals, etc.)
You can combine these augmenters in your template for richer prompt variations.

## Template Format

Templates use a dictionary format with specific keys for different components:

```python
template = {
  "instruction": "You are a helpful assistant. Please answer the following questions.",
  "instruction variations": ["paraphrase_with_llm"],
  "prompt format": "Q: {question}\nOptions: {options}\nA: {answer}",
  "prompt format variations": ["format structure"],
  "question": ["shuffle", "typos and noise"],
  "options": ["enumerate"],
  "gold": {
    'field': 'answer',
    'type': 'index',
    'options_field': 'options'
  },
  "few_shot": {
    'count': 2,
    'format': 'shared_random_n',
    'split': 'train'
  }
}
```

## API Reference

### MultiPromptifier Class

```python
class MultiPromptifier:
    def __init__(self):
        """Initialize MultiPromptifier."""
        
    def load_dataframe(self, df: pd.DataFrame) -> None:
        """Load data from pandas DataFrame."""
        
    def load_csv(self, filepath: str, **kwargs) -> None:
        """Load data from CSV file."""
        
    def load_dataset(self, dataset_name: str, split: str = "train", **kwargs) -> None:
        """Load data from HuggingFace datasets."""
        
    def set_template(self, template_dict: Dict[str, Any]) -> None:
        """Set template configuration."""
        
    def configure(self, **kwargs) -> None:
        """Configure generation parameters."""
        
    def generate(self, verbose: bool = False) -> List[Dict[str, Any]]:
        """Generate prompt variations."""
        
    def export(self, filepath: str, format: str = "json") -> None:
        """Export variations to file."""
```

## Examples

### Sentiment Analysis

```python
import pandas as pd
from multipromptify import MultiPromptifier

data = pd.DataFrame({
  'text': ['I love this movie!', 'This book is terrible.'],
  'label': ['positive', 'negative']
})

template = {
  'instruction': 'Classify the sentiment',
  'instruction_variations': ['paraphrase_with_llm'],
  'prompt format': f"Text: {text}\nSentiment: {label}",
  'text': ['typos and noise'],
}

mp = MultiPromptifier()
mp.load_dataframe(data)
mp.set_template(template)
mp.configure(
  variations_per_field=3,
  max_variations_per_row=2,
  random_seed=42,
  api_platform="TogetherAI",
  model_name="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
)
variations = mp.generate(verbose=True)
```

### Question Answering with Few-shot

```python
template = {
  'instruction': 'Answer the question:\nQuestion: {question}\nAnswer: {answer}',
  'instruction_variations': ['paraphrase_with_llm'],
  'question': ['semantic'],
  'gold': 'answer',
  'few_shot': {
    'count': 2,
    'format': 'shared_random_n',
    'split': 'train'
  }
}

mp = MultiPromptifier()
mp.load_dataframe(qa_data)
mp.set_template(template)
mp.configure(
  variations_per_field=2,
  api_platform="TogetherAI",
  model_name="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
)
variations = mp.generate(verbose=True)
```

### Multiple Choice with Few-shot

```python
import pandas as pd
from multipromptify import MultiPromptifier

data = pd.DataFrame({
    'question': [
        'What is the largest planet in our solar system?',
        'Which chemical element has the symbol O?',
        'What is the fastest land animal?',
        'What is the smallest prime number?',
        'Which continent is known as the "Dark Continent"?'
    ],
    'options': [
        'Earth, Jupiter, Mars, Venus',
        'Oxygen, Gold, Silver, Iron',
        'Lion, Cheetah, Horse, Leopard',
        '1, 2, 3, 0',
        'Asia, Africa, Europe, Australia'
    ],
    'answer': [1, 0, 1, 1, 1],
    'subject': ['Astronomy', 'Chemistry', 'Biology', 'Mathematics', 'Geography']
})

template = {
    'prompt format': 'Question: {question}\nOptions: {options}\nAnswer:',
    'prompt format variations': ['format structure'],
    'options': ['shuffle', 'enumerate'],
    'gold': {
        'field': 'answer',
        'type': 'index',
        'options_field': 'options'
    },
    'few_shot': {
        'count': 2,
        'format': 'shared_random_n',
        'split': 'train'
    }
}

mp = MultiPromptifier()
mp.load_dataframe(data)
mp.set_template(template)
mp.configure(max_rows=5, variations_per_field=1)
variations = mp.generate(verbose=True)
for v in variations:
    print(v['prompt'])
```



### Example Output Format

A typical output from `mp.generate()` or the exported JSON file looks like this (for a multiple choice template):

```json
[
  {
    "prompt": "Answer the following multiple choice question:\nQuestion: What is 2+2?\nOptions: 3, 4, 5, 6\nAnswer:",
    "original_row_index": 1,
    "variation_count": 1,
    "template_config": {
      "instruction": "Answer the following multiple choice question:\nQuestion: {question}\nOptions: {options}\nAnswer: {answer}",
      "options": ["shuffle"],
      "gold": {
        "field": "answer",
        "type": "index",
        "options_field": "options"
      },
      "few_shot": {
        "count": 1,
        "format": "shared_random_n",
        "split": "train"
      }
    },
    "field_values": {
      "options": "3, 4, 5, 6"
    },
    "gold_updates": {
      "answer": "1"
    },
    "conversation": [
      {
        "role": "user",
        "content": "Answer the following multiple choice question:\nQuestion: What is 2+2?\nOptions: 3, 4, 5, 6\nAnswer:"
      },
      {
        "role": "assistant",
        "content": "1"
      },
      {
        "role": "user",
        "content": "Answer the following multiple choice question:\nQuestion: What is the capital of France?\nOptions: London, Berlin, Paris, Madrid\nAnswer:"
      }
    ]
  }
]

```
## 📖 Detailed Guide

### Data Loading
```python
# CSV
mp.load_csv('data.csv')

# JSON
mp.load_json('data.json')

# HuggingFace
mp.load_dataset('squad', split='train[:100]')

# DataFrame
mp.load_dataframe(df)
```

### Generation Options
```python
mp.configure(
    max_rows=10,                    # How many data rows to use
    variations_per_field=3,         # Variations per field (default: 3)
    max_variations_per_row=50,      # Cap on total variations per row
    random_seed=42,                 # For reproducibility
    api_platform="TogetherAI",      # or "OpenAI"
    model_name="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
)
```

### Export Formats
```python
# JSON - Full data with metadata
mp.export("output.json", format="json")

# CSV - Flattened for spreadsheets
mp.export("output.csv", format="csv")

# TXT - Plain prompts only
mp.export("output.txt", format="txt")
```

## Web UI Interface

MultiPromptify 2.0 includes a modern, interactive web interface built with **Streamlit**.

The UI guides you through a simple 3-step workflow:

1. **Upload Data**: Load your dataset (CSV/JSON) or use built-in samples. Preview and validate your data before continuing.
2. **Build Template**: Create or select a prompt template, with smart suggestions based on your data. See a live preview of your template.
3. **Generate & Export**: Configure generation settings, run the variation process, and export your results in various formats.

The Streamlit UI is the easiest way to explore, test, and generate prompt variations visually.

## 🔧 Advanced Features

### Performance Optimization

MultiPromptify automatically optimizes performance by pre-generating variations for shared fields:

- **Instruction variations** (`instruction variations`) are generated once and reused across all data rows
- **Prompt format variations** (`prompt format variations`) are generated once and reused across all data rows

This optimization is especially important for LLM-based augmenters like `paraphrase_with_llm` that would otherwise run the same API calls repeatedly for identical text.

### Gold Field Configuration

**Simple format** (for text answers):
```python
'gold': 'answer'  # Just the column name
```

**Advanced format** (for index-based answers):
```python
'gold': {
    'field': 'answer',
    'type': 'index',        # Answer is an index
    'options_field': 'options'  # Column with the options
}
```

### Few-Shot Configuration

Few-shot examples can be configured with different sampling strategies:

| Format | Description | Use Case |
|--------|-------------|----------|
| `shared_first_n` | Always uses the first N examples from available data (deterministic, shared for all rows) | When you want consistent, predictable examples |
| `shared_random_n` | Always uses the same N random examples (with fixed seed, shared for all rows) | When you want random but consistent examples across all rows |
| `random_per_row` | Randomly samples different examples for each row (using row index as seed) | When you want variety and different examples per question |

**Example:**
```python
"few_shot": {
    "count": 2,                    # Number of examples to use
    "format": "shared_random_n",   # Sampling strategy
    "split": "train"               # Use only training data for examples
}
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details. 