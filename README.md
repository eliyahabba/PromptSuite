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

### Streamlit Interface (Recommended)

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

### Command Line Interface

```bash
multipromptify --template '{"instruction": "{instruction}: {text}", "text": ["paraphrase_with_llm"], "gold": "label"}' \
               --data data.csv --max-variations-per-row 50
```

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
  'question': ['format_structure'],
}
mp.set_template(template)

# Generate variations
mp.configure(max_rows=2, variations_per_field=3)
variations = mp.generate(verbose=True)

# Export results
mp.export("output.json", format="json")
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
        "format": "fixed",
        "split": "all"
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

### Full Example: Using All Main Template Keys

Below is a recommended way to define a template using all the main template keys from `multipromptify.core.template_keys`:

```python
template = {
  "instruction": "You are a helpful assistant. Please answer the following questions.",
  "instruction variations": ["format_structure"],  # Variation types for the instruction
  "prompt format": "Q: {question}\nOptions: {options}\nA: {answer}",
  "prompt format variations": ["paraphrase_with_llm"],  # Variation types for the prompt format
  "question": ["typos_and_noise"],
  "options": ["shuffle"],
  "gold": {
    'field': 'answer',
    'type': 'index',
    'options_field': 'options'
  },
  "few_shot": {
    'count': 2,
    'format': 'rotating',
    'split': 'all'
  }
}
```

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
- `format_structure` - Semantic-preserving format changes (e.g., separators, casing, field order)
- `typos_and_noise` - Injects typos, random case, extra whitespace, and punctuation noise for robustness
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
  "prompt format variations": ["format_structure"],
  "question": ["shuffle", "typos_and_noise""typos_and_noise"],
  "options": ["enumerate"],
  "gold": {
    'field': 'answer',
    'type': 'index',
    'options_field': 'options'
  },
  "few_shot": {
    'count': 2,
    'format': 'rotating',
    'split': 'all'
  }
}
```


## Features

### Template System
- **Python f-string compatibility**: Use familiar `{variable}` syntax
- **Variation annotations**: Specify variation types with `:type` syntax
- **Flexible column mapping**: Reference any column from your data
- **Literal support**: Use static strings and numbers

### Input Handling
- **CSV/DataFrame support**: Direct pandas DataFrame or CSV file input
- **HuggingFace datasets**: Full compatibility with datasets library
- **Dictionary inputs**: Support for various input types

### Web UI Features
- **Sample Datasets**: Built-in datasets for quick testing
- **Template Suggestions**: Smart suggestions based on your data
- **Real-time Validation**: Instant feedback on template syntax
- **Live Preview**: Test templates before full generation
- **Advanced Analytics**: Distribution charts, field analysis
- **Search & Filter**: Find specific variations quickly
- **Multiple Export Formats**: JSON, CSV, TXT, and custom formats

### Few-shot Examples
```python
# Different examples per sample
few_shot = [
    ["Example 1 for sample 1", "Example 2 for sample 1"],
    ["Example 1 for sample 2", "Example 2 for sample 2"]
]

# Same examples for all samples
few_shot = ("Example 1", "Example 2")
```

## Command Line Interface

### Basic Commands

```bash
# Basic usage
multipromptify --template '{"instruction": "{instruction}: {question}", "question": ["paraphrase_with_llm"], "gold": "answer"}' \
               --data data.csv

# With output file
multipromptify --template '{"instruction": "{instruction}: {question}", "question": ["paraphrase_with_llm"], "gold": "answer"}' \
               --data data.csv \
               --output variations.json

# Specify number of variations PER ROW
multipromptify --template '{"instruction": "{instruction}: {question}", "instruction": ["semantic"], "question": ["format_structure"], "gold": "answer"}' \
               --data data.csv \
               --max-variations-per-row 50
```

### Advanced Options

```bash
# With few-shot examples
multipromptify --template '{"instruction": "{instruction}: {question}", "question": ["paraphrase_with_llm"], "gold": "answer", "few_shot": {"count": 2, "format": "fixed", "split": "all"}}' \
               --data data.csv \
               --max-variations-per-row 50

# Output in different formats
multipromptify --template '{"instruction": "{instruction}: {question}", "instruction": ["semantic"], "question": ["format_structure"], "gold": "answer"}' \
               --data data.csv \
               --format csv \
               --output variations.csv
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

### Minimal Example

A minimal example for basic usage: loading data, setting a template, and generating variations.

```python
import pandas as pd
from multipromptify import MultiPromptifier

data = pd.DataFrame({
    'question': ['What is 2+2?', 'What is the capital of France?'],
    'answer': ['4', 'Paris']
})

template = {
    'instruction': 'Please answer the following questions.',
    'prompt format': 'Q: {question}\nA: {answer}',
    'question': ['format_structure']
}

mp = MultiPromptifier()
mp.load_dataframe(data)
mp.set_template(template)
mp.configure(max_rows=2, variations_per_field=2)
variations = mp.generate(verbose=True)
print(variations)
```

### Sentiment Analysis

```python
import pandas as pd
from multipromptify import MultiPromptifier

data = pd.DataFrame({
  'text': ['I love this movie!', 'This book is terrible.'],
  'label': ['positive', 'negative']
})

template = {
  'instruction': 'Classify the sentiment: "{text}"\nSentiment: {label}',
  'instruction_variations': ['semantic'],
  'text': ['paraphrase_with_llm'],
  'gold': 'label'
}

mp = MultiPromptifier()
mp.load_dataframe(data)
mp.set_template(template)
mp.configure(
  max_rows=GenerationDefaults.MAX_ROWS,
  variations_per_field=GenerationDefaults.VARIATIONS_PER_FIELD,
  max_variations_per_row=GenerationDefaults.MAX_VARIATIONS_PER_ROW,
  random_seed=GenerationDefaults.RANDOM_SEED,
  api_platform=GenerationDefaults.API_PLATFORM,
  model_name=GenerationDefaults.MODEL_NAME
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
    'format': 'rotating',
    'split': 'all'
  }
}

mp = MultiPromptifier()
mp.load_dataframe(qa_data)
mp.set_template(template)
mp.configure(
  max_rows=GenerationDefaults.MAX_ROWS,
  variations_per_field=GenerationDefaults.VARIATIONS_PER_FIELD,
  max_variations_per_row=GenerationDefaults.MAX_VARIATIONS_PER_ROW,
  random_seed=GenerationDefaults.RANDOM_SEED,
  api_platform=GenerationDefaults.API_PLATFORM,
  model_name=GenerationDefaults.MODEL_NAME
)
variations = mp.generate(verbose=True)
```

### Multiple Choice with Dynamic System Prompt and Few-shot

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
    'instruction': 'The following are multiple choice questions (with answers) about {subject}.',
    'prompt format': 'Question: {question}\nOptions: {options}\nAnswer:',
    'question': ['format_structure'],
    'options': ['shuffle'],
    'gold': {
        'field': 'answer',
        'type': 'index',
        'options_field': 'options'
    },
    'few_shot': {
        'count': 2,
        'format': 'rotating',
        'split': 'all'
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

## Web UI Interface

MultiPromptify 2.0 includes a modern, interactive web interface built with **Streamlit**.

The UI guides you through a simple 3-step workflow:

1. **Upload Data**: Load your dataset (CSV/JSON) or use built-in samples. Preview and validate your data before continuing.
2. **Build Template**: Create or select a prompt template, with smart suggestions based on your data. See a live preview of your template.
3. **Generate & Export**: Configure generation settings, run the variation process, and export your results in various formats.

The Streamlit UI is the easiest way to explore, test, and generate prompt variations visually.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details. 