<div align="center">
  <img src="logo.png" alt="PromptSuite Logo" width="120">
  <h1>PromptSuite: A Task-Agnostic Framework for Multi-Prompt Generation</h1>
</div>
A tool that creates multi-prompt datasets from single-prompt datasets using templates with variation specifications.

## Overview

PromptSuite transforms your single-prompt datasets into rich multi-prompt datasets by applying various types of variations specified in your templates. It supports HuggingFace-compatible datasets and provides both a command-line interface and a modern web UI.

## 📚 Documentation

- 📖 **[Complete API Guide](docs/api-guide.md)** - Python API reference and examples
- 🏗️ **[Developer Documentation](docs/dev/)** - For contributors and developers

## Installation

### From PyPI (Recommended)

```bash
pip install promptsuite
```

### From GitHub (Latest)

```bash
pip install git+https://github.com/eliyahabba/PromptSuite.git
```

### From Source

```bash
git clone https://github.com/eliyahabba/PromptSuite.git
cd PromptSuite
pip install -e .
```

## Quick Start
### Command Line Interface

```bash
promptsuite --template '{"instruction": "{instruction}: {text}", "text": ["paraphrase_with_llm"], "gold": "label"}' \
               --data data.csv --max-variations-per-row 50
```
### Streamlit Interface

Launch the modern Streamlit interface for an intuitive experience:

```bash
# If installed via pip
promptsuite-ui

# From project root (development)
python src/promptsuite/ui/main.py
```

The web UI provides:
- 📁 **Step 1**: Upload data or use sample datasets
- 🔧 **Step 2**: Build templates with smart suggestions
- ⚡ **Step 3**: Generate variations with real-time progress and export results


### Python API

```python
from promptsuite import PromptSuite
import pandas as pd

# Initialize
ps = PromptSuite()

# Load data
data = [{"question": "What is 2+2?", "answer": "4"}]
ps.load_dataframe(pd.DataFrame(data))

# Configure template
template = {
  'instruction': 'Please answer the following questions.',
  'prompt format': 'Q: {question}\nA: {answer}',
  'question': ['typos and noise'],
}
ps.set_template(template)

# Generate variations
ps.configure(max_rows=2, variations_per_field=3)
variations = ps.generate(verbose=True)

# Export results
ps.export("output.json", format="json")
```

For more detailed examples of API usage, refer to the `examples/` directory.

## 📖 Code Examples

### Sentiment Analysis

```python
import pandas as pd
from promptsuite import PromptSuite

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

ps = PromptSuite()
ps.load_dataframe(data)
ps.set_template(template)
ps.configure(
  variations_per_field=3,
  max_variations_per_row=2,
  random_seed=42,
  api_platform="TogetherAI",  # or "OpenAI", "Anthropic", "Google", "Cohere"
  model_name="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
)
variations = ps.generate(verbose=True)
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
    'format': 'same_examples__synchronized_order_variations',
    'split': 'train'
  }
}

ps = PromptSuite()
ps.load_dataframe(qa_data)
ps.set_template(template)
ps.configure(
  variations_per_field=2,
  api_platform="TogetherAI",  # or "OpenAI", "Anthropic", "Google", "Cohere"
  model_name="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
)
variations = ps.generate(verbose=True)
```

### Multiple Choice with Few-shot

```python
import pandas as pd
from promptsuite import PromptSuite

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
        'format': 'same_examples__synchronized_order_variations',
        'split': 'train'
    }
}

ps = PromptSuite()
ps.load_dataframe(data)
ps.set_template(template)
ps.configure(max_rows=5, variations_per_field=1)
variations = ps.generate(verbose=True)
for v in variations:
    print(v['prompt'])
```



### Example Output Format

A typical output from `ps.generate()` or the exported JSON file looks like this (for a multiple choice template):

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
        "format": "same_examples__synchronized_order_variations",
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
ps.load_csv('data.csv')

# JSON
ps.load_json('data.json')

# HuggingFace
ps.load_dataset('squad', split='train[:100]')

# DataFrame
ps.load_dataframe(df)
```

### Generation Options
```python
ps.configure(
    max_rows=10,                    # How many data rows to use
    variations_per_field=3,         # Variations per field (default: 3)
    max_variations_per_row=50,      # Cap on total variations per row
    random_seed=42,                 # For reproducibility
    api_platform="TogetherAI",      # or "OpenAI", "Anthropic", "Google", "Cohere"
    model_name="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
)
```

### Export Formats
```python
# JSON - Full data with metadata
ps.export("output.json", format="json")

# CSV - Flattened for spreadsheets
ps.export("output.csv", format="csv")

# TXT - Plain prompts only
ps.export("output.txt", format="txt")
```

## Web UI Interface

PromptSuite 2.0 includes a modern, interactive web interface built with **Streamlit**.

The UI guides you through a simple 3-step workflow:

1. **Upload Data**: Load your dataset (CSV/JSON) or use built-in samples. Preview and validate your data before continuing.
2. **Build Template**: Create or select a prompt template, with smart suggestions based on your data. See a live preview of your template.
3. **Generate & Export**: Configure generation settings, run the variation process, and export your results in various formats.

The Streamlit UI is the easiest way to explore, test, and generate prompt variations visually.

## 🤖 Supported AI Platforms

PromptSuite supports multiple AI platforms with automatic dependency detection:

### Core Platforms (Always Available)
- **TogetherAI**: Open-source models (Llama, Mistral, etc.)
- **OpenAI**: GPT models (GPT-4, GPT-3.5, etc.)

### Extended Platforms (Optional Dependencies)
- **Anthropic**: Claude models (claude-3-haiku, claude-3-sonnet, claude-3-opus)
- **Google**: Gemini models (gemini-1.5-flash, gemini-1.5-pro)
- **Cohere**: Command models (command-r-plus, command-r)

### Installation

Install optional platform dependencies:
```bash
pip install -r requirements-optional.txt
```

Or install specific platforms:
```bash
pip install anthropic          # For Anthropic/Claude
pip install google-generativeai # For Google/Gemini
pip install cohere             # For Cohere
```

### API Keys

Set environment variables for the platforms you want to use:
```bash
export TOGETHER_API_KEY="your_together_key"
export OPENAI_API_KEY="your_openai_key"
export ANTHROPIC_API_KEY="your_anthropic_key"
export GOOGLE_API_KEY="your_google_key"
export COHERE_API_KEY="your_cohere_key"
```

### Platform Selection

```python
# Automatic platform detection
from promptsuite.shared.model_client import get_supported_platforms, is_platform_available

available_platforms = [p for p in get_supported_platforms() if is_platform_available(p)]
print(f"Available platforms: {available_platforms}")

# Use different platforms
ps.configure(api_platform="OpenAI", model_name="gpt-4o-mini")
ps.configure(api_platform="Anthropic", model_name="claude-3-haiku-20240307")
ps.configure(api_platform="Google", model_name="gemini-1.5-flash")
```

### Adding New Platforms

See [Platform Integration Guide](docs/platform-integration-guide.md) for instructions on adding support for additional AI platforms.

## 🔧 Advanced Features

### Performance Optimization

PromptSuite automatically optimizes performance by pre-generating variations for shared fields:

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

Few-shot examples can be configured with different sampling strategies. The format names clearly indicate what varies across data rows and variations:

**Format naming convention: `<examples_strategy>__<order_strategy>`**

- **Examples strategy**: Whether examples are the same or different across data rows
- **Order strategy**: Whether the order of examples varies across variations

| Format | Description | Use Case |
|--------|-------------|----------|
| `same_examples__no_variations` | Same examples for all rows, no variations (single variation per row) | When you want consistent, predictable examples |
| `same_examples__synchronized_order_variations` | Same examples for all rows, synchronized order variations across all rows | When you want consistent examples but test different orderings |
| `different_examples__same_shuffling_order_across_rows` | Different examples per row, same shuffling order across rows | When you want unique examples per question but consistent ordering patterns |
| `different_examples__different_order_per_variation` | Different examples and different order per variation | When you want maximum variety and different examples per question |

**Examples:**

```python
# same_examples__no_variations
# Row 1: [Example A, Example B]
# Row 2: [Example A, Example B]  # Same examples, no variations

# same_examples__synchronized_order_variations  
# Row 1, Variation 1: [Example A, Example B]
# Row 1, Variation 2: [Example B, Example A]
# Row 2, Variation 1: [Example A, Example B]  # Same order as Row 1, Variation 1
# Row 2, Variation 2: [Example B, Example A]  # Same order as Row 1, Variation 2

# different_examples__same_shuffling_order_across_rows
# Row 1, Variation 1: [Example A, Example B]
# Row 1, Variation 2: [Example A, Example B]  # Same examples for this row
# Row 2, Variation 1: [Example C, Example D]  # Different examples
# Row 2, Variation 2: [Example C, Example D]  # Same examples for this row

# different_examples__different_order_per_variation
# Row 1, Variation 1: [Example A, Example B]
# Row 1, Variation 2: [Example C, Example D]  # Different examples
# Row 2, Variation 1: [Example E, Example F]  # Different examples
# Row 2, Variation 2: [Example G, Example H]  # Different examples
```

**Example:**
```python
"few_shot": {
    "count": 2,                    # Number of examples to use
    "format": "same_examples__synchronized_order_variations",   # Sampling strategy
    "split": "train"              # Use only training data for examples
}
```

This feature is useful when you want each test question to have unique few-shot examples for context, but don't need multiple variations of the few-shot examples themselves.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details. 
