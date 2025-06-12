# MultiPromptify

A tool that creates multi-prompt datasets from single-prompt datasets using templates with variation specifications.

## Overview

MultiPromptify transforms your single-prompt datasets into rich multi-prompt datasets by applying various types of variations specified in your templates. It supports HuggingFace-compatible datasets and provides both a command-line interface and a modern web UI.

## Installation

```bash
pip install multipromptify
```

Or install from source:
```bash
git clone <repository>
cd multipromptify
pip install -e .
```

### With Web UI Support

```bash
# Install with web UI components
pip install -e ".[ui]"
```

## Quick Start

### Web UI (Recommended)

Launch the modern web interface for an intuitive experience:

```bash
# From project root
python src/ui/run_streamlit.py

# Or use the demo script
python demo_ui.py
```

The web UI provides:
- 📁 **Step 1**: Upload data or use sample datasets
- 🔧 **Step 2**: Build templates with smart suggestions
- ⚡ **Step 3**: Generate variations with real-time progress
- 🎉 **Step 4**: Analyze results and export in multiple formats

### Command Line Interface

```bash
multipromptify --template "{instruction:semantic}: {col1:paraphrase}" \
               --data data.csv \
               --instruction "Classify the sentiment"
```

### Python API

```python
from multipromptify import MultiPromptify
import pandas as pd

# Your data
data = pd.DataFrame({
    'question': ['What is 2+2?', 'What color is the sky?'],
    'options': ['A)3 B)4 C)5', 'A)Red B)Blue C)Green']
})

# Template with variation specifications
template = "{instruction:semantic}: {few_shot}\n Question: {question:paraphrase}\n Options: {options}"

# Initialize and generate variations
mp = MultiPromptify()
variations = mp.generate_variations(
    template=template,
    data=data,
    instruction="Choose the correct answer",
    few_shot=["Example: 1+1=2"]
)

print(f"Generated {len(variations)} prompt variations")
```

## Template Format

Templates use Python f-string syntax with custom variation annotations:

```python
"{instruction:semantic}: {few_shot}\n Question: {question:paraphrase}\n Options: {options:non-semantic}"
```

Supported variation types:
- `:semantic` - Semantic variations (meaning-preserving)
- `:paraphrase` - Paraphrasing variations
- `:non-semantic` - Non-semantic variations (formatting, etc.)
- `:lexical` - Word choice variations
- `:syntactic` - Sentence structure variations
- `:surface` - Surface-level formatting variations

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
  - Literals (strings/numbers): Applied to entire dataset
  - Lists: Applied per sample/row
  - Few-shot examples: Flexible list or tuple formats

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
multipromptify --template "{instruction:semantic}: {question:paraphrase}" \
               --data data.csv \
               --instruction "Answer the question"

# With output file
multipromptify --template "{instruction}: {question:paraphrase}" \
               --data data.csv \
               --instruction "Answer this" \
               --output variations.json

# Specify number of variations
multipromptify --template "{instruction:semantic}: {question}" \
               --data data.csv \
               --instruction "Solve this" \
               --max-variations 50
```

### Advanced Options

```bash
# With few-shot examples from file
multipromptify --template "{instruction}: {few_shot}\n{question:paraphrase}" \
               --data data.csv \
               --instruction "Answer the question" \
               --few-shot-file examples.txt \
               --few-shot-count 3

# Output to HuggingFace dataset format
multipromptify --template "{instruction:semantic}: {question}" \
               --data data.csv \
               --instruction "Solve this" \
               --output-format hf \
               --output dataset_variations/
```

## API Reference

### MultiPromptify Class

```python
class MultiPromptify:
    def __init__(self, max_variations: int = 100):
        """Initialize MultiPromptify generator."""
        
    def generate_variations(
        self,
        template: str,
        data: Union[pd.DataFrame, str, dict],
        instruction: str = None,
        few_shot: Union[list, tuple] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Generate prompt variations based on template."""
        
    def parse_template(self, template: str) -> Dict[str, str]:
        """Parse template to extract columns and variation types."""
        
    def save_variations(
        self,
        variations: List[Dict[str, Any]],
        output_path: str,
        format: str = "json"
    ):
        """Save variations to file."""
```

## Examples

### Sentiment Analysis

```python
import pandas as pd
from multipromptify import MultiPromptify

data = pd.DataFrame({
    'text': ['I love this movie!', 'This book is terrible.'],
    'label': ['positive', 'negative']
})

template = "{instruction:semantic}: '{text:paraphrase}'\nSentiment: {label}"

mp = MultiPromptify()
variations = mp.generate_variations(
    template=template,
    data=data,
    instruction="Classify the sentiment of the following text"
)
```

### Question Answering with Few-shot

```python
template = "{instruction:paraphrase}: {few_shot}\n\nQuestion: {question:semantic}\nAnswer:"

few_shot_examples = [
    "Q: What is the capital of France? A: Paris",
    "Q: What is 2+2? A: 4"
]

variations = mp.generate_variations(
    template=template,
    data=qa_data,
    instruction="Answer the following question",
    few_shot=few_shot_examples
)
```

### Multiple Choice

```python
template = "{instruction:semantic}:\n\n{context:paraphrase}\n\nQuestion: {question}\nOptions:\n{options:non-semantic}\n\nAnswer:"

variations = mp.generate_variations(
    template=template,
    data=mc_data,
    instruction="Choose the best answer"
)
```

## Web UI Screenshots

The MultiPromptify 2.0 web interface provides an intuitive workflow:

1. **Data Upload**: Upload CSV/JSON files or select from sample datasets
2. **Template Builder**: Create templates with smart suggestions and real-time validation
3. **Generation**: Configure settings and watch real-time progress
4. **Results**: Analyze, search, filter, and export your variations

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details. 