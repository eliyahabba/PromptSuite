# PromptSuite Python API

The PromptSuite Python API provides a clean, programmatic interface for generating prompt variations without using the Streamlit web interface. This allows for easy integration into scripts, notebooks, and other Python applications.

## Installation

The API uses the existing PromptSuite codebase. Make sure you have all dependencies installed:

```bash
pip install pandas
pip install datasets  # Optional: for HuggingFace dataset loading
pip install python-dotenv  # Optional: for environment variable loading
```

## Quick Start

```python
from promptsuite import PromptSuite
import pandas as pd

# Initialize
ps = PromptSuite()

# Load data
data = [
    {"question": "What is 2+2?", "answer": "4"},
    {"question": "What is the capital of France?", "answer": "Paris"}
]
ps.load_dataframe(pd.DataFrame(data))

# Configure template
template = {
    'instruction_template': 'Question: {question}\nAnswer: {answer}',
    'question': ['format structure'],
    'gold': 'answer'
}
ps.set_template(template)

# Configure generation
ps.configure(max_rows=2, variations_per_field=3, max_variations_per_row=10)

# Generate variations
variations = ps.generate(verbose=True)

# Export results
ps.export("output.json", format="json")
```

## Minimal Example (No gold, no few_shot)

```python
import pandas as pd
from promptsuite import PromptSuite

data = pd.DataFrame({
    'question': ['What is 2+2?', 'What is the capital of France?'],
    'answer': ['4', 'Paris']
})

template = {
    'system_prompt_template': 'Please answer the following questions.',
    'instruction_template': 'Q: {question}\nA: {answer}',
    'question': ['format structure']
}

ps = PromptSuite()
ps.load_dataframe(data)
ps.set_template(template)
ps.configure(max_rows=2, variations_per_field=2)
variations = ps.generate(verbose=True)
print(variations)
```

## Multiple Choice with Dynamic System Prompt and Few-shot

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
    'system_prompt_template': 'The following are multiple choice questions (with answers) about {subject}.',
    'instruction_template': 'Question: {question}\nOptions: {options}\nAnswer:',
    'question': ['format structure'],
    'options': ['shuffle'],
    'gold': {
        'field': 'answer',
        'type': 'index',
        'options_field': 'options'
    },
    'few_shot': {
        'count': 2,
        'format': 'different_examples__different_order_per_variation',
        'split': 'all'
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

## API Reference

### Initialization

```python
ps = PromptSuite()
```

### Data Loading Methods

#### `load_dataset(dataset_name, split="train", **kwargs)`
Load data from HuggingFace datasets library.

```python
ps.load_dataset("squad", split="train")
ps.load_dataset("glue", "mrpc", split="validation")
```

#### `load_csv(filepath, **kwargs)`
Load data from CSV file.

```python
ps.load_csv("data.csv")
ps.load_csv("data.csv", encoding="utf-8")
```

#### `load_json(filepath, **kwargs)`
Load data from JSON file.

```python
ps.load_json("data.json")
```

#### `load_dataframe(df)`
Load data from pandas DataFrame.

```python
df = pd.read_csv("data.csv")
ps.load_dataframe(df)
```

### Template Configuration

#### `set_template(template_dict)`
Set the template configuration using dictionary format.

```python
template = {
    'instruction_template': 'Answer the question: {question}\nAnswer: {answer}',
    'instruction': ['paraphrase_with_llm'],           # Vary the instruction
    'question': ['format structure'],                 # Apply semantic-preserving format changes to question
    'options': ['shuffle', 'typos and noise'],       # Shuffle and add noise to options
    'gold': {                                # Gold answer configuration
        'field': 'answer',
        'type': 'index',                     # 'value' or 'index'
        'options_field': 'options'
    },
    'few_shot': {                           # Few-shot configuration
        'count': 2,
        'format': 'different_examples__different_order_per_variation',        # 'same_examples__no_variations', 'same_examples__synchronized_order_variations', 'different_examples__same_shuffling_order_across_rows', or 'different_examples__different_order_per_variation'
        'split': 'all'                       # 'all', 'train', or 'test'
    }
}
ps.set_template(template)
```

### Generation Configuration

#### `configure(**kwargs)`
Configure generation parameters.

```python
ps.configure(
    max_rows=10,                    # Maximum rows from data to use
    variations_per_field=3,         # Variations per field
    max_variations_per_row=50,      # Maximum variations per row (not global)
    random_seed=42,                 # Random seed for reproducibility
    api_platform="TogetherAI",      # API platform for LLM-based variations
    model_name="meta-llama/Llama-3.1-8B-Instruct-Turbo"  # Model name
)
```

### Generation

#### `generate(verbose=False)`
Generate prompt variations.

```python
variations = ps.generate(verbose=True)
```

### Export

#### `export(filepath, format="json")`
Export variations to file.

```python
ps.export("output.json", format="json")
ps.export("output.csv", format="csv")
ps.export("output.txt", format="txt")
```

## Template Format

Templates use a dictionary format with specific keys for different components:

### Template Keys

- `instruction_template` - The main template string with placeholders
- `instruction` - List of variation types for the instruction
- `prompt_format_variations` - List of variation types for the prompt format
- Field names with variation lists (e.g., `'question': ['format structure', 'paraphrase_with_llm']`)
- `gold` - Gold answer configuration
- `few_shot` - Few-shot examples configuration

### Supported Variation Types

- `paraphrase_with_llm` - Paraphrasing variations (LLM-based)
- `format_structure` (`FORMAT_STRUCTURE_VARIATION`) - Semantic-preserving format changes (e.g., separators, casing, field order)
- `typos and noise` (`TYPOS_AND_NOISE_VARIATION`) - Injects typos, random case, extra whitespace, and punctuation noise for robustness
- `context` - Context-based variations
- `shuffle` - Shuffle options/elements (for multiple choice)
- `enumerate` - Enumerate list fields (e.g., 1. 2. 3. 4., A. B. C. D., roman numerals, etc.)
- `rewording` - (Deprecated, kept for backward compatibility; now maps to `typos and noise`)

### Example Templates

#### Simple QA Template

```python
template = {
    'instruction_template': 'Answer the question: {question}\nAnswer: {answer}',
    'question': ['format structure'],
    'gold': 'answer'
}
```

#### Multiple Choice Template

```python
template = {
    'instruction_template': 'Choose the correct answer:\nQ: {question}\nOptions: {options}\nA: {answer}',
    'question': ['format structure'],
    'options': ['shuffle', 'typos and noise'],
    'gold': {
        'field': 'answer',
        'type': 'index',
        'options_field': 'options'
    }
}
```

#### Complex Template with Context

```python
template = {
    'instruction_template': 'Context: {context}\nQuestion: {question}\nAnswer: {answer}',
    'context': ['format structure'],
    'question': ['format structure', 'paraphrase_with_llm'],
    'gold': {
        'field': 'answer',
        'type': 'value'
    },
    'few_shot': {
        'count': 2,
        'format': 'rotating',
        'split': 'all'
    }
}
```

## Examples

### Sentiment Analysis

```python
import pandas as pd
from promptsuite import PromptSuite

data = pd.DataFrame({
    'text': ['I love this movie!', 'This book is terrible.'],
    'label': ['positive', 'negative']
})

template = {
    'instruction_template': 'Classify the sentiment: "{text}"\nSentiment: {label}',
    'instruction_variations': ['semantic'],
    'text': ['paraphrase_with_llm'],
    'gold': 'label'
}

ps = PromptSuite()
ps.load_dataframe(data)
ps.set_template(template)
ps.configure(
    max_rows=GenerationDefaults.MAX_ROWS,
    variations_per_field=GenerationDefaults.VARIATIONS_PER_FIELD,
    max_variations_per_row=GenerationDefaults.MAX_VARIATIONS_PER_ROW,
    random_seed=GenerationDefaults.RANDOM_SEED,
    api_platform=GenerationDefaults.API_PLATFORM,
    model_name=GenerationDefaults.MODEL_NAME
)
variations = ps.generate(verbose=True)
```

### Question Answering with Few-shot

```python
template = {
    'instruction_template': 'Answer the question:\nQuestion: {question}\nAnswer: {answer}',
    'instruction_variations': ['paraphrase_with_llm'],
    'question': ['semantic'],
    'gold': 'answer',
    'few_shot': {
        'count': 2,
        'format': 'rotating',
        'split': 'all'
    }
}

ps = PromptSuite()
ps.load_dataframe(qa_data)
ps.set_template(template)
ps.configure(
    max_rows=GenerationDefaults.MAX_ROWS,
    variations_per_field=GenerationDefaults.VARIATIONS_PER_FIELD,
    max_variations_per_row=GenerationDefaults.MAX_VARIATIONS_PER_ROW,
    random_seed=GenerationDefaults.RANDOM_SEED,
    api_platform=GenerationDefaults.API_PLATFORM,
    model_name=GenerationDefaults.MODEL_NAME
)
variations = ps.generate(verbose=True)
```

### Multiple Choice with Dynamic System Prompt and Few-shot

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
    'instruction_template': 'The following are multiple choice questions (with answers) about {subject}.\nQuestion: {question}\nOptions: {options}\nAnswer:',
    'question': ['format structure'],
    'options': ['shuffle'],
    'gold': {
        'field': 'answer',
        'type': 'index',
        'options_field': 'options'
    },
    'few_shot': {
        'count': 2,
        'format': 'different_examples__different_order_per_variation',
        'split': 'all'
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

## Advanced Features

### Using Constants for Template Keys

You can import constants to avoid typos and ensure consistency:

```python
from promptsuite.core.template_keys import (
    INSTRUCTION, PROMPT_FORMAT, QUESTION_KEY, OPTIONS_KEY, GOLD_KEY,
    FORMAT_STRUCTURE_VARIATION, TYPOS_AND_NOISE_VARIATION, PARAPHRASE_WITH_LLM
)

template = {
    INSTRUCTION: 'Answer the question: {question}\nAnswer: {answer}',
    PROMPT_FORMAT_VARIATIONS: ['format structure'],
    QUESTION_KEY: ['typos and noise'],
    OPTIONS_KEY: ['shuffle', 'enumerate'],
    GOLD_KEY: {
        'field': 'answer',
        'type': 'index',
        'options_field': 'options'
    }
}
```

### Environment Variables

The API automatically detects API keys from environment variables:

```bash
export TOGETHER_API_KEY="your_together_key"
export OPENAI_API_KEY="your_openai_key"
```

### Platform-Specific Configuration

```python
# Use TogetherAI
ps.configure(api_platform="TogetherAI", model_name="meta-llama/Llama-3.1-8B-Instruct-Turbo")

# Use OpenAI
ps.configure(api_platform="OpenAI", model_name="gpt-3.5-turbo")
```

## Troubleshooting

### Common Issues

1. **No API key found**: Set your API key in environment variables
2. **Template parsing errors**: Check template syntax and field names
3. **Too many variations**: Reduce `variations_per_field` or `max_variations_per_row`
4. **Memory issues**: Process data in smaller chunks

### Debug Mode

Enable verbose output to see detailed information:

```python
variations = ps.generate(verbose=True)
```

### Export for Analysis

Export results to analyze the generated variations:

```python
ps.export("debug_output.json", format="json")
```

## Best Practices

1. **Start simple**: Begin with basic templates and add complexity gradually
2. **Use constants**: Import template keys to avoid typos
3. **Test with small data**: Validate templates with small datasets first
4. **Monitor variation count**: Be aware of combinatorial explosion with multiple variation types
5. **Use appropriate variation types**: Choose variation types that match your use case
   - **format_structure**: For semantic-preserving format changes
   - **typos and noise**: For robustness testing
   - **paraphrase_with_llm**: For semantic variations (requires API key)

## Variation Types Reference

### Core Variation Types

- **paraphrase_with_llm**: Uses LLM to generate semantic variations of the text
- **format structure** (`FORMAT_STRUCTURE_VARIATION`): Applies semantic-preserving format changes to the prompt, such as changing separators, field order, or casing, inspired by the FORMATSPREAD paper.
- **typos and noise** (`TYPOS_AND_NOISE_VARIATION`): Injects various types of noise (typos, character swaps, random case, extra whitespace, punctuation) for robustness testing, while protecting placeholders.

### Utility Variation Types

- **context**: Adds contextual information to questions
- **shuffle**: Randomly reorders elements (e.g., multiple choice options)
- **enumerate**: Adds enumeration to list fields (e.g., 1. 2. 3. 4., A. B. C. D., roman numerals)

### When to Use Each Type

- **format_structure**: When you want to test how different prompt formats affect model performance
- **typos and noise**: When you want to test model robustness to input noise
- **paraphrase_with_llm**: When you want semantic variations that preserve meaning
- **shuffle**: For multiple choice questions to test option order independence
- **enumerate**: To add structure to list fields

### Few-Shot Configuration

Few-shot examples can be configured with different sampling strategies:

| Format | Description | Use Case |
|--------|-------------|----------|
| `same_examples__no_variations` | Same examples for all rows, no variations (single variation per row) | When you want consistent, predictable examples |
| `same_examples__synchronized_order_variations` | Same examples for all rows, synchronized order variations across all rows | When you want consistent examples but test different orderings |
| `different_examples__same_shuffling_order_across_rows` | Different examples per row, same shuffling order across rows | When you want unique examples per question but consistent ordering patterns |
| `different_examples__different_order_per_variation` | Different examples and different order per variation | When you want maximum variety and different examples per question |

**Example:**
```python
"few_shot": {
    "count": 2,                    # Number of examples to use
    "format": "same_examples__synchronized_order_variations",   # Sampling strategy
    "split": "train"               # Use only training data for examples
}
```

### Supported Variation Types 