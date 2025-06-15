# MultiPromptify Python API

The MultiPromptify Python API provides a clean, programmatic interface for generating prompt variations without using the Streamlit web interface. This allows for easy integration into scripts, notebooks, and other Python applications.

## Installation

The API uses the existing MultiPromptify codebase. Make sure you have all dependencies installed:

```bash
pip install pandas
pip install datasets  # Optional: for HuggingFace dataset loading
pip install python-dotenv  # Optional: for environment variable loading
```

## Quick Start

```python
from multipromptify import MultiPromptifyAPI
import pandas as pd

# Initialize
mp = MultiPromptifyAPI()

# Load data
data = [
    {"question": "What is 2+2?", "answer": "4"},
    {"question": "What is the capital of France?", "answer": "Paris"}
]
mp.load_dataframe(pd.DataFrame(data))

# Configure template
template = {
    'instruction_template': 'Question: {question}\nAnswer: {answer}',
    'question': ['surface'],
    'gold': 'answer'
}
mp.set_template(template)

# Configure generation
mp.configure(max_rows=2, variations_per_field=3, max_variations=10)

# Generate variations
variations = mp.generate(verbose=True)

# Export results
mp.export("output.json", format="json")
```

## API Reference

### Initialization

```python
mp = MultiPromptifyAPI()
```

### Data Loading Methods

#### `load_dataset(dataset_name, split="train", **kwargs)`
Load data from HuggingFace datasets library.

```python
mp.load_dataset("squad", split="train")
mp.load_dataset("glue", "mrpc", split="validation")
```

#### `load_csv(filepath, **kwargs)`
Load data from CSV file.

```python
mp.load_csv("data.csv")
mp.load_csv("data.csv", encoding="utf-8")
```

#### `load_json(filepath, **kwargs)`
Load data from JSON file.

```python
mp.load_json("data.json")
```

#### `load_dataframe(df)`
Load data from pandas DataFrame.

```python
df = pd.read_csv("data.csv")
mp.load_dataframe(df)
```

### Template Configuration

#### `set_template(template_dict)`
Set the template configuration using dictionary format.

```python
template = {
    'instruction_template': 'Answer the question: {question}\nAnswer: {answer}',
    'instruction': ['paraphrase'],           # Vary the instruction
    'question': ['surface'],                 # Apply surface variations to question
    'options': ['shuffle', 'surface'],       # Shuffle and vary options
    'gold': {                                # Gold answer configuration
        'field': 'answer',
        'type': 'index',                     # 'value' or 'index'
        'options_field': 'options'
    },
    'few_shot': {                           # Few-shot configuration
        'count': 2,
        'format': 'fixed',                   # 'fixed' or 'rotating'
        'split': 'all'                       # 'all', 'train', or 'test'
    }
}
mp.set_template(template)
```

### Generation Configuration

#### `configure(**kwargs)`
Configure generation parameters.

```python
mp.configure(
    max_rows=10,                    # Maximum rows from data to use
    variations_per_field=3,         # Variations per field
    max_variations=50,              # Maximum total variations
    random_seed=42,                 # For reproducibility
    api_platform="TogetherAI",      # AI platform: "TogetherAI" or "OpenAI"
    api_key="your_api_key",         # For paraphrase variations (optional if env var set)
    model_name="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
)
```

**Platform-specific API Key Selection:**
- When `api_platform="TogetherAI"` → uses `TOGETHER_API_KEY` environment variable
- When `api_platform="OpenAI"` → uses `OPENAI_API_KEY` environment variable
- You can override with explicit `api_key` parameter

### Generation

#### `generate(verbose=False)`
Generate variations with optional progress logging.

```python
# Generate with progress messages
variations = mp.generate(verbose=True)

# Generate silently
variations = mp.generate()
```

### Results and Export

#### `get_results()`
Get generated variations as Python list.

```python
variations = mp.get_results()
```

#### `get_stats()`
Get generation statistics dictionary.

```python
stats = mp.get_stats()
print(f"Generated {stats['total_variations']} variations")
```

#### `export(filepath, format="json")`
Export results to file.

```python
mp.export("output.json", format="json")
mp.export("output.csv", format="csv")
mp.export("output.txt", format="txt")
mp.export("output_conversation.txt", format="conversation")
```

### Utility Methods

#### `info()`
Print current configuration and status information.

```python
mp.info()
```

## Template Format

The API uses dictionary templates with the following structure:

### Required Fields

- `instruction_template`: The base template with placeholders (e.g., `"Question: {question}\nAnswer: {answer}"`)

### Optional Fields

- Field names with variation lists (e.g., `'question': ['surface', 'paraphrase']`)
- `gold`: Gold answer configuration for tracking correct answers
- `few_shot`: Few-shot examples configuration
- `instruction`: Variations to apply to the instruction template itself

### Variation Types

- `surface`: Surface-level text variations (synonyms, etc.)
- `paraphrase`: AI-generated paraphrases (requires API key)
- `context`: Context-based variations
- `shuffle`: Shuffle options/elements (for multiple choice)

### Gold Field Configuration

For tracking correct answers in multiple choice questions:

```python
'gold': {
    'field': 'answer',           # Column containing the correct answer
    'type': 'index',             # 'value' for exact match, 'index' for position
    'options_field': 'options'   # Column containing the options (for index type)
}
```

### Few-Shot Configuration

For adding few-shot examples:

```python
'few_shot': {
    'count': 2,                  # Number of examples
    'format': 'fixed',           # 'fixed' (same examples) or 'rotating' (different)
    'split': 'all'               # 'all', 'train', or 'test'
}
```

## Advanced Examples

### Multiple Choice Questions

```python
template = {
    'instruction_template': '''Answer the following multiple choice question:
Question: {question}
Options: {options}
Answer: {answer}''',
    'instruction': ['paraphrase'],
    'question': ['surface'],
    'options': ['shuffle', 'surface'],
    'gold': {
        'field': 'answer',
        'type': 'index',
        'options_field': 'options'
    },
    'few_shot': {
        'count': 2,
        'format': 'fixed',
        'split': 'all'
    }
}
```

### Reading Comprehension

```python
template = {
    'instruction_template': '''Context: {context}
Question: {question}
Answer: {answer}''',
    'instruction': ['paraphrase'],
    'context': ['surface'],
    'question': ['surface', 'paraphrase'],
    'gold': 'answer',
    'few_shot': {
        'count': 1,
        'format': 'rotating',
        'split': 'train'
    }
}
```

### Simple Q&A

```python
template = {
    'instruction_template': 'Q: {question}\nA: {answer}',
    'question': ['surface'],
    'gold': 'answer'
}
```

## Error Handling

The API provides clear error messages for common issues:

- **No data loaded**: Use one of the `load_*` methods first
- **No template set**: Use `set_template()` first
- **Invalid template**: Template validation errors with specific details
- **Missing API key**: Platform-specific warning when paraphrase variations are used without API key
- **Invalid platform**: Only "TogetherAI" and "OpenAI" are supported
- **File not found**: Clear file path errors for CSV/JSON loading
- **Invalid format**: Export format validation

## Platform and API Key Configuration

### Automatic Platform-based API Key Selection

The API automatically selects the correct environment variable based on your chosen platform:

```python
# Using TogetherAI (default)
mp.configure(api_platform="TogetherAI")  # Uses TOGETHER_API_KEY env var

# Using OpenAI
mp.configure(api_platform="OpenAI")      # Uses OPENAI_API_KEY env var
```

### Setting API Keys (3 ways):

1. **Environment Variable** (Recommended):
```bash
# For TogetherAI
export TOGETHER_API_KEY="your_together_api_key"

# For OpenAI
export OPENAI_API_KEY="your_openai_api_key"
```

2. **`.env` File**:
```bash
# Create .env file in your project root
echo "TOGETHER_API_KEY=your_together_api_key" > .env
# or
echo "OPENAI_API_KEY=your_openai_api_key" > .env
```

3. **Programmatically**:
```python
mp.configure(api_key="your_api_key")
```

### Platform Switching

You can easily switch between platforms:

```python
# Start with TogetherAI
mp.configure(api_platform="TogetherAI")

# Switch to OpenAI
mp.configure(api_platform="OpenAI")  # Automatically uses OPENAI_API_KEY

# Override with specific key
mp.configure(api_platform="OpenAI", api_key="custom_key")
```

## Progress Logging

When `verbose=True` is used with `generate()`, you'll see progress messages:

```
🚀 Starting MultiPromptify generation...
   Using platform: TogetherAI
🔄 Step 1/5: Initializing MultiPromptify...
📊 Step 2/5: Preparing data... (using first 10 rows)
⚙️ Step 3/5: Configuring generation parameters...
⚡ Step 4/5: Generating variations... (AI is working on variations)
📈 Step 5/5: Computing statistics...
✅ Generated 25 variations in 12.3 seconds
```

## Integration with Existing Code

The API is designed to be identical in functionality to the Streamlit interface. Results generated programmatically will be the same as those from the web interface when using identical templates and data.

You can easily migrate from the web interface to the API by:

1. Copying your template configuration from the Streamlit interface
2. Loading your data using the appropriate `load_*` method
3. Using the same generation parameters
4. Running `generate()` to get identical results 