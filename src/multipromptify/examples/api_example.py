#!/usr/bin/env python3
"""
MultiPromptify API Example Script

This script demonstrates how to use the MultiPromptifier class for programmatic
generation of prompt variations.
"""
import os

import pandas as pd

from multipromptify import MultiPromptifier
from multipromptify.core.template_keys import (
    PROMPT_FORMAT_VARIATIONS, QUESTION_KEY, GOLD_KEY, FEW_SHOT_KEY, OPTIONS_KEY, CONTEXT_KEY,
    PARAPHRASE_WITH_LLM, REWORDING, CONTEXT_VARIATION, SHUFFLE_VARIATION, ENUMERATE_VARIATION,
    PROMPT_FORMAT, INSTRUCTION_VARIATIONS, INSTRUCTION
)


def example_with_sample_data_few_shot():
    # Create instance
    mp = MultiPromptifier()

    # Load data with at least 4 examples for few-shot
    data = pd.DataFrame({
        'question': [
            'What is 2+2?',
            'What is 5+3?',
            'What is 10-4?',
            'What is 3*3?',
            'What is 20/4?'
        ],
        'answer': ['4', '8', '6', '9', '5']
    })
    mp.load_dataframe(data)

    # Set template with few-shot configuration
    template = {
        INSTRUCTION: 'The following are multiple choice questions (with answers) about math.',
        PROMPT_FORMAT: 'Question: {question}\nAnswer: {answer}',
        'question': [REWORDING],  # surface variations
        'gold': 'answer',
        'few_shot': {
            'count': 2,  # Use 2 examples
            'format': 'rotating',  # Different examples each time
            'split': 'all'  # Use all data for examples
        }
    }
    mp.set_template(template)

    # Configure and generate
    mp.configure(max_rows=4, variations_per_field=2)
    variations = mp.generate(verbose=True)

    # Display results with few-shot examples
    print(f"\n✅ Generated {len(variations)} variations")
    print("\n" + "=" * 50)

    # Show first few variations to see few-shot in action
    for i, var in enumerate(variations[:3]):
        print(f"\nVariation {i + 1}:")
        print("-" * 50)
        print(var['prompt'])
        print("-" * 50)

    # Export results
    mp.export("few_shot_examples.json", format="json")
    print("\n✅ Exported to few_shot_examples.json")

    # Show info
    mp.info()


def example_with_enumerate():
    """Example demonstrating the new enumerate functionality."""

    print("🚀 MultiPromptify API Example with Enumerate")
    print("=" * 50)

    # Initialize the API
    mp = MultiPromptifier()

    # Create sample data
    sample_data = [
        {
            "question": "What is the capital of France?",
            "options": ["London", "Berlin", "Paris", "Madrid"],
            "answer": 2  # Paris is at index 2
        },
        {
            "question": "What is 2+2?",
            "options": ["3", "4", "5", "6"],
            "answer": 1  # 4 is at index 1
        },
        {
            "question": "Which planet is closest to the Sun?",
            "options": ["Venus", "Mercury", "Earth", "Mars"],
            "answer": 1  # Mercury is at index 1
        }
    ]

    df = pd.DataFrame(sample_data)

    # Load the data
    print("\n1. Loading data...")
    mp.load_dataframe(df)
    print("📝 Data format: answers are indices (0-based), not text values")

    # Configure template with enumerate
    print("\n2. Setting template with enumerate...")
    template = {
        INSTRUCTION: 'The following are multiple choice questions (with answers) about general knowledge.',
        PROMPT_FORMAT: 'Question: {question}\nOptions: {options}\nAnswer: {answer}',
        QUESTION_KEY: [REWORDING],
        GOLD_KEY: {
            'field': 'answer',
            'type': 'index',
            'options_field': 'options'
        },
        ENUMERATE_VARIATION: {
            'field': 'options',  # Which field to enumerate
            'type': '1234'  # Use numbers: 1. 2. 3. 4.
        }
    }

    mp.set_template(template)
    print("✅ Template configured with enumerate field")
    print("   - Will enumerate 'options' field with numbers (1234)")

    # Configure generation parameters
    print("\n3. Configuring generation...")
    mp.configure(
        max_rows=3,
        variations_per_field=2,
        max_variations=10,
        random_seed=42
    )

    # Show current status
    print("\n4. Current status:")
    mp.info()

    # Generate variations
    print("\n5. Generating variations...")
    variations = mp.generate(verbose=True)

    # Show results
    print(f"\n6. Results: Generated {len(variations)} variations")

    # Display first few variations to see enumerate in action
    for i, variation in enumerate(variations[:7]):
        print(f"\nVariation {i + 1}:")
        print("-" * 50)
        print(variation.get('prompt', 'No prompt found'))
        print("-" * 50)

    # Export results
    print("\n8. Exporting results...")
    mp.export("enumerate_example.json", format="json")

    print("\n✅ Enumerate example completed successfully!")


def example_enumerate_types():
    """Example showing different enumerate types."""

    print("\n" + "=" * 50)
    print("🔢 Different Enumerate Types Example")
    print("=" * 50)

    mp = MultiPromptifier()

    # Simple data
    data = [{
        "question": "Which is correct?",
        "options": ["Option A", "Option B", "Option C", "Option D"],
        "answer": 0
    }]
    mp.load_dataframe(pd.DataFrame(data))

    # Test different enumerate types
    enumerate_types = [
        ("1234", "Numbers"),
        ("ABCD", "Uppercase letters"),
        ("abcd", "Lowercase letters"),
        ("roman", "Roman numerals")
    ]

    for enum_type, description in enumerate_types:
        print(f"\n--- {description} ({enum_type}) ---")

        template = {
            INSTRUCTION: 'The following are multiple choice questions (with answers) about general knowledge.',
            PROMPT_FORMAT: 'Question: {question}\nOptions: {options}\nAnswer: {answer}',
            QUESTION_KEY: [REWORDING],
            GOLD_KEY: {
                'field': 'answer',
                'type': 'index',
                'options_field': 'options'
            },
            ENUMERATE_VARIATION: {
                'field': 'options',
                'type': enum_type
            }
        }

        mp.set_template(template)
        mp.configure(max_rows=1, variations_per_field=1, max_variations=1)

        try:
            variations = mp.generate(verbose=False)
            if variations:
                print("Result:")
                print(variations[0].get('prompt', 'No prompt'))
        except Exception as e:
            print(f"Error with {enum_type}: {e}")


def example_with_sample_data():
    """Example using sample data with different template configurations."""

    print("🚀 MultiPromptify API Example")
    print("=" * 50)

    # Initialize the API
    mp = MultiPromptifier()

    # Create sample data - NOTE: answers are indices (0-based) not the actual text
    # because we use 'type': 'index' in the gold configuration
    sample_data = [
        {
            "question": "What is the capital of France?",
            "options": ["London", "Berlin", "Paris", "Madrid"],
            "answer": 2  # Paris is at index 2
        },
        {
            "question": "What is 2+2?",
            "options": ["3", "4", "5", "6"],
            "answer": 1  # 4 is at index 1
        },
        {
            "question": "Which planet is closest to the Sun?",
            "options": ["Venus", "Mercury", "Earth", "Mars"],
            "answer": 1  # Mercury is at index 1
        }
    ]

    df = pd.DataFrame(sample_data)

    # Load the data
    print("\n1. Loading data...")
    mp.load_dataframe(df)
    print("📝 Data format: answers are indices (0-based), not text values")
    print("   Example: Paris = index 2 in ['London', 'Berlin', 'Paris', 'Madrid']")

    # Configure template (dictionary format)
    print("\n2. Setting template...")
    template = {
        INSTRUCTION: 'The following are multiple choice questions (with answers) about general knowledge.',
        PROMPT_FORMAT: 'Question: {question}\nOptions: {options}\nAnswer: {answer}',
        INSTRUCTION_VARIATIONS: [REWORDING],
        PROMPT_FORMAT_VARIATIONS: [REWORDING],
        GOLD_KEY: {
            'field': 'answer',
            'type': 'index',  # This means answer field contains indices, not text
            'options_field': 'options'
        },
        # FEW_SHOT_KEY: {
        #     'count': 2,
        #     'format': 'fixed',
        #     'split': 'all'
        # }
    }

    mp.set_template(template)

    # Configure generation parameters
    print("\n3. Configuring generation...")
    mp.configure(
        max_rows=1,  # Use first 3 rows (need at least 3 for few_shot count=2)
        variations_per_field=2,  # 3 variations per field
        max_variations=20,  # Maximum 20 total variations
        random_seed=42,  # For reproducibility
        api_platform="TogetherAI",  # Platform selection
        model_name="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
    )

    # Show current status
    print("\n4. Current status:")
    mp.info()

    # Generate variations
    print("\n5. Generating variations...")
    variations = mp.generate(verbose=True)

    # Show results
    print(f"\n6. Results: Generated {len(variations)} variations")

    # Display first few variations
    for i, variation in enumerate(variations[:3]):
        print(f"\nVariation {i + 1}:")
        print("-" * 40)
        print(variation.get('prompt', 'No prompt found'))
        print()

    # Get statistics
    stats = mp.get_stats()
    if stats:
        print("\n7. Statistics:")
        for key, value in stats.items():
            print(f"   {key}: {value}")

    # Export results
    print("\n8. Exporting results...")
    mp.export("output_example.json", format="json")
    mp.export("output_example.csv", format="csv")

    print("\n✅ Example completed successfully!")


def example_platform_switching():
    """Example showing how to switch between AI platforms."""

    print("\n" + "=" * 50)
    print("🔄 Platform Switching Example")
    print("=" * 50)

    # Initialize API
    mp = MultiPromptifier()

    # Create simple data
    data = [{"question": "What is AI?", "answer": "Artificial Intelligence"}]
    mp.load_dataframe(pd.DataFrame(data))

    # Simple template with paraphrase (requires API key)
    template = {
        INSTRUCTION: 'The following are multiple choice questions (with answers) about general knowledge.',
        PROMPT_FORMAT: 'Question: {question}\nAnswer: {answer}',
        QUESTION_KEY: [PARAPHRASE_WITH_LLM],
        GOLD_KEY: 'answer'  # Simple format - just the field name
    }
    mp.set_template(template)

    print("\n1. Default platform (TogetherAI):")
    mp.info()

    print("\n2. Switching to OpenAI:")
    mp.configure(api_platform="OpenAI")
    mp.info()

    print("\n3. Back to TogetherAI with custom model:")
    mp.configure(
        api_platform="TogetherAI",
        model_name="meta-llama/Llama-3.1-8B-Instruct-Turbo"
    )
    mp.info()

    print("\n4. Manual API key override:")
    mp.configure(api_key="manual_key_override")
    mp.info()


def example_with_huggingface():
    """Example using HuggingFace datasets (SQuAD) with classic QA template and gold field expression extraction."""
    print("\n" + "=" * 50)
    print("🤗 HuggingFace Dataset Example (SQuAD, zero-shot, classic QA, gold field expression)")
    print("=" * 50)

    try:
        from multipromptify import MultiPromptifier
        mp = MultiPromptifier()

        # Load 3 examples from SQuAD directly
        print("\n1. Loading SQuAD dataset (3 samples)...")
        mp.load_dataset("rajpurkar/squad", split="train[:3]")

        # Classic QA template with gold field expression for SQuAD
        template = {
            INSTRUCTION: 'The following are multiple choice questions (with answers) about general knowledge.',
            PROMPT_FORMAT: 'Read the context and answer the question.\\nContext: {context}\\nQuestion: {question}\\nAnswer:',
            CONTEXT_KEY: [REWORDING],  # Reword the context
            QUESTION_KEY: [],
            GOLD_KEY: "answers['text'][0]"
        }
        mp.set_template(template)
        mp.configure(max_rows=3, variations_per_field=1, max_variations=1)

        print("\n2. Generating variations...")
        variations = mp.generate(verbose=True)

        print(f"\n✅ Generated {len(variations)} variations\n")
        for i, v in enumerate(variations):
            print(f"Prompt {i + 1}:")
            print(v["prompt"])
            # print("Expected answer:", v["answers['text'][0]"])
            print("-" * 40)

    except Exception as e:
        print(f"❌ HuggingFace example failed: {e}")


def example_different_templates():
    """Examples showing different template configurations."""

    print("\n" + "=" * 50)
    print("📝 Different Template Examples")
    print("=" * 50)

    # Simple QA template (text-based answers)
    simple_template = {
        INSTRUCTION: 'The following are multiple choice questions (with answers) about general knowledge.',
        PROMPT_FORMAT: 'Question: {question}\nAnswer: {answer}',
        QUESTION_KEY: [REWORDING],
        GOLD_KEY: 'answer'  # Simple format for text answers
    }

    # Multiple choice template (index-based answers)
    multiple_choice_template = {
        INSTRUCTION: 'The following are multiple choice questions (with answers) about general knowledge.',
        PROMPT_FORMAT: 'Choose the correct answer:\nQ: {question}\nOptions: {options}\nA: {answer}',
        QUESTION_KEY: [REWORDING],
        OPTIONS_KEY: [REWORDING, SHUFFLE_VARIATION],
        GOLD_KEY: {
            'field': 'answer',
            'type': 'index',  # Answer is index in options
            'options_field': 'options'
        }
    }

    # Complex template with multiple variations
    complex_template = {
        INSTRUCTION: 'The following are multiple choice questions (with answers) about general knowledge.',
        PROMPT_FORMAT: 'Context: {context}\nQuestion: {question}\nAnswer: {answer}',
        CONTEXT_KEY: [REWORDING, PARAPHRASE_WITH_LLM],
        QUESTION_KEY: [REWORDING],
        GOLD_KEY: {
            'field': 'answer',
            'type': 'value'  # Answer is text value
        },
        FEW_SHOT_KEY: {
            'count': 1,
            'format': 'rotating',
            'split': 'all'
        }
    }

    # Platform-specific template with different configurations
    platform_templates = {
        'TogetherAI': {
            INSTRUCTION: 'The following are multiple choice questions (with answers) about general knowledge.',
            PROMPT_FORMAT: 'Using Llama model: {question}\nAnswer: {answer}',
            QUESTION_KEY: [REWORDING],
            GOLD_KEY: 'answer'
        },
        'OpenAI': {
            INSTRUCTION: 'The following are multiple choice questions (with answers) about general knowledge.',
            PROMPT_FORMAT: 'Using GPT model: {question}\nAnswer: {answer}',
            QUESTION_KEY: [REWORDING],
            GOLD_KEY: 'answer'
        }
    }

    print("Simple template structure (text answers):")
    for key, value in simple_template.items():
        print(f"   {key}: {value}")

    print("\nMultiple choice template (index answers):")
    for key, value in multiple_choice_template.items():
        print(f"   {key}: {value}")

    print("\nComplex template structure:")
    for key, value in complex_template.items():
        print(f"   {key}: {value}")

    print("\nPlatform-specific templates:")
    for platform, template in platform_templates.items():
        print(f"\n{platform} template:")
        for key, value in template.items():
            print(f"   {key}: {value}")


def example_gold_field_formats():
    """Example showing different gold field configuration formats."""

    print("\n" + "=" * 50)
    print("🏆 Gold Field Configuration Examples")
    print("=" * 50)

    # Example data for different formats
    print("1. Index-based multiple choice data:")
    index_data = [
        {
            "question": "What color is the sky?",
            "options": ["Red", "Blue", "Green", "Yellow"],
            "answer": 1  # Blue (index 1)
        }
    ]
    print("   Data:", index_data[0])

    index_template = {
        INSTRUCTION: 'The following are multiple choice questions (with answers) about general knowledge.',
        PROMPT_FORMAT: 'Q: {question}\nOptions: {options}\nA: {answer}',
        GOLD_KEY: {
            'field': 'answer',
            'type': 'index',
            'options_field': 'options'
        }
    }
    print("   Template gold config:", index_template[GOLD_KEY])

    print("\n2. Value-based multiple choice data:")
    value_data = [
        {
            "question": "What color is the sky?",
            "options": ["Red", "Blue", "Green", "Yellow"],
            "answer": "Blue"  # Text value
        }
    ]
    print("   Data:", value_data[0])

    value_template = {
        INSTRUCTION: 'The following are multiple choice questions (with answers) about general knowledge.',
        PROMPT_FORMAT: 'Q: {question}\nOptions: {options}\nA: {answer}',
        GOLD_KEY: {
            'field': 'answer',
            'type': 'value',
            'options_field': 'options'
        }
    }
    print("   Template gold config:", value_template[GOLD_KEY])


def example_environment_variables():
    """Example showing how to work with environment variables."""

    print("\n" + "=" * 50)
    print("🌍 Environment Variables Example")
    print("=" * 50)

    import os

    # Show current environment variables
    print("Current API key environment variables:")
    together_key = os.getenv("TOGETHER_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    print(f"   TOGETHER_API_KEY: {'✅ Set' if together_key else '❌ Not set'}")
    print(f"   OPENAI_API_KEY: {'✅ Set' if openai_key else '❌ Not set'}")

    # Initialize API and show how keys are automatically selected
    mp = MultiPromptifier()

    print(f"\nDefault platform API key detection:")
    print(f"   Platform: {mp.config['api_platform']}")
    print(f"   API Key: {'✅ Found' if mp.config['api_key'] else '❌ Not found'}")
    # Test platform switching
    print(f"\nTesting platform switching:")
    for platform in ["TogetherAI", "OpenAI"]:
        mp.configure(api_platform=platform)
        key_found = mp.config['api_key'] is not None
        print(f"   {platform}: {'✅ API key found' if key_found else '❌ No API key'}")


def example_with_simple_qa():
    """Example loading 5 examples from simple_qa_test.csv (simple QA format)."""
    import os
    print("\n" + "=" * 50)
    print("📄 Simple QA CSV Example (simple_qa_test.csv)")
    print("=" * 50)

    # Path to the CSV file
    csv_path = os.path.join(os.path.dirname(__file__), '../../../data/simple_qa_test.csv')
    csv_path = os.path.abspath(csv_path)

    # Load the first 5 rows from the CSV
    df = pd.read_csv(csv_path).head(5)
    print(f"Loaded {len(df)} rows from {csv_path}")
    print(df[['problem', 'answer']])

    # Initialize the API
    mp = MultiPromptifier()
    mp.load_dataframe(df)

    # Set a simple QA template
    template = {
        INSTRUCTION: 'The following are multiple choice questions (with answers) about general knowledge.',
        PROMPT_FORMAT: 'Question: {problem}\nAnswer: {answer}',
        GOLD_KEY: 'answer',
        FEW_SHOT_KEY: {
            'count': 2,
            'format': 'rotating',
            'split': 'all'
        }
    }
    mp.set_template(template)

    # Configure and generate
    mp.configure(max_rows=5, variations_per_field=2)
    variations = mp.generate(verbose=True)

    print(f"\n✅ Generated {len(variations)} variations from simple_qa_test.csv")
    print("\n" + "=" * 50)
    for i, var in enumerate(variations[:3]):
        print(f"\nVariation {i + 1}:")
        print("-" * 50)
        print(var['prompt'])
        print("-" * 50)

    # Export results
    mp.export("simple_qa_example.json", format="json")
    print("\n✅ Exported to simple_qa_example.json")
    mp.info()


def example_answer_the_question_prompt_only():
    """Example: Prompt instructs to answer the question, but does not include the answer (no gold, no few-shot)."""
    print("\n" + "=" * 50)
    print("📝 Example: 'Answer the question' Prompt Only (No Gold, No Few-shot)")
    print("=" * 50)

    # Sample data: question + answer, but we use only the question in the prompt
    data = [
        {"question": "What is the capital of France?", "answer": "Paris"},
        {"question": "How many days are in a week?", "answer": "7"},
        {"question": "Who wrote Romeo and Juliet?", "answer": "Shakespeare"}
    ]
    import pandas as pd
    from multipromptify import MultiPromptifier

    df = pd.DataFrame(data)
    mp = MultiPromptifier()
    mp.load_dataframe(df)

    # Template: instructs to answer the question, but does not include the answer
    template = {
        INSTRUCTION: 'The following are multiple choice questions (with answers) about general knowledge.',
        PROMPT_FORMAT: 'Please answer the following question:\n{question}',
        QUESTION_KEY: [REWORDING]
    }
    mp.set_template(template)

    mp.configure(max_rows=3, variations_per_field=2)
    variations = mp.generate(verbose=True)

    print(f"\n✅ Generated {len(variations)} variations (prompt only, no gold, no few-shot)")
    for i, var in enumerate(variations[:3]):
        print(f"\nVariation {i + 1}:")
        print("-" * 50)
        print(var['prompt'])
        print("-" * 50)

    mp.export("answer_the_question_prompt_only.json", format="json")
    print("\n✅ Exported to answer_the_question_prompt_only.json")
    mp.info()


def example_with_system_prompt_few_shot():
    mp = MultiPromptifier()
    data = pd.DataFrame({
        'question': ['What is 2+2?', 'What is 3*3?', 'What is 5+3?'],
        'answer': ['4', '9', '8']
    })
    mp.load_dataframe(data)
    template = {
        INSTRUCTION: 'You are a helpful math assistant. Answer clearly.',
        PROMPT_FORMAT: 'Question: {question}\nAnswer: {answer}',
        QUESTION_KEY: [REWORDING],
        GOLD_KEY: 'answer',
        FEW_SHOT_KEY: {
            'count': 2,
            'format': 'rotating',
            'split': 'all'
        }
    }
    mp.set_template(template)
    mp.configure(max_rows=3, variations_per_field=1)
    variations = mp.generate(verbose=True)
    print("\n=== System Prompt Few-shot Example ===")
    for v in variations:
        print(v['prompt'])
        print("--- Conversation:")
        for msg in v['conversation']:
            print(f"[{msg['role']}] {msg['content']}")
        print("====================\n")


def example_system_prompt_with_placeholder():
    print("\n=== System Prompt with Placeholder Example ===")
    mp = MultiPromptifier()
    data = pd.DataFrame({
        'question': [
            'What is the largest planet in our solar system?',
            'Which chemical element has the symbol O?',
            'What is the fastest land animal?',
            'What is the smallest prime number?',
            'Which continent is known as the \"Dark Continent\"?'
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
    mp.load_dataframe(data)
    template = {
        INSTRUCTION: 'The following are multiple choice questions (with answers) about {subject}.',
        PROMPT_FORMAT: 'Question: {question}\nOptions: {options}\nAnswer:',
        QUESTION_KEY: [REWORDING],
        OPTIONS_KEY: ['shuffle'],
        GOLD_KEY: {
            'field': 'answer',
            'type': 'index',
            'options_field': 'options'
        }
    }
    mp.set_template(template)
    mp.configure(max_rows=5, variations_per_field=1)
    variations = mp.generate(verbose=True)
    for i, var in enumerate(variations):
        print(f"\nVariation {i + 1}:")
        print("-" * 50)
        print(var['prompt'])
        print("-" * 50)


def example_system_prompt_with_placeholder_and_few_shot():
    print("\n=== System Prompt with Placeholder + Few-shot Example ===")
    mp = MultiPromptifier()
    data = pd.DataFrame({
        'question': [
            'What is the largest planet in our solar system?',
            'Which chemical element has the symbol O?',
            'What is the fastest land animal?',
            'What is the smallest prime number?',
            'Which continent is known as the \"Dark Continent\"?'
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
    mp.load_dataframe(data)
    template = {
        INSTRUCTION: 'The following are multiple choice questions (with answers) about {subject}.',
        PROMPT_FORMAT: 'Question: {question}\nOptions: {options}\nAnswer:',
        QUESTION_KEY: [REWORDING],
        OPTIONS_KEY: ['shuffle'],
        GOLD_KEY: {
            'field': 'answer',
            'type': 'index',
            'options_field': 'options'
        },
        FEW_SHOT_KEY: {
            'count': 2,
            'format': 'rotating',
            'split': 'all'
        }
    }
    mp.set_template(template)
    mp.configure(max_rows=5, variations_per_field=2)
    variations = mp.generate(verbose=True)
    for i, var in enumerate(variations):
        print(f"\nVariation {i + 1}:")
        print("-" * 50)
        print(var['prompt'])
        print("--- Conversation:")
        for msg in var['conversation']:
            print(f"[{msg['role']}] {msg['content']}")
        print("-" * 50)


def example_system_prompt_with_context_and_few_shot():
    """Example demonstrating context variations with both few-shot and zero-shot examples."""
    print("\n=== System Prompt with Context Variations + Few-shot/Zero-shot Examples ===")

    # Check if API key is available
    import os
    api_key = os.getenv("TOGETHER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  Warning: No API key found!")
        print("   Context variations require an API key to work properly.")
        print("   Set your API key with:")
        print("   export TOGETHER_API_KEY='your_key'")
        print("   or")
        print("   export OPENAI_API_KEY='your_key'")
        print("   The example will still run but context variations may not work as expected.\n")

    # Initialize the API
    mp = MultiPromptifier()

    # Create sample data with questions about different subjects
    data = pd.DataFrame({
        'question': [
            'What is the capital of France?',
            'Which planet is closest to the Sun?',
            'What is the chemical symbol for gold?',
            'How many sides does a triangle have?',
            'Who wrote Romeo and Juliet?',
            'What is the largest ocean on Earth?',
            'Which element has the atomic number 1?',
            'What is the square root of 16?'
        ],
        'options': [
            'London, Berlin, Paris, Madrid',
            'Venus, Mercury, Earth, Mars',
            'Au, Ag, Fe, Cu',
            '2, 3, 4, 5',
            'Shakespeare, Dickens, Austen, Twain',
            'Atlantic, Pacific, Indian, Arctic',
            'Helium, Hydrogen, Oxygen, Carbon',
            '2, 4, 8, 16'
        ],
        'answer': [2, 1, 0, 1, 0, 1, 1, 1],  # 0-based indices
        'subject': ['Geography', 'Astronomy', 'Chemistry', 'Mathematics', 'Literature', 'Geography', 'Chemistry',
                    'Mathematics']
    })

    mp.load_dataframe(data)
    print(f"📝 Loaded {len(data)} questions across different subjects")

    # Test 1: Zero-shot with context variations
    print("\n1️⃣ Zero-shot with Context Variations:")
    print("-" * 50)

    template_zero_shot = {
        INSTRUCTION: 'You are a knowledgeable assistant. Answer the following multiple choice questions.',
        PROMPT_FORMAT: 'Question: {question}\nOptions: {options}\nAnswer:',
        QUESTION_KEY: [CONTEXT_VARIATION],  # Use context variations
        OPTIONS_KEY: ['shuffle'],
        GOLD_KEY: {
            'field': 'answer',
            'type': 'index',
            'options_field': 'options'
        }
    }

    mp.set_template(template_zero_shot)
    mp.configure(max_rows=1, variations_per_field=2, max_variations=6)

    variations_zero_shot = mp.generate(verbose=True)

    print(f"\n✅ Generated {len(variations_zero_shot)} zero-shot variations with context")

    # Show variations with context (longer prompts)
    context_variations = [v for v in variations_zero_shot if len(v.get('prompt', '')) > 400]
    no_context_variations = [v for v in variations_zero_shot if len(v.get('prompt', '')) <= 400]

    print(f"   - {len(context_variations)} variations WITH context")
    print(f"   - {len(no_context_variations)} variations WITHOUT context")

    # Show first variation without context
    if no_context_variations:
        print(f"\nZero-shot Variation (No Context):")
        print("-" * 40)
        print(no_context_variations[0]['prompt'])
        print("-" * 40)

    # Show first variation with context
    if context_variations:
        print(f"\nZero-shot Variation (With Context):")
        print("-" * 40)
        context_prompt = context_variations[0]['prompt']
        if len(context_prompt) > 800:
            print(context_prompt[:800] + "...")
        else:
            print(context_prompt)
        print("-" * 40)

    # Export zero-shot results
    print("\n4️⃣ Exporting zero-shot results...")
    mp.export("context_variations_zero_shot.json", format="json")
    print("   - context_variations_zero_shot.json")

    # Test 2: Few-shot with context variations
    print("\n2️⃣ Few-shot with Context Variations:")
    print("-" * 50)

    template_few_shot = {
        INSTRUCTION: 'You are a knowledgeable assistant. Answer the following multiple choice questions.',
        PROMPT_FORMAT: 'Question: {question}\nOptions: {options}\nAnswer:',
        QUESTION_KEY: [CONTEXT_VARIATION],  # Use context variations
        OPTIONS_KEY: ['shuffle'],
        GOLD_KEY: {
            'field': 'answer',
            'type': 'index',
            'options_field': 'options'
        },
        FEW_SHOT_KEY: {
            'count': 2,
            'format': 'rotating',
            'split': 'all'
        }
    }

    mp.set_template(template_few_shot)
    mp.configure(max_rows=3, variations_per_field=2, max_variations=6)

    variations_few_shot = mp.generate(verbose=True)

    print(f"\n✅ Generated {len(variations_few_shot)} few-shot variations with context")

    # Show variations with context (longer prompts)
    context_variations_fs = [v for v in variations_few_shot if len(v.get('prompt', '')) > 400]
    no_context_variations_fs = [v for v in variations_few_shot if len(v.get('prompt', '')) <= 400]

    print(f"   - {len(context_variations_fs)} variations WITH context")
    print(f"   - {len(no_context_variations_fs)} variations WITHOUT context")

    # Show first variation without context
    if no_context_variations_fs:
        print(f"\nFew-shot Variation (No Context):")
        print("-" * 40)
        print(no_context_variations_fs[0]['prompt'])
        print("\n--- Conversation (Few-shot examples):")
        for msg in no_context_variations_fs[0]['conversation']:
            print(f"[{msg['role']}] {msg['content']}")
        print("-" * 40)

    # Show first variation with context
    if context_variations_fs:
        print(f"\nFew-shot Variation (With Context):")
        print("-" * 40)
        context_prompt_fs = context_variations_fs[0]['prompt']
        if len(context_prompt_fs) > 800:
            print(context_prompt_fs[:800] + "...")
        else:
            print(context_prompt_fs)
        print("\n--- Conversation (Few-shot examples):")
        for msg in context_variations_fs[0]['conversation']:
            print(f"[{msg['role']}] {msg['content']}")
        print("-" * 40)

    # Test 3: Compare context variations with and without few-shot
    print("\n3️⃣ Comparison: Context Variations Impact:")
    print("-" * 50)

    # Export results
    print("\n5️⃣ Exporting results...")
    mp.export("context_variations_few_shot.json", format="json")

    print("✅ Exported to:")
    print("   - context_variations_few_shot.json")

    if not api_key:
        print("\n💡 To see context variations in action:")
        print("   1. Set your API key: export TOGETHER_API_KEY='your_key'")
        print("   2. Run this example again")
        print("   3. You'll see questions with added background context")

    print("\n✅ Context variations with few-shot/zero-shot example completed!")


def example_simple_context_variations():
    """Simple example showing context variations concept without requiring API key."""
    print("\n=== Simple Context Variations Example (No API Key Required) ===")

    # Initialize the API
    mp = MultiPromptifier()

    # Simple data
    data = pd.DataFrame({
        'question': [
            'What is 2+2?',
            'What color is the sky?',
            'How many days are in a week?'
        ],
        'answer': ['4', 'Blue', '7']
    })

    mp.load_dataframe(data)
    print(f"📝 Loaded {len(data)} simple questions")

    # Template with rewordings (works without API key)
    template = {
        INSTRUCTION: 'You are a helpful assistant. Answer the following questions.',
        PROMPT_FORMAT: 'Question: {question}\nAnswer: {answer}',
        QUESTION_KEY: [REWORDING],  # This works without API key
        GOLD_KEY: 'answer'
    }

    mp.set_template(template)
    mp.configure(max_rows=3, variations_per_field=2, max_variations=6)

    variations = mp.generate(verbose=True)

    print(f"\n✅ Generated {len(variations)} variations with rewordings")
    for i, var in enumerate(variations[:3]):
        print(f"\nVariation {i + 1}:")
        print("-" * 40)
        print(var['prompt'])
        print("-" * 40)

    print("\n💡 This example shows how rewordings work without API key.")
    print("   Context variations would add background information but require API access.")

    # Export results
    mp.export("simple_context_example.json", format="json")
    print("✅ Exported to simple_context_example.json")


def example_enumerate_as_field_variation():
    """Example demonstrating enumerate as a field variation to get multiple enumeration types."""
    print("\n=== Enumerate as Field Variation Example ===")

    # Initialize the API
    mp = MultiPromptifier()

    # Create sample data
    data = pd.DataFrame({
        'question': [
            'What is the capital of France?',
        ],
        'options': [
            'London, Berlin, Paris, Madrid',
        ],
        'answer': [2]  # 0-based indices
    })

    mp.load_dataframe(data)
    print(f"📝 Loaded {len(data)} questions")

    # Configure template with enumerate as field variation
    print("\n2. Setting template with enumerate as field variation...")
    template = {
        INSTRUCTION: 'The following are multiple choice questions (with answers) about general knowledge.',
        PROMPT_FORMAT: 'Question: {question}\nOptions: {options}\nAnswer: {answer}',
        # QUESTION_KEY: [REWORDING],
        OPTIONS_KEY: [SHUFFLE_VARIATION, ENUMERATE_VARIATION],  # Use enumerate as field variation
        GOLD_KEY: {
            'field': 'answer',
            'type': 'index',
            'options_field': 'options'
        }
    }

    mp.set_template(template)
    print("✅ Template configured with enumerate as field variation")
    print("   - Will generate multiple enumeration types for options field")

    # Configure generation parameters
    print("\n3. Configuring generation...")
    mp.configure(
        max_rows=1,
        variations_per_field=2,  # Generate 4 variations with different enumeration types
        max_variations=8,
        random_seed=42
    )

    # Generate variations
    print("\n4. Generating variations...")
    variations = mp.generate(verbose=True)

    # Show results
    print(f"\n5. Results: Generated {len(variations)} variations")

    # Display variations to see different enumeration types
    for i, variation in enumerate(variations):
        print(f"\nVariation {i + 1}:")
        print("-" * 50)
        print(variation.get('prompt', 'No prompt found'))
        print("-" * 50)

    # Export results
    print("\n6. Exporting results...")
    mp.export("enumerate_field_variation.json", format="json")

    print("\n✅ Enumerate as field variation example completed!")


def example_many_augmenters_on_small_dataset():
    """Example: Apply context, shuffle, rewording, and paraphrase on a tiny dataset (2 rows)."""
    print("\n=== Many Augmenters on Small Dataset Example ===")
    import os
    from multipromptify import MultiPromptifier
    import pandas as pd
    from multipromptify.core.template_keys import (
        INSTRUCTION, PROMPT_FORMAT, QUESTION_KEY, OPTIONS_KEY, GOLD_KEY,
        REWORDING, PARAPHRASE_WITH_LLM, CONTEXT_VARIATION, SHUFFLE_VARIATION
    )

    # Check API key for context/paraphrase
    api_key = os.getenv("TOGETHER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  Warning: No API key found! Some augmenters may not work.")

    # Tiny dataset
    data = pd.DataFrame({
        'question': [
            'What is the capital of France?',
            'What is 2+2?'
        ],
        'options': [
            'London, Berlin, Paris, Madrid',
            '3, 4, 5, 6'
        ],
        'answer': [2, 1]  # 0-based indices
    })

    mp = MultiPromptifier()
    mp.load_dataframe(data)
    print(f"📝 Loaded {len(data)} questions")

    # Template: apply all augmenters
    template = {
        INSTRUCTION: 'Answer the following multiple choice questions.',
        INSTRUCTION_VARIATIONS: [PARAPHRASE_WITH_LLM],  # Reword the instruction
        PROMPT_FORMAT: 'Question: {question}\nOptions: {options}\nAnswer: {answer}',
        QUESTION_KEY: [REWORDING, CONTEXT_VARIATION],
        OPTIONS_KEY: [SHUFFLE_VARIATION, ENUMERATE_VARIATION],
        GOLD_KEY: {
            'field': 'answer',
            'type': 'index',
            'options_field': 'options'
        }
    }
    mp.set_template(template)
    print("✅ Template with context, shuffle, rewording, paraphrase")

    # Configure: all variations, but limit for demo
    mp.configure(
        max_rows=2,
        variations_per_field=2,  # 2 per augmenter per field
        max_variations=16,
        random_seed=42
    )
    mp.export("many_augmenters_small_dataset.json", format="json")
    print("\nGenerating variations...")
    variations = mp.generate(verbose=True)
    print(f"\n✅ Generated {len(variations)} variations\n")
    for i, v in enumerate(variations):
        print(f"\nVariation {i + 1}:")
        print("-" * 50)
        print(v.get('prompt', 'No prompt'))
        print("-" * 50)
    print("\nDone.")


def example_paraphrase_instruction_only():
    """Test: Single multiple choice question, only INSTRUCTION uses PARAPHRASE_WITH_LLM, with {subject} placeholder."""
    print("\n=== Paraphrase Instruction Only Example ===")
    # Single example
    data = pd.DataFrame({
        'question': [
            'What is the capital of France?'
        ],
        'options': [
            'London, Berlin, Paris, Madrid'
        ],
        'answer': [2],  # 0-based index
        'subject': ['Geography']
    })

    mp = MultiPromptifier()
    mp.load_dataframe(data)
    print(f"📝 Loaded {len(data)} question")

    template = {
        INSTRUCTION: 'The following are multiple choice questions (with answers) about {subject}.',
        INSTRUCTION_VARIATIONS: [PARAPHRASE_WITH_LLM],
        PROMPT_FORMAT: 'Question: {question}\nOptions: {options}\nAnswer:',
        GOLD_KEY: {
            'field': 'answer',
            'type': 'index',
            'options_field': 'options'
        }
    }
    mp.set_template(template)
    print("✅ Template with only instruction paraphrasing")

    mp.configure(max_rows=1, variations_per_field=10, max_variations=20)
    variations = mp.generate(verbose=True)
    print(f"\n✅ Generated {len(variations)} variations\n")
    for i, v in enumerate(variations):
        print(f"\nVariation {i + 1}:")
        print("-" * 50)
        print(v.get('prompt', 'No prompt'))
        print("-" * 50)
    print("\nDone.")


if __name__ == "__main__":
    # Run the examples
    # example_with_sample_data()
    # example_with_enumerate()
    # example_enumerate_types()
    # example_enumerate_as_field_variation()  # New example with enumerate as field variation

    # Uncomment other examples as needed:
    # example_with_sample_data_few_shot()
    # example_with_system_prompt_few_shot()
    # example_platform_switching()
    # example_with_huggingface()
    # example_different_templates()
    # example_gold_field_formats()
    # example_environment_variables()
    # example_with_simple_qa()
    # example_system_prompt_with_placeholder()
    # example_system_prompt_with_placeholder_and_few_shot()

    # Run context examples
    # example_simple_context_variations()  # Works without API key
    # example_system_prompt_with_context_and_few_shot()  # Full context example

    # example_many_augmenters_on_small_dataset()
    example_paraphrase_instruction_only()
    print("\n🎉 All examples completed!")
    print("\nNext steps:")
    print("1. Install datasets library: pip install datasets")
    print("2. Set your API keys:")
    print("   export TOGETHER_API_KEY='your_together_key'")
    print("   export OPENAI_API_KEY='your_openai_key'")
    print("3. Try the new enumerate feature in your templates:")
    print("   'enumerate': {'field': 'options', 'type': '1234'}")
    print("4. Try with your own data and templates")
