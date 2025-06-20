#!/usr/bin/env python3
"""
MultiPromptify API Example Script

This script demonstrates how to use the MultiPromptifier class for programmatic
generation of prompt variations.
"""

import pandas as pd
from multipromptify import MultiPromptifier
from multipromptify.core.template_keys import (
    INSTRUCTION_TEMPLATE_KEY, INSTRUCTION_KEY, QUESTION_KEY, GOLD_KEY, FEW_SHOT_KEY, OPTIONS_KEY, CONTEXT_KEY, PROBLEM_KEY,
    PARAPHRASE_WITH_LLM, REWORDING, CONTEXT_VARIATION, SHUFFLE_VARIATION, MULTIDOC_VARIATION, ENUMERATE_VARIATION, GOLD_FIELD,
    INSTRUCTION_TEMPLATE_FIELD
)

def example_with_sample_data2():
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
        INSTRUCTION_TEMPLATE_KEY: 'Answer the math question:\nQuestion: {question}\nAnswer: {answer}',
        QUESTION_KEY: [REWORDING],  # surface variations
        GOLD_KEY: 'answer',
        FEW_SHOT_KEY: {
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
        INSTRUCTION_TEMPLATE_KEY: 'Answer the following multiple choice question:\nQuestion: {question}\nOptions: {options}\nAnswer: {answer}',
        QUESTION_KEY: [REWORDING],
        GOLD_KEY: {
            'field': 'answer',
            'type': 'index',
            'options_field': 'options'
        },
        ENUMERATE_VARIATION: {
            'field': 'options',  # Which field to enumerate
            'type': '1234'       # Use numbers: 1. 2. 3. 4.
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
    for i, variation in enumerate(variations[:2]):
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
            INSTRUCTION_TEMPLATE_KEY: 'Question: {question}\nOptions: {options}\nAnswer: {answer}',
            GOLD_KEY: {'field': 'answer', 'type': 'index', 'options_field': 'options'},
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
        INSTRUCTION_TEMPLATE_KEY: 'Answer the following multiple choice question:\nQuestion: {question}\nOptions: {options}\nAnswer: {answer}',
        OPTIONS_KEY: [REWORDING, SHUFFLE_VARIATION],
        GOLD_KEY: {
            'field': 'answer',
            'type': 'index',  # This means answer field contains indices, not text
            'options_field': 'options'
        },
        FEW_SHOT_KEY: {
            'count': 2,
            'format': 'fixed',
            'split': 'all'
        }
    }
    
    mp.set_template(template)
    
    # Configure generation parameters
    print("\n3. Configuring generation...")
    mp.configure(
        max_rows=3,                    # Use first 3 rows (need at least 3 for few_shot count=2)
        variations_per_field=3,        # 3 variations per field
        max_variations=20,             # Maximum 20 total variations
        random_seed=42,                # For reproducibility
        api_platform="TogetherAI",     # Platform selection
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
        print(f"\nVariation {i+1}:")
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
        INSTRUCTION_TEMPLATE_KEY: 'Question: {question}\nAnswer: {answer}',
        INSTRUCTION_KEY: [PARAPHRASE_WITH_LLM],
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
            "instruction_template": "Read the context and answer the question.\\nContext: {context}\\nQuestion: {question}\\nAnswer:",
            # "instruction": ["paraphrase_with_llm"],  # Paraphrase the instruction
            "context": ["rewording"],                # Reword the context
            "question": [],
            # Use a Python expression to extract the first answer text from the dict
            "gold": "answers['text'][0]"
        }
        mp.set_template(template)
        mp.configure(max_rows=3, variations_per_field=1, max_variations=1)

        print("\n2. Generating variations...")
        variations = mp.generate(verbose=True)

        print(f"\n✅ Generated {len(variations)} variations\n")
        for i, v in enumerate(variations):
            print(f"Prompt {i+1}:")
            print(v["prompt"])
            print("Expected answer:", v["answers['text'][0]"])
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
        INSTRUCTION_TEMPLATE_KEY: 'Question: {question}\nAnswer: {answer}',
        QUESTION_KEY: [REWORDING],
        GOLD_KEY: 'answer'  # Simple format for text answers
    }
    
    # Multiple choice template (index-based answers)
    multiple_choice_template = {
        INSTRUCTION_TEMPLATE_KEY: 'Choose the correct answer:\nQ: {question}\nOptions: {options}\nA: {answer}',
        QUESTION_KEY: [REWORDING, REWORDING],
        OPTIONS_KEY: [REWORDING, REWORDING],
        GOLD_KEY: {
            'field': 'answer',
            'type': 'index',  # Answer is index in options
            'options_field': 'options'
        }
    }
    
    # Complex template with multiple variations
    complex_template = {
        INSTRUCTION_TEMPLATE_KEY: 'Context: {context}\nQuestion: {question}\nAnswer: {answer}',
        INSTRUCTION_KEY: [PARAPHRASE_WITH_LLM],
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
            INSTRUCTION_TEMPLATE_KEY: 'Using Llama model: {question}\nAnswer: {answer}',
            INSTRUCTION_KEY: [PARAPHRASE_WITH_LLM],
            QUESTION_KEY: [REWORDING],
            GOLD_KEY: 'answer'
        },
        'OpenAI': {
            INSTRUCTION_TEMPLATE_KEY: 'Using GPT model: {question}\nAnswer: {answer}',
            INSTRUCTION_KEY: [PARAPHRASE_WITH_LLM],
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
        INSTRUCTION_TEMPLATE_KEY: 'Q: {question}\nOptions: {options}\nA: {answer}',
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
        INSTRUCTION_TEMPLATE_KEY: 'Q: {question}\nOptions: {options}\nA: {answer}',
        GOLD_KEY: {
            'field': 'answer',
            'type': 'value',
            'options_field': 'options'
        }
    }
    print("   Template gold config:", value_template[GOLD_KEY])
    
    print("\n3. Simple text answer data:")
    text_data = [
        {
            "question": "What is the capital of France?",
            "answer": "Paris"
        }
    ]
    print("   Data:", text_data[0])
    
    text_template = {
        INSTRUCTION_TEMPLATE_KEY: 'Q: {question}\nA: {answer}',
        GOLD_KEY: 'answer'  # Simple format
    }
    print("   Template gold config:", text_template[GOLD_KEY])


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
    csv_path = os.path.join(os.path.dirname(__file__), '../../data/simple_qa_test.csv')
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
        INSTRUCTION_TEMPLATE_KEY: 'Question: {problem}\nAnswer: {answer}',
        PROBLEM_KEY: [REWORDING],
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
        INSTRUCTION_TEMPLATE_KEY: 'Please answer the following question:\n{question}',
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


if __name__ == "__main__":
    # Run the examples
    example_with_sample_data()
    # example_with_enumerate()
    # example_enumerate_types()
    
    # Uncomment other examples as needed:
    # example_with_sample_data2()
    # example_platform_switching()
    example_with_huggingface()
    # example_different_templates()
    # example_gold_field_formats()
    # example_environment_variables()
    # example_with_simple_qa()
    print("\n🎉 All examples completed!")
    print("\nNext steps:")
    print("1. Install datasets library: pip install datasets")
    print("2. Set your API keys:")
    print("   export TOGETHER_API_KEY='your_together_key'")
    print("   export OPENAI_API_KEY='your_openai_key'")
    print("3. Try the new enumerate feature in your templates:")
    print("   'enumerate': {'field': 'options', 'type': '1234'}")
    print("4. Try with your own data and templates") 