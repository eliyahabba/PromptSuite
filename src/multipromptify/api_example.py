#!/usr/bin/env python3
"""
MultiPromptify API Example Script

This script demonstrates how to use the MultiPromptifyAPI class for programmatic
generation of prompt variations.
"""

import pandas as pd
from multipromptify import MultiPromptifyAPI

def example_with_sample_data2():
    # Create instance
    mp = MultiPromptifyAPI()

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
        'instruction_template': 'Answer the math question:\nQuestion: {question}\nAnswer: {answer}',
        'question': ['surface'],  # surface variations
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
def example_with_sample_data_with_paraphrase():
    """Example using sample data with different template configurations."""

    print("🚀 MultiPromptify API Example")
    print("=" * 50)

    # Initialize the API
    mp = MultiPromptifyAPI()

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
        'instruction_template': 'Answer the following multiple choice question:\nQuestion: {question}\nOptions: {options}\nAnswer: {answer}',
        'instruction': ['paraphrase'],
        'question': ['surface'],
        'options': ['shuffle', 'surface'],
        'gold': {
            'field': 'answer',
            'type': 'index',  # This means answer field contains indices, not text
            'options_field': 'options'
        },
        'few_shot': {
            'count': 2,
            'format': 'fixed',
            'split': 'all'
        }
    }

    mp.set_template(template)

    # Configure generation parameters
    print("\n3. Configuring generation...")
    mp.configure(
        max_rows=3,  # Use first 3 rows (need at least 3 for few_shot count=2)
        variations_per_field=3,  # 3 variations per field
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


def example_with_sample_data():
    """Example using sample data with different template configurations."""
    
    print("🚀 MultiPromptify API Example")
    print("=" * 50)
    
    # Initialize the API
    mp = MultiPromptifyAPI()
    
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
        'instruction_template': 'Answer the following multiple choice question:\nQuestion: {question}\nOptions: {options}\nAnswer: {answer}',
        # 'instruction': ['paraphrase'],
        'options': ['shuffle', 'surface'],
        'gold': {
            'field': 'answer',
            'type': 'index',  # This means answer field contains indices, not text
            'options_field': 'options'
        },
        'few_shot': {
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
    mp = MultiPromptifyAPI()
    
    # Create simple data
    data = [{"question": "What is AI?", "answer": "Artificial Intelligence"}]
    mp.load_dataframe(pd.DataFrame(data))
    
    # Simple template with paraphrase (requires API key)
    template = {
        'instruction_template': 'Question: {question}\nAnswer: {answer}',
        'instruction': ['paraphrase'],
        'gold': 'answer'  # Simple format - just the field name
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
    """Example using HuggingFace datasets (requires datasets library)."""
    
    print("\n" + "=" * 50)
    print("🤗 HuggingFace Dataset Example")
    print("=" * 50)
    
    try:
        # Initialize API
        mp = MultiPromptifyAPI()
        
        # Load from HuggingFace (this will fail if datasets library is not installed)
        print("\n1. Loading HuggingFace dataset...")
        # Uncomment the line below to try loading from HuggingFace
        mp.load_dataset("squad", split="train")
        
        # print("⚠️ Skipping HuggingFace example - uncomment the load_dataset line to try it")
        # print("   (Requires: pip install datasets)")
        
    except Exception as e:
        print(f"❌ HuggingFace example failed: {e}")


def example_different_templates():
    """Examples showing different template configurations."""
    
    print("\n" + "=" * 50)
    print("📝 Different Template Examples")
    print("=" * 50)
    
    # Simple QA template (text-based answers)
    simple_template = {
        'instruction_template': 'Question: {question}\nAnswer: {answer}',
        'question': ['surface'],
        'gold': 'answer'  # Simple format for text answers
    }
    
    # Multiple choice template (index-based answers)
    multiple_choice_template = {
        'instruction_template': 'Choose the correct answer:\nQ: {question}\nOptions: {options}\nA: {answer}',
        'question': ['surface'],
        'options': ['shuffle', 'surface'],
        'gold': {
            'field': 'answer',
            'type': 'index',  # Answer is index in options
            'options_field': 'options'
        }
    }
    
    # Complex template with multiple variations
    complex_template = {
        'instruction_template': 'Context: {context}\nQuestion: {question}\nAnswer: {answer}',
        'instruction': ['paraphrase'],
        'context': ['surface', 'paraphrase'],
        'question': ['surface'],
        'gold': {
            'field': 'answer',
            'type': 'value'  # Answer is text value
        },
        'few_shot': {
            'count': 1,
            'format': 'rotating',
            'split': 'all'
        }
    }
    
    # Platform-specific template with different configurations
    platform_templates = {
        'TogetherAI': {
            'instruction_template': 'Using Llama model: {question}\nAnswer: {answer}',
            'instruction': ['paraphrase'],
            'question': ['surface'],
            'gold': 'answer'
        },
        'OpenAI': {
            'instruction_template': 'Using GPT model: {question}\nAnswer: {answer}',
            'instruction': ['paraphrase'],
            'question': ['surface'],
            'gold': 'answer'
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
        'instruction_template': 'Q: {question}\nOptions: {options}\nA: {answer}',
        'gold': {
            'field': 'answer',
            'type': 'index',
            'options_field': 'options'
        }
    }
    print("   Template gold config:", index_template['gold'])
    
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
        'instruction_template': 'Q: {question}\nOptions: {options}\nA: {answer}',
        'gold': {
            'field': 'answer',
            'type': 'value',
            'options_field': 'options'
        }
    }
    print("   Template gold config:", value_template['gold'])
    
    print("\n3. Simple text answer data:")
    text_data = [
        {
            "question": "What is the capital of France?",
            "answer": "Paris"
        }
    ]
    print("   Data:", text_data[0])
    
    text_template = {
        'instruction_template': 'Q: {question}\nA: {answer}',
        'gold': 'answer'  # Simple format
    }
    print("   Template gold config:", text_template['gold'])


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
    mp = MultiPromptifyAPI()
    
    print(f"\nDefault platform API key detection:")
    print(f"   Platform: {mp.config['api_platform']}")
    print(f"   API Key: {'✅ Found' if mp.config['api_key'] else '❌ Not found'}")
    # Test platform switching
    print(f"\nTesting platform switching:")
    for platform in ["TogetherAI", "OpenAI"]:
        mp.configure(api_platform=platform)
        key_found = mp.config['api_key'] is not None
        print(f"   {platform}: {'✅ API key found' if key_found else '❌ No API key'}")


if __name__ == "__main__":
    # Run the examples
    example_with_sample_data()
    # example_with_sample_data_with_paraphrase()
    # example_with_sample_data2()
    # example_platform_switching()
    # example_with_huggingface()
    # example_different_templates()
    # example_gold_field_formats()
    # example_environment_variables()
    #
    print("\n🎉 All examples completed!")
    print("\nNext steps:")
    print("1. Install datasets library: pip install datasets")
    print("2. Set your API keys:")
    print("   export TOGETHER_API_KEY='your_together_key'")
    print("   export OPENAI_API_KEY='your_openai_key'")
    print("3. Try with your own data and templates") 