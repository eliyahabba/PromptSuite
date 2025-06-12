#!/usr/bin/env python3
"""
Comprehensive usage examples for MultiPromptify.
"""

import sys
import os
import pandas as pd

# Add the src directory to the path so we can import multipromptify
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from multipromptify import MultiPromptify


def example_1_basic_usage():
    """Example 1: Basic sentiment analysis with variations."""
    print("=== Example 1: Basic Sentiment Analysis ===")
    
    # Sample data
    data = pd.DataFrame({
        'text': ['I love this movie!', 'This book is terrible.', 'The weather is nice today.'],
        'label': ['positive', 'negative', 'neutral']
    })
    
    # Template with semantic and paraphrase variations
    template = "{instruction:semantic}: '{text:paraphrase}'\nSentiment: {label}"
    
    mp = MultiPromptify(max_variations=15)
    variations = mp.generate_variations(
        template=template,
        data=data,
        instruction="Classify the sentiment of the following text"
    )
    
    print(f"Generated {len(variations)} variations")
    print("\nSample variations:")
    for i, var in enumerate(variations[:3], 1):
        print(f"\n--- Variation {i} ---")
        print(var['prompt'])


def example_2_question_answering():
    """Example 2: Question answering with few-shot examples."""
    print("\n=== Example 2: Question Answering with Few-shot ===")
    
    # Sample QA data
    data = pd.DataFrame({
        'question': ['What is the capital of Germany?', 'How many continents are there?'],
        'answer': ['Berlin', 'Seven'],
        'context': ['Geography question', 'General knowledge']
    })
    
    # Template with few-shot examples
    template = "{instruction:paraphrase}: {few_shot}\n\nContext: {context:non-semantic}\nQuestion: {question:semantic}\nAnswer: {answer}"
    
    # Few-shot examples
    few_shot_examples = [
        "Q: What is the capital of France? A: Paris",
        "Q: What is 2+2? A: 4"
    ]
    
    mp = MultiPromptify(max_variations=10)
    variations = mp.generate_variations(
        template=template,
        data=data,
        instruction="Answer the following question based on the context",
        few_shot=few_shot_examples
    )
    
    print(f"Generated {len(variations)} variations")
    print("\nSample variation:")
    print(variations[0]['prompt'])


def example_3_multiple_choice():
    """Example 3: Multiple choice questions with different variation types."""
    print("\n=== Example 3: Multiple Choice Questions ===")
    
    # Sample multiple choice data
    data = pd.DataFrame({
        'question': ['What is the largest planet?', 'Which element has symbol O?'],
        'options': ['A) Earth B) Jupiter C) Mars', 'A) Oxygen B) Gold C) Silver'],
        'answer': ['B', 'A'],
        'subject': ['Astronomy', 'Chemistry']
    })
    
    # Template with different variation types
    template = "{instruction:semantic}:\n\nSubject: {subject:lexical}\nQuestion: {question:paraphrase}\nOptions: {options:non-semantic}\n\nAnswer: {answer}"
    
    mp = MultiPromptify(max_variations=8)
    variations = mp.generate_variations(
        template=template,
        data=data,
        instruction="Choose the correct answer from the options below"
    )
    
    print(f"Generated {len(variations)} variations")
    print("\nSample variation:")
    print(variations[0]['prompt'])


def example_4_file_operations():
    """Example 4: Working with files and different output formats."""
    print("\n=== Example 4: File Operations ===")
    
    # Use the sample CSV file
    csv_file = os.path.join(os.path.dirname(__file__), 'sample_data.csv')
    
    template = "{instruction:semantic}: {context:paraphrase}\nQ: {question:paraphrase}\nA: {answer}"
    
    mp = MultiPromptify(max_variations=6)
    
    # Load from CSV file
    variations = mp.generate_variations(
        template=template,
        data=csv_file,
        instruction="Please provide the answer to this question"
    )
    
    print(f"Loaded data from CSV and generated {len(variations)} variations")
    
    # Save in different formats
    mp.save_variations(variations, 'output_variations.json', format='json')
    mp.save_variations(variations, 'output_variations.csv', format='csv')
    
    print("Saved variations in JSON and CSV formats")
    
    # Show statistics
    stats = mp.get_stats(variations)
    print("\nStatistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Clean up
    for file in ['output_variations.json', 'output_variations.csv']:
        if os.path.exists(file):
            os.remove(file)


def example_5_advanced_features():
    """Example 5: Advanced features - different input types."""
    print("\n=== Example 5: Advanced Features ===")
    
    # Dictionary input
    data_dict = {
        'task': ['Translate to French', 'Translate to Spanish'],
        'text': ['Hello world', 'Good morning'],
        'target_lang': ['French', 'Spanish']
    }
    
    template = "{instruction:semantic}: {task:paraphrase}\nText: '{text:surface}'\nTarget Language: {target_lang}"
    
    mp = MultiPromptify(max_variations=8)
    variations = mp.generate_variations(
        template=template,
        data=data_dict,
        instruction="Complete the following translation task"
    )
    
    print(f"Generated {len(variations)} variations from dictionary input")
    print("\nSample variation:")
    print(variations[0]['prompt'])


def main():
    """Run all examples."""
    print("MultiPromptify Usage Examples")
    print("=" * 60)
    
    examples = [
        example_1_basic_usage,
        example_2_question_answering,
        example_3_multiple_choice,
        example_4_file_operations,
        example_5_advanced_features
    ]
    
    for example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"Error in {example_func.__name__}: {e}")
    
    print("\n" + "=" * 60)
    print("All examples completed!")


if __name__ == '__main__':
    main() 