#!/usr/bin/env python3
"""
Test script for MultiPromptify to demonstrate functionality.
"""

import sys
import os
import pandas as pd

# Add the src directory to the path so we can import multipromptify
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from multipromptify import MultiPromptify
from src.multipromptify.template_keys import (
    INSTRUCTION_TEMPLATE_KEY, INSTRUCTION_KEY, QUESTION_KEY, GOLD_KEY, FEW_SHOT_KEY, OPTIONS_KEY, CONTEXT_KEY, PROBLEM_KEY,
    PARAPHRASE_WITH_LLM, REWORDING, SHUFFLE_VARIATION, GOLD_FIELD, INSTRUCTION_TEMPLATE_FIELD
)


def test_basic_functionality():
    """Test basic MultiPromptify functionality."""
    print("=== Testing Basic Functionality ===")
    
    try:
        # Create sample data
        data = pd.DataFrame({
            'question': ['What is 2+2?', 'What color is the sky?'],
            'answer': ['4', 'Blue'],
            'context': ['Math problem', 'General knowledge']
        })
        
        # Template with variations
        template = "{instruction:semantic}: {context:paraphrase_with_llm}\nQuestion: {question:paraphrase_with_llm}\nAnswer: {answer}"
        
        # Initialize MultiPromptify
        mp = MultiPromptify(max_variations_per_row=20)
        
        # Generate variations
        variations = mp.generate_variations(
            template=template,
            data=data,
            instruction="Please answer the following question"
        )
        
        print(f"Generated {len(variations)} variations")
        
        # Show first few variations
        for i, var in enumerate(variations[:3], 1):
            print(f"\n--- Variation {i} ---")
            print(var['prompt'])
        
        # Show statistics
        stats = mp.get_stats(variations)
        print(f"\n=== Statistics ===")
        for key, value in stats.items():
            print(f"{key}: {value}")
            
        print("✓ Basic functionality test passed")
        return True
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_template_parsing():
    """Test template parsing functionality."""
    print("\n=== Testing Template Parsing ===")
    
    try:
        mp = MultiPromptify()
        
        # Test valid templates
        valid_templates = [
            "{instruction:semantic}: {question:paraphrase_with_llm}",
            "{instruction}: {context:non-semantic} {question}",
            "{few_shot}\n{question:lexical} -> {answer:rewording}"
        ]
        
        for template in valid_templates:
            is_valid, errors = mp.template_parser.validate_template(template)
            fields = mp.parse_template(template)
            print(f"Template: {template}")
            print(f"Valid: {is_valid}")
            print(f"Fields: {fields}")
            print()
        
        # Test invalid template
        invalid_template = "{instruction:invalid_type}: {question"
        is_valid, errors = mp.template_parser.validate_template(invalid_template)
        print(f"Invalid template: {invalid_template}")
        print(f"Valid: {is_valid}")
        print(f"Errors: {errors}")
        
        print("✓ Template parsing test passed")
        return True
        
    except Exception as e:
        print(f"✗ Template parsing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_few_shot_examples():
    """Test few-shot examples functionality."""
    print("\n=== Testing Few-shot Examples ===")
    
    try:
        data = pd.DataFrame({
            'question': ['What is 10-5?'],
            'answer': ['5']
        })
        
        template = "{instruction:paraphrase_with_llm}: {few_shot}\n\nQuestion: {question}\nAnswer: {answer}"
        
        mp = MultiPromptify(max_variations_per_row=10)
        
        # Test with few-shot examples
        few_shot = ["Q: What is 1+1? A: 2", "Q: What is 3*3? A: 9"]
        
        variations = mp.generate_variations(
            template=template,
            data=data,
            instruction="Solve the math problem",
            few_shot=few_shot
        )
        
        print(f"Generated {len(variations)} variations with few-shot examples")
        print(f"\nSample variation:")
        print(variations[0]['prompt'])
        
        print("✓ Few-shot examples test passed")
        return True
        
    except Exception as e:
        print(f"✗ Few-shot examples test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_file_io():
    """Test file input/output functionality."""
    print("\n=== Testing File I/O ===")
    
    try:
        # Create test CSV file
        test_data = pd.DataFrame({
            'text': ['I love this!', 'This is terrible.'],
            'sentiment': ['positive', 'negative']
        })
        test_file = 'test_data.csv'
        test_data.to_csv(test_file, index=False)
        
        mp = MultiPromptify(max_variations_per_row=5)
        template = "{instruction:semantic}: '{text:paraphrase_with_llm}'\nSentiment: {sentiment}"
        
        # Load from file
        variations = mp.generate_variations(
            template=template,
            data=test_file,
            instruction="Classify the sentiment"
        )
        
        print(f"Loaded data from CSV and generated {len(variations)} variations")
        
        # Save variations
        output_file = 'test_variations.json'
        mp.save_variations(variations, output_file, format='json')
        print(f"Saved variations to {output_file}")
        
        # Clean up
        if os.path.exists(test_file):
            os.remove(test_file)
        if os.path.exists(output_file):
            os.remove(output_file)
        
        print("✓ File I/O test passed")
        return True
        
    except Exception as e:
        print(f"✗ File I/O test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("MultiPromptify Test Suite")
    print("=" * 50)
    
    # Run all tests
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Template Parsing", test_template_parsing),
        ("Few-shot Examples", test_few_shot_examples),
        ("File I/O", test_file_io)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\nRunning {test_name} test...")
        try:
            result = test_func()
            if result:
                passed += 1
        except Exception as e:
            print(f"✗ Test {test_name} failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed!")
        return True
    else:
        print("✗ Some tests failed!")
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1) 