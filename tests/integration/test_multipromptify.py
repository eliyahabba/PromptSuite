#!/usr/bin/env python3
"""
Test script for PromptSuiteEngine to demonstrate functionality.
"""

import sys
import os
import pandas as pd

# Add the src directory to the path so we can import promptsuite
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from promptsuite import PromptSuiteEngine
from promptsuite.core.template_keys import (
    INSTRUCTION, PROMPT_FORMAT, GOLD_KEY, FEW_SHOT_KEY,
    PARAPHRASE_WITH_LLM, SHUFFLE_VARIATION
)


def test_basic_functionality():
    """Test basic PromptSuiteEngine functionality."""
    print("=== Testing Basic Functionality ===")
    
    try:
        # Create sample data
        data = pd.DataFrame({
            'question': ['What is 2+2?', 'What color is the sky?'],
            'answer': ['4', 'Blue'],
            'context': ['Math problem', 'General knowledge']
        })
        
        # Template with variations (new format)
        template = {
            INSTRUCTION: "Please answer the following question",
            PROMPT_FORMAT: "Context: {context}\nQuestion: {question}\nAnswer: {answer}",
            'context': [PARAPHRASE_WITH_LLM],
            'question': [PARAPHRASE_WITH_LLM]
        }
        
        # Initialize PromptSuiteEngine
        sp = PromptSuiteEngine(max_variations_per_row=20)
        
        # Generate variations
        variations = sp.generate_variations(
            template=template,
            data=data
        )
        
        print(f"Generated {len(variations)} variations")
        
        # Show first few variations
        for i, var in enumerate(variations[:3], 1):
            print(f"\n--- Variation {i} ---")
            print(var['prompt'])
        
        # Show statistics
        stats = sp.get_stats(variations)
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
        sp = PromptSuiteEngine()
        
        # Test valid templates (new format)
        valid_templates = [
            {
                INSTRUCTION: "Answer the question",
                PROMPT_FORMAT: "Question: {question}\nAnswer: {answer}",
                'question': [PARAPHRASE_WITH_LLM]
            },
            {
                INSTRUCTION: "Process the input",
                PROMPT_FORMAT: "Context: {context}\nQuestion: {question}",
                'context': [PARAPHRASE_WITH_LLM]
            }
        ]
        
        for template in valid_templates:
            is_valid, errors = sp.template_parser.validate_template(template)
            fields = sp.parse_template(template)
            print(f"Template: {template}")
            print(f"Valid: {is_valid}")
            print(f"Fields: {fields}")
            print()
        
        # Test invalid template
        invalid_template = {
            INSTRUCTION: "Test",
            PROMPT_FORMAT: "Question: {question}",
            'question': ["invalid_variation_type"]
        }
        is_valid, errors = sp.template_parser.validate_template(invalid_template)
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
        
        template = {
            INSTRUCTION: "Solve the math problem",
            PROMPT_FORMAT: "Question: {question}\nAnswer: {answer}",
            FEW_SHOT_KEY: {
                'count': 2,
                'format': 'shared_ordered_first_n',
                'split': 'all'
            }
        }
        
        sp = PromptSuiteEngine(max_variations_per_row=10)
        
        # Add few-shot data to the dataframe
        few_shot_data = pd.DataFrame({
            'question': ['What is 10-5?', 'What is 1+1?', 'What is 3*3?'],
            'answer': ['5', '2', '9']
        })
        
        variations = sp.generate_variations(
            template=template,
            data=few_shot_data
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
        
        sp = PromptSuiteEngine(max_variations_per_row=5)
        template = {
            INSTRUCTION: "Classify the sentiment",
            PROMPT_FORMAT: "Text: '{text}'\nSentiment: {sentiment}",
            'text': [PARAPHRASE_WITH_LLM]
        }
        
        # Load from file
        variations = sp.generate_variations(
            template=template,
            data=test_file
        )
        
        print(f"Loaded data from CSV and generated {len(variations)} variations")
        
        # Save variations
        output_file = 'test_variations.json'
        sp.save_variations(variations, output_file, format='json')
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
    print("PromptSuiteEngine Test Suite")
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