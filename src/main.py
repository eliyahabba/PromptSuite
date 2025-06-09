import argparse
import sys
import os
from typing import List, Dict, Any

# Add the project root to the path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from integration.pipeline import AugmentationPipeline
from integration.combinatorial import VariationCombiner

def main():
    """Main entry point for the multi-prompt evaluation tool."""
    parser = argparse.ArgumentParser(description='Multi-Prompt Evaluation Tool')
    parser.add_argument('--input', '-i', type=str, help='Input prompt or file path')
    parser.add_argument('--output', '-o', type=str, help='Output file path')
    parser.add_argument('--max-combinations', type=int, default=100, 
                        help='Maximum number of combinations to generate')
    args = parser.parse_args()
    
    # Get the input prompt
    prompt = ""
    if args.input:
        if os.path.isfile(args.input):
            with open(args.input, 'r', encoding='utf-8') as f:
                prompt = f.read()
        else:
            prompt = args.input
    else:
        print("Please provide an input prompt or file.")
        return
    
    # Initialize the pipeline
    pipeline = AugmentationPipeline()
    pipeline.load_components()
    
    # Process the prompt
    variations_by_axis = pipeline.process(prompt)
    
    # Print the variations by axis
    print(f"Found {len(variations_by_axis)} axes that can be varied:")
    for axis_name, variations in variations_by_axis.items():
        print(f"  {axis_name}: {len(variations)} variations")
    
    # Generate combinations
    combiner = VariationCombiner(max_combinations=args.max_combinations)
    combined_variations = combiner.combine(variations_by_axis)
    
    print(f"Generated {len(combined_variations)} combined variations")
    
    # Output the results
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(f"Original prompt:\n{prompt}\n\n")
            f.write(f"Variations by axis:\n")
            for axis_name, variations in variations_by_axis.items():
                f.write(f"\n{axis_name}:\n")
                for i, var in enumerate(variations):
                    f.write(f"Variation {i+1}:\n{var}\n\n")
            
            f.write(f"\nCombined variations:\n")
            for i, var in enumerate(combined_variations):
                f.write(f"Combination {i+1}:\n{var}\n\n")
    else:
        # Print a sample of variations to the console
        print("\nSample variations:")
        for axis_name, variations in variations_by_axis.items():
            print(f"\n{axis_name}:")
            for i, var in enumerate(variations[:2]):  # Show only first 2 variations
                print(f"Variation {i+1}:\n{var[:100]}..." if len(var) > 100 else var)
                
        print("\nSample combined variations:")
        for i, var in enumerate(combined_variations[:3]):  # Show only first 3 combinations
            print(f"Combination {i+1}:\n{var[:100]}..." if len(var) > 100 else var)

if __name__ == "__main__":
    main() 