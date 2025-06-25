#!/usr/bin/env python3
"""
MMLU Task: Local CSV - Subject-wise processing
This module provides a class for generating prompt variations for MMLU tasks, 
processing each subject separately.
"""

import os
import sys
from pathlib import Path
import pandas as pd
import ast
from typing import Dict, Any, List

# Add the project root to the path to import multipromptify and multipromptify_tasks
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from multipromptify.core.template_keys import (
    INSTRUCTION, PROMPT_FORMAT, QUESTION_KEY, OPTIONS_KEY, GOLD_KEY,
    PARAPHRASE_WITH_LLM, FORMAT_STRUCTURE_VARIATION, TYPOS_AND_NOISE_VARIATION,
    SHUFFLE_VARIATION, ENUMERATE_VARIATION, INSTRUCTION_VARIATIONS, PROMPT_FORMAT_VARIATIONS,
    FEW_SHOT_KEY
)

from multipromptify_tasks.tasks.base_task import BaseTask


class MMLUTask(BaseTask):
    """Task for generating MMLU prompt variations by subject."""
    
    def __init__(self, subject: str = None):
        self.subject = subject
        self.original_subject = subject  # Keep original subject name for file naming
        if subject:
            # Convert subject name for display (replace _ with space)
            display_subject = subject.replace('_', ' ')
            task_name = f"MMLU Task: {display_subject}"
            output_filename = f"mmlu_{subject}_variations.json"  # Keep _ in filename
        else:
            task_name = "MMLU Task: All Subjects"
            output_filename = "mmlu_all_variations.json"
        
        super().__init__(
            task_name=task_name,
            output_filename=output_filename
        )
    
    def load_data(self) -> None:
        """Load MMLU data from local CSV file."""
        csv_path = Path(__file__).parent.parent / "raw_data/mmlu_sample.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"MMLU CSV file not found: {csv_path}")
        
        print(f"Loading MMLU data from {csv_path}")
        df = pd.read_csv(csv_path)
        df['choices'] = df['choices'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        
        # Create a display version of subject for the prompt template
        df['subject_display'] = df['subject'].str.replace('_', ' ', regex=False)
        
        # Filter by subject if specified (using original subject name with underscores)
        if self.subject:
            original_len = len(df)
            df = df[df['subject'] == self.subject]
            display_subject = self.subject.replace('_', ' ')
            print(f"✅ Filtered to subject '{display_subject}': {len(df)} rows (from {original_len} total)")
        else:
            print(f"✅ Processing all subjects: {len(df)} rows")
        
        if len(df) == 0:
            raise ValueError(f"No data found for subject: {self.subject}")
        
        self.mp.load_dataframe(df)
        subject_name = self.subject.replace('_', ' ') if self.subject else 'all subjects'
        print(f"✅ Loaded {len(df)} rows for MMLU {subject_name}")
    
    def get_template(self) -> Dict[str, Any]:
        """Get template configuration for MMLU task according to paper specifications."""
        return {
            INSTRUCTION: "The following are multiple choice questions (with answers) about {subject_display}.",
            INSTRUCTION_VARIATIONS: [PARAPHRASE_WITH_LLM],
            PROMPT_FORMAT: "Question: {question}\nChoices: {choices}\nAnswer:\n{answer}",
            PROMPT_FORMAT_VARIATIONS: [FORMAT_STRUCTURE_VARIATION],
            'choices': [SHUFFLE_VARIATION, ENUMERATE_VARIATION],
            GOLD_KEY: {
                'field': 'answer',
                'type': 'index',
                'options_field': 'choices'
            },
            FEW_SHOT_KEY: {
                'count': 5,  # Reduced from 5 to work with smaller datasets
                'format': 'random_per_row',
                'split': 'all'
            }
        }


def get_available_subjects() -> List[str]:
    """Get list of available subjects from the MMLU dataset."""
    csv_path = Path(__file__).parent.parent / "raw_data/mmlu_sample.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"MMLU CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    subjects = df['subject'].unique().tolist()
    return sorted(subjects)


def generate_all_subjects():
    """Generate variations for all subjects separately."""
    # Create output directory
    output_dir = Path(__file__).parent.parent / "data" / "mmlu"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    subjects = get_available_subjects()
    print(f"🎯 Found {len(subjects)} subjects:")
    for i, subject in enumerate(subjects, 1):
        display_name = subject.replace('_', ' ')
        print(f"  {i:2d}. {display_name} ({subject})")
    
    generated_files = []
    
    for subject in subjects:
        display_name = subject.replace('_', ' ')
        print(f"\n📚 Processing subject: {display_name}")
        print("=" * 50)
        
        try:
            task = MMLUTask(subject=subject)
            
            # Override output path to save in mmlu folder
            output_file = output_dir / f"mmlu_{subject}_variations.json"
            
            # Create a custom generate method that uses the correct path
            def custom_generate():
                """Custom generate method that uses the correct output path."""
                print(f"🚀 Starting {task.task_name}")
                print("=" * 60)

                # Load data
                print("\n1. Loading data...")
                task.load_data()

                # Configure template
                print("\n2. Setting up template...")
                template = task.get_template()
                task.mp.set_template(template)
                print("✅ Template configured")

                # Configure generation parameters
                print(f"\n3. Configuring generation...")
                task.mp.configure(
                    max_rows=10,  # Use a reasonable default
                    variations_per_field=4,
                    max_variations_per_row=10,
                    random_seed=42,
                    api_platform="TogetherAI",
                    model_name="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
                )

                # Generate variations
                print("\n4. Generating prompt variations...")
                variations = task.mp.generate(verbose=True)

                # Display results
                print(f"\n✅ Generated {len(variations)} variations")

                # Show a few examples
                print("\n5. Sample variations:")
                for i, var in enumerate(variations[:3]):
                    print(f"\nVariation {i + 1}:")
                    print("-" * 50)
                    prompt = var.get('prompt', 'No prompt found')
                    if len(prompt) > 500:
                        prompt = prompt[:500] + "..."
                    print(prompt)
                    print("-" * 50)

                # Export results using the correct path
                print(f"\n6. Exporting results to {output_file}...")
                task.mp.export(str(output_file), format="json")
                print("✅ Export completed!")

                # Show final statistics
                print("\n7. Final statistics:")
                task.mp.info()

                return str(output_file)
            
            # Use the custom generate method
            generated_file = custom_generate()
            generated_files.append(generated_file)
            print(f"✅ Completed {display_name}: {generated_file}")
            
        except Exception as e:
            print(f"❌ Error processing {display_name}: {e}")
            continue
    
    print(f"\n🎉 All subjects completed! Generated {len(generated_files)} files:")
    for file in generated_files:
        print(f"  📄 {file}")
    
    return generated_files


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate MMLU prompt variations")
    parser.add_argument("--subject", help="Generate variations for a specific subject only")
    parser.add_argument("--list-subjects", action="store_true", help="List available subjects")
    parser.add_argument("--all", action="store_true", help="Generate variations for all subjects", default=True)
    
    args = parser.parse_args()
    
    if args.list_subjects:
        subjects = get_available_subjects()
        print(f"Available subjects ({len(subjects)}):")
        for subject in subjects:
            display_name = subject.replace('_', ' ')
            print(f"  - {display_name} ({subject})")
    elif args.subject:
        task = MMLUTask(subject=args.subject)
        output_file = task.generate()
        display_name = args.subject.replace('_', ' ')
        print(f"\n🎉 MMLU task for {display_name} completed! Output saved to: {output_file}")
    elif args.all:
        generate_all_subjects()
    else:
        print("Please specify --subject <name>, --all, or --list-subjects")
        print("Use --list-subjects to see available subjects") 