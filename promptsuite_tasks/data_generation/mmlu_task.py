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

# Add the project root to the path to import promptsuite and promptsuite_tasks
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Add current directory to path for local imports
current_dir = Path(__file__).parent.parent
sys.path.insert(0, str(current_dir))

from promptsuite.core.template_keys import (
    INSTRUCTION, PROMPT_FORMAT, QUESTION_KEY, OPTIONS_KEY, GOLD_KEY,
    PARAPHRASE_WITH_LLM, FORMAT_STRUCTURE_VARIATION, TYPOS_AND_NOISE_VARIATION,
    SHUFFLE_VARIATION, ENUMERATE_VARIATION, INSTRUCTION_VARIATIONS, PROMPT_FORMAT_VARIATIONS,
    FEW_SHOT_KEY
)

from .base_task import BaseTask
from constants import (
    DEFAULT_VARIATIONS_PER_FIELD, DEFAULT_PLATFORM, DEFAULT_MODEL_NAME,
    DEFAULT_MAX_VARIATIONS_PER_ROW, DEFAULT_MAX_ROWS, DEFAULT_RANDOM_SEED
)


class MMLUTask(BaseTask):
    """Task for generating MMLU prompt variations by subject."""
    
    def __init__(self, subject: str = None,
                 variations_per_field: int = DEFAULT_VARIATIONS_PER_FIELD,
                 api_platform: str = DEFAULT_PLATFORM,
                 model_name: str = DEFAULT_MODEL_NAME,
                 max_rows: int = DEFAULT_MAX_ROWS,
                 max_variations_per_row: int = DEFAULT_MAX_VARIATIONS_PER_ROW,
                 random_seed: int = DEFAULT_RANDOM_SEED):
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
            output_filename=output_filename,
            variations_per_field=variations_per_field,
            api_platform=api_platform,
            model_name=model_name,
            max_rows=max_rows,  # Pass through from MMLUTask __init__
            max_variations_per_row=max_variations_per_row,  # Pass through
            random_seed=random_seed  # Pass through
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
                'split': 'train'
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


def generate_all_subjects(variations_per_field, api_platform, model_name, max_rows, max_variations_per_row, random_seed):
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
            task = MMLUTask(
                subject=subject,
                variations_per_field=variations_per_field,
                api_platform=api_platform,
                model_name=model_name,
                max_rows=max_rows,  # Pass from generate_all_subjects
                max_variations_per_row=max_variations_per_row,  # Pass from generate_all_subjects
                random_seed=random_seed  # Pass from generate_all_subjects
            )
            
            # Override output path to save in mmlu folder
            output_file = output_dir / f"mmlu_{subject}_variations.json"
            
            # Create a custom generate method that uses the correct path
            def custom_generate():
                """Custom generate method that uses the correct output path."""
                print(f"🚀 Starting {task.task_name}")
                print("=" * 60)
                print("\n1. Loading data...")
                task.load_data()
                print("\n2. Setting up template...")
                template = task.get_template()
                task.mp.set_template(template)
                print("✅ Template configured")
                print(f"\n3. Configuring generation...")
                print(f"   Variations per field: {task.variations_per_field}")
                print(f"   API Platform: {task.api_platform}")
                print(f"   Model: {task.model_name}")
                print(f"   Max rows: {task.max_rows}")
                print(f"   Max variations per row: {task.max_variations_per_row}")
                print(f"   Random seed: {task.random_seed}")
                task.mp.configure(
                    max_rows=task.max_rows,
                    variations_per_field=task.variations_per_field,
                    max_variations_per_row=task.max_variations_per_row,
                    random_seed=task.random_seed,
                    api_platform=task.api_platform,
                    model_name=task.model_name
                )
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


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate MMLU prompt variations")
    parser.add_argument("--subject", help="Specific subject to process (e.g., anatomy, chemistry)", default='international_law')
    parser.add_argument("--all", action="store_true", help="Process all subjects")
    parser.add_argument("--rows", type=int, help="Number of rows to process", default=10)
    parser.add_argument("--variations", type=int, help="Number of variations per row", default=DEFAULT_MAX_VARIATIONS_PER_ROW)
    parser.add_argument("--variations_per_field", type=int, default=DEFAULT_VARIATIONS_PER_FIELD)
    parser.add_argument("--api_platform", type=str, default=DEFAULT_PLATFORM)
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--random_seed", type=int, help="Random seed for generation", default=DEFAULT_RANDOM_SEED)
    args = parser.parse_args()
    
    if args.all:
        generate_all_subjects(
            variations_per_field=args.variations_per_field,
            api_platform=args.api_platform,
            model_name=args.model_name,
            max_rows=args.rows,  # Pass through from argparse
            max_variations_per_row=args.variations,  # Pass through
            random_seed=args.random_seed  # Pass through
        )
    elif args.subject:
        task = MMLUTask(
            subject=args.subject,
            variations_per_field=args.variations_per_field,
            api_platform=args.api_platform,
            model_name=args.model_name,
            max_rows=args.rows,  # Pass from argparse to MMLUTask __init__
            max_variations_per_row=args.variations,  # Pass from argparse
            random_seed=args.random_seed  # Pass from argparse
        )
        if args.rows is not None or args.variations is not None:
            task.override_config(rows=args.rows, variations=args.variations)
        task.generate()
    else:
        print("Please specify either --subject <subject_name> or --all")
        print("\nAvailable subjects:")
        try:
            subjects = get_available_subjects()
            for subject in subjects:
                display_name = subject.replace('_', ' ')
                print(f"  - {display_name} ({subject})")
        except Exception as e:
            print(f"Error getting subjects: {e}") 