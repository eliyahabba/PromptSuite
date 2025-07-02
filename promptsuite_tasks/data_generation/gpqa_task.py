#!/usr/bin/env python3
"""
GPQA Task: Graduate-level Google-proof Q&A
This module provides a class for generating prompt variations for GPQA tasks.
"""

from typing import Dict, Any, List
import argparse
import pandas as pd
from pathlib import Path
import sys
import random

from datasets import load_dataset
from promptsuite.core.template_keys import (
    INSTRUCTION, PROMPT_FORMAT, GOLD_KEY,
    PARAPHRASE_WITH_LLM, FORMAT_STRUCTURE_VARIATION, TYPOS_AND_NOISE_VARIATION,
    INSTRUCTION_VARIATIONS, PROMPT_FORMAT_VARIATIONS, FEW_SHOT_KEY,
    SHUFFLE_VARIATION, ENUMERATE_VARIATION
)
from base_task import BaseTask
from promptsuite_tasks.constants import (
    DEFAULT_VARIATIONS_PER_FIELD, DEFAULT_PLATFORM, DEFAULT_MODEL_NAME,
    DEFAULT_MAX_VARIATIONS_PER_ROW, DEFAULT_MAX_ROWS, DEFAULT_RANDOM_SEED
)


class GPQATask(BaseTask):
    """Task for generating GPQA prompt variations."""

    def __init__(self,
                 variations_per_field: int = DEFAULT_VARIATIONS_PER_FIELD,
                 api_platform: str = DEFAULT_PLATFORM,
                 model_name: str = DEFAULT_MODEL_NAME,
                 max_rows: int = DEFAULT_MAX_ROWS,
                 max_variations_per_row: int = DEFAULT_MAX_VARIATIONS_PER_ROW,
                 random_seed: int = DEFAULT_RANDOM_SEED):

        task_name = "GPQA Task: Graduate-level Google-proof Q&A"
        output_filename = "gpqa_diamond_variations.json"

        super().__init__(
            task_name=task_name,
            output_filename=output_filename,
            variations_per_field=variations_per_field,
            api_platform=api_platform,
            model_name=model_name,
            max_rows=max_rows,
            max_variations_per_row=max_variations_per_row,
            random_seed=random_seed
        )

    def load_data(self) -> None:
        """Load GPQA Diamond dataset from HuggingFace and create train/test splits."""
        print("Loading GPQA Diamond dataset...")
        
        # Load the full dataset (only train split available)
        dataset = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
        df = pd.DataFrame(dataset)
        
        print(f"✅ Loaded {len(df)} total rows from GPQA Diamond")
        
        # Create train/test split manually since only train split is available
        total_rows = len(df)
        indices = list(range(total_rows))
        random.seed(42)  # Fixed seed for reproducible splits
        random.shuffle(indices)
        
        train_size = int(total_rows * 0.4)
        train_indices = set(indices[:train_size])
        
        # Add split column
        df['split'] = ['train' if i in train_indices else 'test' for i in range(total_rows)]
        
        train_count = sum(1 for split in df['split'] if split == 'train')
        test_count = sum(1 for split in df['split'] if split == 'test')
        
        print(f"✅ Created splits: {train_count} train, {test_count} test")
        
        self.mp.load_dataframe(df)
        self.post_process()
        print("✅ Data loaded and post-processed")

    def post_process(self) -> None:
        """Post-process GPQA data to create choices format like MMLU."""
        print("Post-processing GPQA data...")
        
        # Get the dataframe from PromptSuite
        df = self.mp.data
        
        # Create choices list combining all answer options (like MMLU format)
        df['choices'] = df.apply(lambda row: [
            row['Incorrect Answer 1'],
            row['Incorrect Answer 2'], 
            row['Incorrect Answer 3'],
            row['Correct Answer']
        ], axis=1)
        
        # Create answer index field (correct answer is always at index 3)
        df['answer'] = 3  # The correct answer is always the last item (index 3)
        
        # Update the data in PromptSuite
        self.mp.data = df
        
        print(f"✅ Created choices field with 4 options per question")
        print(f"✅ Created answer index field (correct answer at index 3)")

    def get_template(self) -> Dict[str, Any]:
        """Get template configuration for GPQA task."""
        return {
            INSTRUCTION: "Here are some example questions from experts. An explanation is given before the final answer. Answer the final question yourself, giving your reasoning beforehand.",
            # INSTRUCTION_VARIATIONS: [PARAPHRASE_WITH_LLM],
            PROMPT_FORMAT: "Question: {Question}\nChoices: {choices}\nAnswer: {answer}",
            PROMPT_FORMAT_VARIATIONS: [FORMAT_STRUCTURE_VARIATION],
            # 'Question': [TYPOS_AND_NOISE_VARIATION],
            'choices': [SHUFFLE_VARIATION, ENUMERATE_VARIATION],  # Add enumeration and shuffling like MMLU
            GOLD_KEY: {
                'field': 'answer',  # Use the index field we created
                'type': 'index',    # Use index-based approach like MMLU
                'options_field': 'choices'
            },
            FEW_SHOT_KEY: {
                'count': 5,  # Number of few-shot examples
                'format': 'random_per_row',  # Random examples per row
                'split': 'train'  # Use training split for few-shot examples
            }
        }

#
# def generate_gpqa_variations(variations_per_field, api_platform, model_name, max_rows, max_variations_per_row, random_seed):
#     """Generate variations for GPQA task."""
#     # Create output directory
#     output_dir = Path(__file__).parent.parent / "tasks_data" / "generated_data" / "gpqa"
#     output_dir.mkdir(parents=True, exist_ok=True)
#
#     print("🎯 Processing GPQA Task")
#     print("=" * 50)
#
#     try:
#         task = GPQATask(
#             variations_per_field=variations_per_field,
#             api_platform=api_platform,
#             model_name=model_name,
#             max_rows=max_rows,
#             max_variations_per_row=max_variations_per_row,
#             random_seed=random_seed
#         )
#
#         # Override output path to save in gpqa folder
#         output_file = output_dir / "gpqa_diamond_variations.json"
#
#         # Generate using custom path
#         print(f"🚀 Starting {task.task_name}")
#         print("=" * 60)
#         print("\n1. Loading data...")
#         task.load_data()
#         print("\n2. Setting up template...")
#         template = task.get_template()
#         task.mp.set_template(template)
#         print("✅ Template configured")
#         print(f"\n3. Configuring generation...")
#         print(f"   Variations per field: {task.variations_per_field}")
#         print(f"   API Platform: {task.api_platform}")
#         print(f"   Model: {task.model_name}")
#         print(f"   Max rows: {task.max_rows}")
#         print(f"   Max variations per row: {task.max_variations_per_row}")
#         print(f"   Random seed: {task.random_seed}")
#         task.mp.configure(
#             max_rows=task.max_rows,
#             variations_per_field=task.variations_per_field,
#             max_variations_per_row=task.max_variations_per_row,
#             random_seed=task.random_seed,
#             api_platform=task.api_platform,
#             model_name=task.model_name
#         )
#         print("\n4. Generating prompt variations...")
#         variations = task.mp.generate(verbose=True)
#
#         # Display results
#         print(f"\n✅ Generated {len(variations)} variations")
#
#         # Show a few examples
#         print("\n5. Sample variations:")
#         for i, var in enumerate(variations[:3]):
#             print(f"\nVariation {i + 1}:")
#             print("-" * 50)
#             prompt = var.get('prompt', 'No prompt found')
#             if len(prompt) > 500:
#                 prompt = prompt[:500] + "..."
#             print(prompt)
#             print("-" * 50)
#
#         # Export results using the correct path
#         print(f"\n6. Exporting results to {output_file}...")
#         task.mp.export(str(output_file), format="json")
#         print("✅ Export completed!")
#
#         # Show final statistics
#         print("\n7. Final statistics:")
#         task.mp.info()
#
#         print(f"✅ Completed GPQA Task: {output_file}")
#         return str(output_file)
#
#     except Exception as e:
#         print(f"❌ Error processing GPQA task: {e}")
#         raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate GPQA prompt variations")
    parser.add_argument("--rows", type=int, help="Number of rows to process", default=DEFAULT_MAX_ROWS)
    parser.add_argument("--variations", type=int, help="Number of variations per row", default=DEFAULT_MAX_VARIATIONS_PER_ROW)
    parser.add_argument("--variations_per_field", type=int, default=DEFAULT_VARIATIONS_PER_FIELD)
    parser.add_argument("--api_platform", type=str, default=DEFAULT_PLATFORM)
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--random_seed", type=int, help="Random seed for generation", default=DEFAULT_RANDOM_SEED)
    args = parser.parse_args()

    task = GPQATask(
        variations_per_field=args.variations_per_field,
        api_platform=args.api_platform,
        model_name=args.model_name,
        max_rows=args.rows,
        max_variations_per_row=args.variations,
        random_seed=args.random_seed
    )
    if args.rows is not None or args.variations is not None:
        task.override_config(rows=args.rows, variations=args.variations)
    task.generate() 