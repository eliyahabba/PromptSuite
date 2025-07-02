#!/usr/bin/env python3
"""
Math Problems Task: GSM8K
This module provides a class for generating prompt variations for math problem solving tasks.
"""

from typing import Dict, Any, List
import argparse
import pandas as pd
from pathlib import Path
import sys
import re

from datasets import load_dataset
from promptsuite.core.template_keys import (
    INSTRUCTION, PROMPT_FORMAT, GOLD_KEY,
    PARAPHRASE_WITH_LLM, FORMAT_STRUCTURE_VARIATION, TYPOS_AND_NOISE_VARIATION,
    INSTRUCTION_VARIATIONS, PROMPT_FORMAT_VARIATIONS, FEW_SHOT_KEY
)
from base_task import BaseTask
from promptsuite_tasks.constants import (
    DEFAULT_VARIATIONS_PER_FIELD, DEFAULT_PLATFORM, DEFAULT_MODEL_NAME,
    DEFAULT_MAX_VARIATIONS_PER_ROW, DEFAULT_MAX_ROWS, DEFAULT_RANDOM_SEED
)


class MathTask(BaseTask):
    """Task for generating math problem solving prompt variations."""

    def __init__(self,
                 variations_per_field: int = DEFAULT_VARIATIONS_PER_FIELD,
                 api_platform: str = DEFAULT_PLATFORM,
                 model_name: str = DEFAULT_MODEL_NAME,
                 max_rows: int = DEFAULT_MAX_ROWS,
                 max_variations_per_row: int = DEFAULT_MAX_VARIATIONS_PER_ROW,
                 random_seed: int = DEFAULT_RANDOM_SEED):

        task_name = "Math Problems Task: GSM8K"
        output_filename = "math_gsm8k_variations.json"

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
        """Load GSM8K dataset from HuggingFace - both train and test splits, combine, and load as one DataFrame."""
        print("Loading GSM8K train dataset...")
        train_ds = load_dataset("gsm8k", "main", split="train[:100]")
        train_df = pd.DataFrame(train_ds)
        train_df['split'] = 'train'
        print(f"✅ Loaded {len(train_df)} train rows")

        print("Loading GSM8K test dataset...")
        test_ds = load_dataset("gsm8k", "main", split="test[:100]")
        test_df = pd.DataFrame(test_ds)
        test_df['split'] = 'test'
        print(f"✅ Loaded {len(test_df)} test rows")

        # Combine datasets
        df = pd.concat([train_df, test_df], ignore_index=True)
        print(f"✅ Combined total: {len(df)} rows")

        self.mp.load_dataframe(df)
        print("✅ Data loaded")

    def get_template(self) -> Dict[str, Any]:
        """Get template configuration for math problem solving task."""
        return {
            INSTRUCTION: "Let's think step by step. Solve the following math problem and provide the final numerical answer in the format #### answer.",
            # INSTRUCTION_VARIATIONS: [PARAPHRASE_WITH_LLM],
            PROMPT_FORMAT: "Question: {question}\nAnswer: {answer}",
            PROMPT_FORMAT_VARIATIONS: [FORMAT_STRUCTURE_VARIATION],
            'question': [TYPOS_AND_NOISE_VARIATION],
            GOLD_KEY: 'answer',  # The original answer field with full solution and #### format
            FEW_SHOT_KEY: {
                'count': 3,  # Number of few-shot examples
                'format': 'random_per_row',  # Random examples per row
                'split': 'train'  # Use training split for few-shot examples
            }
        }


def generate_math_variations(variations_per_field, api_platform, model_name, max_rows, max_variations_per_row, random_seed):
    """Generate variations for math problems task."""
    # Create output directory
    output_dir = Path(__file__).parent.parent / "tasks_data" / "generated_data" / "math"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("🎯 Processing Math Problems Task")
    print("=" * 50)

    try:
        task = MathTask(
            variations_per_field=variations_per_field,
            api_platform=api_platform,
            model_name=model_name,
            max_rows=max_rows,
            max_variations_per_row=max_variations_per_row,
            random_seed=random_seed
        )

        # Override output path to save in math folder
        output_file = output_dir / "math_gsm8k_variations.json"

        # Generate using custom path
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

        print(f"✅ Completed Math Problems Task: {output_file}")
        return str(output_file)

    except Exception as e:
        print(f"❌ Error processing math problems task: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate math problem solving prompt variations")
    parser.add_argument("--rows", type=int, help="Number of rows to process", default=DEFAULT_MAX_ROWS)
    parser.add_argument("--variations", type=int, help="Number of variations per row", default=DEFAULT_MAX_VARIATIONS_PER_ROW)
    parser.add_argument("--variations_per_field", type=int, default=DEFAULT_VARIATIONS_PER_FIELD)
    parser.add_argument("--api_platform", type=str, default=DEFAULT_PLATFORM)
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--random_seed", type=int, help="Random seed for generation", default=DEFAULT_RANDOM_SEED)
    args = parser.parse_args()

    task = MathTask(
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