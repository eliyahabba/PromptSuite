#!/usr/bin/env python3
"""
Code Generation Task: HumanEval
This module provides a class for generating prompt variations for code generation tasks.
"""

import argparse
import random
from typing import Dict, Any

import pandas as pd
from datasets import load_dataset

from base_task import BaseTask
from multipromptify.core.template_keys import (
    INSTRUCTION, PROMPT_FORMAT, GOLD_KEY,
    FORMAT_STRUCTURE_VARIATION, TYPOS_AND_NOISE_VARIATION,
    PROMPT_FORMAT_VARIATIONS, FEW_SHOT_KEY, INSTRUCTION_VARIATIONS, PARAPHRASE_WITH_LLM
)
from multipromptify_tasks.constants import (
    DEFAULT_VARIATIONS_PER_FIELD, DEFAULT_PLATFORM, DEFAULT_MODEL_NAME,
    DEFAULT_MAX_VARIATIONS_PER_ROW, DEFAULT_MAX_ROWS, DEFAULT_RANDOM_SEED
)


class CodeGenerationTask(BaseTask):
    """Task for generating code generation prompt variations using HumanEval dataset."""

    def __init__(self,
                 variations_per_field: int = DEFAULT_VARIATIONS_PER_FIELD,
                 api_platform: str = DEFAULT_PLATFORM,
                 model_name: str = DEFAULT_MODEL_NAME,
                 max_rows: int = DEFAULT_MAX_ROWS,
                 max_variations_per_row: int = DEFAULT_MAX_VARIATIONS_PER_ROW,
                 random_seed: int = DEFAULT_RANDOM_SEED):

        task_name = "Code Generation Task: HumanEval"
        output_filename = "code_generation_humaneval_variations.json"

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
        """Load HumanEval dataset using datasets library."""
        print("Loading HumanEval dataset...")

        try:
            # Load HumanEval dataset
            dataset = load_dataset("openai_humaneval", split="test")
            print(f"✅ Loaded HumanEval dataset: {len(dataset)} problems")

            # Convert to DataFrame format expected by MultiPromptifier
            df_data = []
            for i, example in enumerate(dataset):
                # Extract task_id, prompt, and canonical_solution
                task_id = example['task_id']
                prompt = example['prompt']
                canonical_solution = example['canonical_solution']

                # Create entry for MultiPromptifier
                df_data.append({
                    'task_id': task_id,
                    'prompt': prompt,
                    'canonical_solution': canonical_solution,
                    'split': 'test'  # All HumanEval data is test split
                })

            # Convert to DataFrame and load into MultiPromptifier
            df = pd.DataFrame(df_data)

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
            # save as df in
            # save with pandas
            df.to_csv('/Users/ehabba/PycharmProjects/MultiPromptifyPipeline/multipromptify_tasks/tasks_data/raw_data/code_generation_humaneval.csv', index=False)
            print(f"✅ Loaded {len(df)} code generation problems")

        except Exception as e:
            print(f"❌ Error loading HumanEval dataset: {e}")
            raise ValueError(f"Failed to load HumanEval data: {e}")

    def get_template(self) -> Dict[str, Any]:
        """Get template configuration for code generation task."""
        return {
            INSTRUCTION: "You are an expert Python programmer helping users solve programming problems. When the user gives you a problem description and function signature, respond with a correct and concise Python implementation. Only return the body of the function (no extra explanation). Your solutions should be efficient, readable, and match any examples provided. Avoid unnecessary comments unless needed for clarity. Please generate Python code with proper indentation",
            INSTRUCTION_VARIATIONS: [PARAPHRASE_WITH_LLM],
            PROMPT_FORMAT: "Problem:\n{prompt}\n\nSolution:\n{canonical_solution}",
            PROMPT_FORMAT_VARIATIONS: [FORMAT_STRUCTURE_VARIATION],
            GOLD_KEY: 'canonical_solution',  # The canonical solution is the gold standard
            FEW_SHOT_KEY: {
                'count': 5,  # Use 2 few-shot examples
                'format': 'random_per_row',
                'split': 'train'  # Use test split for few-shot (since all data is test)
            }
        }


def main():
    """Main function to generate code generation variations."""
    parser = argparse.ArgumentParser(description="Generate code generation prompt variations")
    parser.add_argument("--rows", type=int, help="Number of rows to process", default=DEFAULT_MAX_ROWS)
    parser.add_argument("--variations", type=int, help="Number of variations per row",
                        default=DEFAULT_MAX_VARIATIONS_PER_ROW)
    parser.add_argument("--variations_per_field", type=int, default=DEFAULT_VARIATIONS_PER_FIELD)
    parser.add_argument("--api_platform", type=str, default=DEFAULT_PLATFORM)
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--random_seed", type=int, help="Random seed for generation", default=DEFAULT_RANDOM_SEED)
    args = parser.parse_args()

    task = CodeGenerationTask(
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


if __name__ == "__main__":
    main()
