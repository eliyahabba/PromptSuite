#!/usr/bin/env python3
"""
Question Answering Task: SQuAD
This module provides a class for generating prompt variations for question answering tasks.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any
import pandas as pd
from datasets import load_dataset

# Add the project root to the path to import promptsuite and promptsuite_tasks
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Add current directory to path for local imports
current_dir = Path(__file__).parent.parent
sys.path.insert(0, str(current_dir))

from promptsuite.core import FEW_SHOT_KEY
from promptsuite.core.template_keys import (
    INSTRUCTION, PROMPT_FORMAT, GOLD_KEY, CONTEXT_KEY,
    PARAPHRASE_WITH_LLM, FORMAT_STRUCTURE_VARIATION, TYPOS_AND_NOISE_VARIATION, INSTRUCTION_VARIATIONS,
    PROMPT_FORMAT_VARIATIONS, CONTEXT_VARIATION
)
from base_task import BaseTask

from promptsuite_tasks.constants import (
    DEFAULT_VARIATIONS_PER_FIELD, DEFAULT_PLATFORM, DEFAULT_MODEL_NAME,
    DEFAULT_MAX_VARIATIONS_PER_ROW, DEFAULT_MAX_ROWS, DEFAULT_RANDOM_SEED
)


class QATask(BaseTask):
    """Task for generating question answering prompt variations."""

    def __init__(self, variations_per_field: int = DEFAULT_VARIATIONS_PER_FIELD,
                 api_platform: str = DEFAULT_PLATFORM,
                 model_name: str = DEFAULT_MODEL_NAME,
                 max_rows: int = DEFAULT_MAX_ROWS,
                 max_variations_per_row: int = DEFAULT_MAX_VARIATIONS_PER_ROW,
                 random_seed: int = DEFAULT_RANDOM_SEED):
        super().__init__(
            task_name="Question Answering Task: SQuAD",
            output_filename="question_answering_squad_variations.json",
            variations_per_field=variations_per_field,
            api_platform=api_platform,
            model_name=model_name,
            max_rows=max_rows,
            max_variations_per_row=max_variations_per_row,
            random_seed=random_seed
        )

    def load_data(self) -> None:
        """Load SQuAD dataset from HuggingFace - both train and test splits, combine, and load as one DataFrame."""
        print("Loading SQuAD train dataset...")
        train_ds = load_dataset("rajpurkar/squad", split="train[:100]")
        train_df = pd.DataFrame(train_ds)
        train_df['split'] = 'train'
        print(f"✅ Loaded {len(train_df)} train rows")

        print("Loading SQuAD test dataset...")
        test_ds = load_dataset("rajpurkar/squad", split="validation[:100]")
        test_df = pd.DataFrame(test_ds)
        test_df['split'] = 'test'
        print(f"✅ Loaded {len(test_df)} test rows")

        # Combine
        df = pd.concat([train_df, test_df], ignore_index=True)
        print(f"✅ Combined total: {len(df)} rows")

        # Load into promptsuite
        self.mp.load_dataframe(df)
        self.post_process()
        print("✅ Data post-processed")

    def post_process(self) -> None:
        """Extract answer text from SQuAD answers structure."""
        self.mp.data['answer'] = self.mp.data['answers'].apply(lambda x: x['text'][0])
        print(f"✅ Processed {len(self.mp.data)} rows:")
        print(f"   - Train: {len(self.mp.data[self.mp.data['split'] == 'train'])} rows")
        print(f"   - Test: {len(self.mp.data[self.mp.data['split'] == 'test'])} rows")

    def get_template(self) -> Dict[str, Any]:
        """Get template configuration for question answering task."""
        return {
            INSTRUCTION: "Given the following context, answer the question as accurately and concisely as possible.",
            INSTRUCTION_VARIATIONS: [PARAPHRASE_WITH_LLM],  # AI-powered rephrasing of instructions
            PROMPT_FORMAT: "Context: {context}\nQuestion: {question}\nAnswer: {answer}",
            PROMPT_FORMAT_VARIATIONS: [FORMAT_STRUCTURE_VARIATION],  # Semantic-preserving format changes
            CONTEXT_KEY: [
                # CONTEXT_VARIATION,  # Context for the question
                TYPOS_AND_NOISE_VARIATION,  # Robustness testing with noise
            ],
            FEW_SHOT_KEY: {
                'count': 3,  # Reduced from 5 to work with smaller datasets
                'format': 'random_per_row',
                'split': 'train'
            },
            GOLD_KEY: "answer"  # The answer text is the gold standard
        }


def process(variations_per_field=DEFAULT_VARIATIONS_PER_FIELD,
            api_platform=DEFAULT_PLATFORM,
            model_name=DEFAULT_MODEL_NAME,
            max_rows=DEFAULT_MAX_ROWS,
            max_variations_per_row=DEFAULT_MAX_VARIATIONS_PER_ROW,
            random_seed=DEFAULT_RANDOM_SEED):
    """Process QA task with train/test split functionality."""
    # Create output directory
    output_dir = Path(__file__).parent.parent / "tasks_data" / "generated_data" / "qa"
    output_dir.mkdir(parents=True, exist_ok=True)
    print("🎯 Processing Question Answering Task: SQuAD")
    print("=" * 50)

    try:
        task = QATask(
            variations_per_field=variations_per_field,
            api_platform=api_platform,
            model_name=model_name,
            max_rows=max_rows,
            max_variations_per_row=max_variations_per_row,
            random_seed=random_seed
        )

        # Override output path to save in data folder
        output_file = output_dir / "question_answering_squad_variations.json"

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
        print(f"✅ Completed Question Answering Task: {generated_file}")
        return generated_file

    except Exception as e:
        print(f"❌ Error processing Question Answering Task: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Question Answering prompt variations")
    parser.add_argument("--rows", type=int, help="Number of rows to process", default=DEFAULT_MAX_ROWS)
    parser.add_argument("--variations", type=int, help="Number of variations per row",
                        default=DEFAULT_MAX_VARIATIONS_PER_ROW)
    parser.add_argument("--variations_per_field", type=int, default=DEFAULT_VARIATIONS_PER_FIELD)
    parser.add_argument("--api_platform", type=str, default=DEFAULT_PLATFORM)
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--random_seed", type=int, help="Random seed for generation", default=DEFAULT_RANDOM_SEED)
    args = parser.parse_args()

    output_file = process(
        variations_per_field=args.variations_per_field,
        api_platform=args.api_platform,
        model_name=args.model_name,
        max_rows=args.rows,
        max_variations_per_row=args.variations,
        random_seed=args.random_seed
    )
    print(f"\n🎉 Question answering task completed! Output saved to: {output_file}")
