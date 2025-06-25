#!/usr/bin/env python3
"""
Question Answering Task: SQuAD
This module provides a class for generating prompt variations for question answering tasks.
"""

from typing import Dict, Any
import argparse

from multipromptify.core import FEW_SHOT_KEY
from multipromptify.core.template_keys import (
    INSTRUCTION, PROMPT_FORMAT, QUESTION_KEY, GOLD_KEY, CONTEXT_KEY,
    PARAPHRASE_WITH_LLM, FORMAT_STRUCTURE_VARIATION, TYPOS_AND_NOISE_VARIATION, INSTRUCTION_VARIATIONS,
    PROMPT_FORMAT_VARIATIONS, CONTEXT_VARIATION
)
from multipromptify_tasks.tasks.base_task import BaseTask
from multipromptify_tasks.constants import (
    DEFAULT_VARIATIONS_PER_FIELD, DEFAULT_PLATFORM, DEFAULT_MODEL_NAME,
    DEFAULT_MAX_VARIATIONS_PER_ROW, DEFAULT_MAX_ROWS, DEFAULT_RANDOM_SEED
)


class QATask(BaseTask):
    """Task for generating question answering prompt variations."""
    
    def __init__(self, variations_per_field: int = DEFAULT_VARIATIONS_PER_FIELD, api_platform: str = DEFAULT_PLATFORM, model_name: str = DEFAULT_MODEL_NAME,
                 max_rows: int = DEFAULT_MAX_ROWS, max_variations_per_row: int = DEFAULT_MAX_VARIATIONS_PER_ROW, random_seed: int = DEFAULT_RANDOM_SEED):
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
        """Load SQuAD dataset from HuggingFace."""
        try:
            self.mp.load_dataset("squad", split="train[:100]")
            print("✅ Successfully loaded SQuAD dataset")
        except Exception as e:
            print(f"❌ Error loading SQuAD dataset: {e}")
            print("Trying alternative dataset...")
            # Fallback to a simpler QA dataset
            self.mp.load_dataset("squad_v2", split="train[:100]")
            print("✅ Successfully loaded SQuAD v2 dataset")
        self.post_process()
        print("✅ Data post-processed")

    def post_process(self) -> None:
        """Extract answer text from SQuAD answers structure."""
        self.mp.data['answer'] = self.mp.data['answers'].apply(lambda x: x['text'][0] if x['text'] else "")

    def get_template(self) -> Dict[str, Any]:
        """Get template configuration for question answering task."""
        return {
            INSTRUCTION: "You are a helpful assistant. Answer the question based on the given context.",
            INSTRUCTION_VARIATIONS: [PARAPHRASE_WITH_LLM],  # AI-powered rephrasing of instructions
            PROMPT_FORMAT: "Context: {context}\nQuestion: {question}\nAnswer: {answer}",
            PROMPT_FORMAT_VARIATIONS: [FORMAT_STRUCTURE_VARIATION],  # Semantic-preserving format changes
            CONTEXT_KEY: [
                CONTEXT_VARIATION,  # Context for the question
                TYPOS_AND_NOISE_VARIATION,   # Robustness testing with noise
            ],
            FEW_SHOT_KEY: {
                'count': 2,  # Reduced from 5 to work with smaller datasets
                'format': 'random_per_row',
                'split': 'all'
            },
            GOLD_KEY: "answer"  # The answer text is the gold standard
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variations_per_field", type=int, default=DEFAULT_VARIATIONS_PER_FIELD)
    parser.add_argument("--api_platform", type=str, default=DEFAULT_PLATFORM)
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--max_rows", type=int, default=DEFAULT_MAX_ROWS)
    parser.add_argument("--max_variations_per_row", type=int, default=DEFAULT_MAX_VARIATIONS_PER_ROW)
    parser.add_argument("--random_seed", type=int, default=DEFAULT_RANDOM_SEED)
    args = parser.parse_args()

    task = QATask(
        variations_per_field=args.variations_per_field,
        api_platform=args.api_platform,
        model_name=args.model_name,
        max_rows=args.max_rows,
        max_variations_per_row=args.max_variations_per_row,
        random_seed=args.random_seed
    )
    output_file = task.generate()
    print(f"\n🎉 Question answering task completed! Output saved to: {output_file}") 