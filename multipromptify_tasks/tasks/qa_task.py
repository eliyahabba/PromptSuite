#!/usr/bin/env python3
"""
Question Answering Task: SQuAD
This module provides a class for generating prompt variations for question answering tasks.
"""

from typing import Dict, Any

from multipromptify.core import FEW_SHOT_KEY
from multipromptify.core.template_keys import (
    INSTRUCTION, PROMPT_FORMAT, QUESTION_KEY, GOLD_KEY, CONTEXT_KEY,
    PARAPHRASE_WITH_LLM, FORMAT_STRUCTURE_VARIATION, TYPOS_AND_NOISE_VARIATION, INSTRUCTION_VARIATIONS,
    PROMPT_FORMAT_VARIATIONS, CONTEXT_VARIATION
)
from multipromptify_tasks.tasks.base_task import BaseTask


class QATask(BaseTask):
    """Task for generating question answering prompt variations."""
    
    def __init__(self):
        super().__init__(
            task_name="Question Answering Task: SQuAD",
            output_filename="question_answering_squad_variations.json"
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
    task = QATask()
    output_file = task.generate()
    print(f"\n🎉 Question answering task completed! Output saved to: {output_file}") 