#!/usr/bin/env python3
"""
Summarization Task: CNN DailyMail
This module provides a class for generating prompt variations for summarization tasks.
"""

from typing import Dict, Any

from multipromptify.core.template_keys import (
    INSTRUCTION, PROMPT_FORMAT, QUESTION_KEY, GOLD_KEY,
    PARAPHRASE_WITH_LLM, FORMAT_STRUCTURE_VARIATION, TYPOS_AND_NOISE_VARIATION,
    CONTEXT_VARIATION, INSTRUCTION_VARIATIONS, PROMPT_FORMAT_VARIATIONS
)
from multipromptify_tasks.tasks.base_task import BaseTask


class SummarizationTask(BaseTask):
    """Task for generating summarization prompt variations."""

    def __init__(self):
        super().__init__(
            task_name="Summarization Task: CNN DailyMail",
            output_filename="summarization_cnn_dailymail_variations.json"
        )

    def load_data(self) -> None:
        """Load CNN DailyMail dataset from HuggingFace."""
        try:
            self.mp.load_dataset("cnn_dailymail", "3.0.0", split="train[:100]")
            print("✅ Successfully loaded CNN DailyMail dataset")
        except Exception as e:
            print(f"❌ Error loading CNN DailyMail dataset: {e}")
            print("Trying alternative dataset...")
            # Fallback to a simpler summarization dataset
            self.mp.load_dataset("samsum", split="train[:100]")
            print("✅ Successfully loaded samsum dataset")

    def get_template(self) -> Dict[str, Any]:
        """Get template configuration for summarization task."""
        return {
            INSTRUCTION: "You are a professional summarizer. Create a concise summary of the following text.",
            INSTRUCTION_VARIATIONS: [PARAPHRASE_WITH_LLM],
            PROMPT_FORMAT: "Article: {article}\nSummary: {highlights}",
            PROMPT_FORMAT_VARIATIONS: [
                FORMAT_STRUCTURE_VARIATION,  # Semantic-preserving format changes
            ],
            'article':
                [TYPOS_AND_NOISE_VARIATION
                 ],  # Add noise to the article text
            GOLD_KEY: "highlights"  # The summary is the gold standard
        }


if __name__ == "__main__":
    task = SummarizationTask()
    output_file = task.generate()
    print(f"\n🎉 Summarization task completed! Output saved to: {output_file}")
