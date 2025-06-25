#!/usr/bin/env python3
"""
Translation Task: WMT14 English to German
This module provides a class for generating prompt variations for translation tasks.
"""

from typing import Dict, Any
import argparse

from multipromptify.core import PROMPT_FORMAT_VARIATIONS, FEW_SHOT_KEY
from multipromptify.core.template_keys import (
    INSTRUCTION, PROMPT_FORMAT, QUESTION_KEY, GOLD_KEY,
    PARAPHRASE_WITH_LLM, FORMAT_STRUCTURE_VARIATION, TYPOS_AND_NOISE_VARIATION, INSTRUCTION_VARIATIONS
)
from multipromptify_tasks.tasks.base_task import BaseTask
from multipromptify_tasks.constants import (
    DEFAULT_VARIATIONS_PER_FIELD, DEFAULT_PLATFORM, DEFAULT_MODEL_NAME,
    DEFAULT_MAX_VARIATIONS_PER_ROW, DEFAULT_MAX_ROWS, DEFAULT_RANDOM_SEED
)


class TranslationTask(BaseTask):
    """Task for generating translation prompt variations."""
    
    def __init__(self, variations_per_field: int = DEFAULT_VARIATIONS_PER_FIELD, api_platform: str = DEFAULT_PLATFORM, model_name: str = DEFAULT_MODEL_NAME,
                 max_rows: int = DEFAULT_MAX_ROWS, max_variations_per_row: int = DEFAULT_MAX_VARIATIONS_PER_ROW, random_seed: int = DEFAULT_RANDOM_SEED):
        super().__init__(
            task_name="Translation Task: WMT14 English to German",
            output_filename="translation_wmt14_en_de_variations.json",
            variations_per_field=variations_per_field,
            api_platform=api_platform,
            model_name=model_name,
            max_rows=max_rows,
            max_variations_per_row=max_variations_per_row,
            random_seed=random_seed
        )

    def load_data(self) -> None:
        """Load WMT14 dataset from HuggingFace."""
        try:
            self.mp.load_dataset("wmt14", "de-en", split="train[:100]")
            print("✅ Successfully loaded WMT14 dataset")
        except Exception as e:
            print(f"❌ Error loading WMT14 dataset: {e}")
            print("Trying alternative dataset...")
            # Fallback to a simpler translation dataset
            self.mp.load_dataset("opus_books", "en-de", split="train[:100]")
            print("✅ Successfully loaded opus_books dataset")
        
        # Post-process the data to flatten translation structure
        self.post_process()
        print("✅ Data post-processed")

    def post_process(self) -> None:
        """Flatten translation dictionary structure."""
        self.mp.data['en'] = self.mp.data['translation'].apply(lambda x: x['en'])
        self.mp.data['de'] = self.mp.data['translation'].apply(lambda x: x['de'])

    def get_template(self) -> Dict[str, Any]:
        """Get template configuration for translation task."""
        return {
            INSTRUCTION: "You are a professional translator. Translate the following English text to German.",
            INSTRUCTION_VARIATIONS: [PARAPHRASE_WITH_LLM],
            PROMPT_FORMAT: "English: {en}\nGerman: {de}",
            PROMPT_FORMAT_VARIATIONS: [FORMAT_STRUCTURE_VARIATION,TYPOS_AND_NOISE_VARIATION],
            GOLD_KEY: "de",  # German translation is the gold standard
            FEW_SHOT_KEY: {
                'count': 2,  # Reduced from 5 to work with smaller datasets
                'format': 'random_per_row',
                'split': 'all'
            },
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

    task = TranslationTask(
        variations_per_field=args.variations_per_field,
        api_platform=args.api_platform,
        model_name=args.model_name,
        max_rows=args.max_rows,
        max_variations_per_row=args.max_variations_per_row,
        random_seed=args.random_seed
    )
    output_file = task.generate()
    print(f"\n🎉 Translation task completed! Output saved to: {output_file}") 