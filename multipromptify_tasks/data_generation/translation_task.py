#!/usr/bin/env python3
"""
Translation Task: Multi-language pairs
This module provides a class for generating prompt variations for translation tasks.
"""

from typing import Dict, Any, List
import argparse
import pandas as pd
from pathlib import Path
import os
import sys

# Add the project root to the path to import multipromptify and multipromptify_tasks
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Add current directory to path for local imports
current_dir = Path(__file__).parent.parent
sys.path.insert(0, str(current_dir))

from multipromptify.core import PROMPT_FORMAT_VARIATIONS, FEW_SHOT_KEY
from multipromptify.core.template_keys import (
    INSTRUCTION, PROMPT_FORMAT, QUESTION_KEY, GOLD_KEY,
    PARAPHRASE_WITH_LLM, FORMAT_STRUCTURE_VARIATION, TYPOS_AND_NOISE_VARIATION, INSTRUCTION_VARIATIONS
)
from .base_task import BaseTask
from constants import (
    DEFAULT_VARIATIONS_PER_FIELD, DEFAULT_PLATFORM, DEFAULT_MODEL_NAME,
    DEFAULT_MAX_VARIATIONS_PER_ROW, DEFAULT_MAX_ROWS, DEFAULT_RANDOM_SEED
)


class TranslationTask(BaseTask):
    """Task for generating translation prompt variations."""
    
    def __init__(self, language_pair: str, variations_per_field: int = DEFAULT_VARIATIONS_PER_FIELD, api_platform: str = DEFAULT_PLATFORM, model_name: str = DEFAULT_MODEL_NAME,
                 max_rows: int = DEFAULT_MAX_ROWS, max_variations_per_row: int = DEFAULT_MAX_VARIATIONS_PER_ROW, random_seed: int = DEFAULT_RANDOM_SEED):
        self.language_pair = language_pair
        # Convert language pair name for display (e.g., cs-en -> Czech to English)
        display_pair = TranslationTask._get_display_name(language_pair)
        task_name = f"Translation Task: {display_pair}"
        output_filename = f"translation_{language_pair}_variations.json"
        
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
    
    @staticmethod
    def _get_display_name(language_pair: str) -> str:
        """Convert language pair code to display name."""
        language_names = {
            'cs': 'Czech',
            'de': 'German', 
            'fr': 'French',
            'hi': 'Hindi',
            'ru': 'Russian',
            'en': 'English'
        }
        
        if '-' in language_pair:
            source, target = language_pair.split('-')
            source_name = language_names.get(source, source.upper())
            target_name = language_names.get(target, target.upper())
            return f"{source_name} to {target_name}"
        return language_pair

    def load_data(self) -> None:
        """Load translation data using datasets library."""
        display_pair = TranslationTask._get_display_name(self.language_pair)
        print(f"Loading WMT14 data for {display_pair} ({self.language_pair})")
        
        # Determine the WMT14 config to use
        # WMT14 only has X->EN configs, so for EN->X we load the reverse and swap
        if self.language_pair.startswith('en-'):
            # For EN->X, load X->EN and we'll reverse it in post_process
            reverse_pair = self.language_pair.split('-')[1] + '-en'  # en-de -> de-en
            wmt14_config = reverse_pair
            self.is_reversed = True
            print(f"Loading reverse config {wmt14_config} for {display_pair}")
        else:
            # For X->EN, load directly
            wmt14_config = self.language_pair
            self.is_reversed = False
        
        # Load WMT14 dataset for the specific language pair
        # Use train[:100] to get first 100 examples for faster processing
        try:
            self.mp.load_dataset("wmt14", wmt14_config, split="train[:100]")
            print(f"✅ Loaded WMT14 dataset for {display_pair}: {len(self.mp.data)} rows")
        except Exception as e:
            print(f"❌ Error loading WMT14 dataset for {wmt14_config}: {e}")
            raise ValueError(f"Failed to load WMT14 data for language pair: {self.language_pair}")
        
        # Post-process the data to create language-specific columns and add split info
        self.post_process()
        
        train_count = sum(1 for split in self.mp.data['split'] if split == 'train')
        test_count = sum(1 for split in self.mp.data['split'] if split == 'test')
        print(f"✅ Data processed for {display_pair} ({train_count} train, {test_count} test)")

    def post_process(self) -> None:
        """Create language-specific columns from WMT14 translation data and add split info."""
        # Extract source and target language codes from language_pair (e.g., "de-en" -> "de", "en")
        source_code, target_code = self.language_pair.split('-')
        
        # WMT14 data structure: DataFrame with 'translation' column containing dicts
        # Each dict has language codes as keys (e.g., {'de': 'text', 'en': 'text'})
        if hasattr(self, 'is_reversed') and self.is_reversed:
            # For EN->X pairs, we loaded X->EN and need to reverse
            # The WMT14 data has the reverse mapping, so we swap source/target
            wmt_source = target_code  # 'en' in the WMT14 data
            wmt_target = source_code  # 'de' in the WMT14 data
            self.mp.data[source_code] = [row[wmt_source] for row in self.mp.data['translation']]  # en text
            self.mp.data[target_code] = [row[wmt_target] for row in self.mp.data['translation']]  # de text
        else:
            # For X->EN pairs, use directly
            self.mp.data[source_code] = [row[source_code] for row in self.mp.data['translation']]
            self.mp.data[target_code] = [row[target_code] for row in self.mp.data['translation']]
        
        # Add train/test split - use 80/20 split with fixed seed for reproducibility
        import random
        total_rows = len(self.mp.data)
        indices = list(range(total_rows))
        random.seed(42)  # Fixed seed for reproducible splits
        random.shuffle(indices)
        
        train_size = int(total_rows * 0.8)
        train_indices = set(indices[:train_size])
        
        self.mp.data['split'] = ['train' if i in train_indices else 'test' for i in range(total_rows)]

    def get_template(self) -> Dict[str, Any]:
        """Get template configuration for translation task."""
        # Dynamic template based on language pair
        display_pair = TranslationTask._get_display_name(self.language_pair)
        # Extract source and target language names and codes
        source_lang, target_lang = display_pair.split(' to ')
        source_code, target_code = self.language_pair.split('-')
        return {
            INSTRUCTION: f"You are a professional translator. Translate the following text from {display_pair}.",
            INSTRUCTION_VARIATIONS: [PARAPHRASE_WITH_LLM],
            PROMPT_FORMAT: f"{source_lang}: {{{source_code}}}\n{target_lang}: {{{target_code}}}",
            PROMPT_FORMAT_VARIATIONS: [FORMAT_STRUCTURE_VARIATION, TYPOS_AND_NOISE_VARIATION],
            GOLD_KEY: target_code,  # Target language code is the gold standard
            FEW_SHOT_KEY: {
                'count': 2,  # Reduced from 5 to work with smaller datasets
                'format': 'random_per_row',
                'split': 'train'
            },
        }


def get_available_language_pairs() -> List[str]:
    """Get list of available language pairs from WMT14 dataset."""
    # Selected language pairs: Czech, Russian, Hindi (both directions)
    selected_pairs = [
        # To English
        "cs-en",  # Czech-English
        "hi-en",  # Hindi-English
        "ru-en",  # Russian-English
        # From English
        "en-cs",  # English-Czech
        "en-hi",  # English-Hindi
        "en-ru",  # English-Russian
    ]
    
    return sorted(selected_pairs)


def generate_all_language_pairs(variations_per_field, api_platform, model_name, max_rows, max_variations_per_row, random_seed):
    """Generate variations for all language pairs separately."""
    # Create output directory
    output_dir = Path(__file__).parent.parent / "data" / "translation"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    language_pairs = get_available_language_pairs()
    print(f"🎯 Found {len(language_pairs)} language pairs:")
    for i, pair in enumerate(language_pairs, 1):
        display_name = TranslationTask._get_display_name(pair)
        print(f"  {i:2d}. {display_name} ({pair})")
    
    generated_files = []
    
    for language_pair in language_pairs:
        display_name = TranslationTask._get_display_name(language_pair)
        print(f"\n🌍 Processing language pair: {display_name}")
        print("=" * 50)
        
        try:
            task = TranslationTask(
                language_pair=language_pair,
                variations_per_field=variations_per_field,
                api_platform=api_platform,
                model_name=model_name,
                max_rows=max_rows,
                max_variations_per_row=max_variations_per_row,
                random_seed=random_seed
            )
            
            # Override output path to save in translation folder
            output_file = output_dir / f"translation_{language_pair}_variations.json"
            
            # Generate using the base class method but with custom output path
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

            generated_files.append(str(output_file))
            print(f"✅ Completed {display_name}: {output_file}")
            
        except Exception as e:
            print(f"❌ Error processing {display_name}: {e}")
            continue
    
    print(f"\n🎉 All language pairs completed! Generated {len(generated_files)} files:")
    for file in generated_files:
        print(f"  📄 {file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate translation prompt variations")
    parser.add_argument("--language_pair", help="Specific language pair to process (e.g., cs-en, de-en)", default=None)
    parser.add_argument("--all", action="store_true", help="Process all language pairs", default=True)
    parser.add_argument("--rows", type=int, help="Number of rows to process", default=DEFAULT_MAX_ROWS)
    parser.add_argument("--variations", type=int, help="Number of variations per row", default=DEFAULT_MAX_VARIATIONS_PER_ROW)
    parser.add_argument("--variations_per_field", type=int, default=DEFAULT_VARIATIONS_PER_FIELD)
    parser.add_argument("--api_platform", type=str, default=DEFAULT_PLATFORM)
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--random_seed", type=int, help="Random seed for generation", default=DEFAULT_RANDOM_SEED)
    args = parser.parse_args()
    
    if args.all:
        generate_all_language_pairs(
            variations_per_field=args.variations_per_field,
            api_platform=args.api_platform,
            model_name=args.model_name,
            max_rows=args.rows,
            max_variations_per_row=args.variations,
            random_seed=args.random_seed
        )
    elif args.language_pair:
        task = TranslationTask(
            language_pair=args.language_pair,
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
    else:
        print("Please specify either --language_pair <pair> or --all")
        print("\nAvailable language pairs:")
        try:
            pairs = get_available_language_pairs()
            for pair in pairs:
                display_name = TranslationTask._get_display_name(pair)
                print(f"  - {display_name} ({pair})")
        except Exception as e:
            print(f"Error getting language pairs: {e}") 