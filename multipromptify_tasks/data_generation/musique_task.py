#!/usr/bin/env python3
"""
MuSiQue Task: Multi-hop Question Answering
This module provides a class for generating prompt variations for MuSiQue tasks.
"""

from typing import Dict, Any, List
import argparse
import pandas as pd
from pathlib import Path
import sys
import random

from datasets import load_dataset
from multipromptify.core.template_keys import (
    INSTRUCTION, PROMPT_FORMAT, GOLD_KEY,
    PARAPHRASE_WITH_LLM, FORMAT_STRUCTURE_VARIATION, TYPOS_AND_NOISE_VARIATION,
    INSTRUCTION_VARIATIONS, PROMPT_FORMAT_VARIATIONS, FEW_SHOT_KEY,
    SHUFFLE_VARIATION, ENUMERATE_VARIATION
)
from base_task import BaseTask
from multipromptify_tasks.constants import (
    DEFAULT_VARIATIONS_PER_FIELD, DEFAULT_PLATFORM, DEFAULT_MODEL_NAME,
    DEFAULT_MAX_VARIATIONS_PER_ROW, DEFAULT_MAX_ROWS, DEFAULT_RANDOM_SEED
)


class MuSiQueTask(BaseTask):
    """Task for generating MuSiQue multi-hop question answering prompt variations."""

    def __init__(self,
                 variations_per_field: int = DEFAULT_VARIATIONS_PER_FIELD,
                 api_platform: str = DEFAULT_PLATFORM,
                 model_name: str = DEFAULT_MODEL_NAME,
                 max_rows: int = DEFAULT_MAX_ROWS,
                 max_variations_per_row: int = DEFAULT_MAX_VARIATIONS_PER_ROW,
                 random_seed: int = DEFAULT_RANDOM_SEED):

        task_name = "MuSiQue Task: Multi-hop Question Answering"
        output_filename = "musique_variations.json"

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

    def format_paragraphs(self, paragraphs_list: List[Dict[str, Any]]) -> str:
        """
        Format paragraphs list into a numbered string format.
        
        Args:
            paragraphs_list: List of paragraph dictionaries with 'idx', 'title', 'paragraph_text'
            
        Returns:
            Formatted string with numbered paragraphs
        """
        if not paragraphs_list:
            return ""
        
        formatted_paragraphs = []
        for i, para in enumerate(paragraphs_list, 1):
            title = para.get('title', f'Paragraph {i}')
            text = para.get('paragraph_text', '')
            formatted_paragraphs.append(f"{i}. {title}: {text}")
        
        return "\n\n".join(formatted_paragraphs)

    def load_data(self) -> None:
        """Load MuSiQue dataset from HuggingFace."""
        print("Loading MuSiQue dataset...")
        
        try:
            # Load train split for few-shot examples and validation split for evaluation
            train_ds = load_dataset("dgslibisey/MuSiQue", split="train[:100]")
            val_ds = load_dataset("dgslibisey/MuSiQue", split="validation[:50]")
            
            # Convert to DataFrames
            train_df = pd.DataFrame(train_ds)
            val_df = pd.DataFrame(val_ds)
            
            # Add split information
            train_df['split'] = 'train'
            val_df['split'] = 'test'  # Use 'test' for consistency with other tasks
            
            # Combine datasets
            df = pd.concat([train_df, val_df], ignore_index=True)
            
            # Format paragraphs into a structured string format
            print("📝 Formatting paragraphs...")
            df['paragraphs_formatted'] = df['paragraphs'].apply(self.format_paragraphs)
            
            # Create a list of paragraph titles for variations (similar to MMLU choices)
            df['paragraph_titles'] = df['paragraphs'].apply(
                lambda paras: [para.get('title', f'Paragraph {i+1}') 
                              for i, para in enumerate(paras)]
            )
            
            print(f"✅ Loaded MuSiQue dataset: {len(train_df)} train + {len(val_df)} validation = {len(df)} total rows")
            print(f"📊 Sample formatted paragraphs preview:")
            if len(df) > 0:
                sample_paras = df['paragraphs_formatted'].iloc[0][:300] + "..." if len(df['paragraphs_formatted'].iloc[0]) > 300 else df['paragraphs_formatted'].iloc[0]
                print(f"   {sample_paras}")
            
            # Load into multipromptify
            self.mp.load_dataframe(df)
            
        except Exception as e:
            print(f"❌ Error loading MuSiQue dataset: {e}")
            raise ValueError(f"Failed to load MuSiQue dataset: {e}")

    def get_template(self) -> Dict[str, Any]:
        """Get template configuration for MuSiQue multi-hop question answering task."""
        return {
            INSTRUCTION: "In this task, you are presented with question, and 20 documents that covers the answer to that question. Deduce your answer solely from the provided documents, avoiding any external data sources. Keep the answer short and concise, leave behind any irrelevant details",
            INSTRUCTION_VARIATIONS: [PARAPHRASE_WITH_LLM],
            PROMPT_FORMAT: "Context:\n{paragraphs_formatted}\n\nQuestion: {question}\nAnswer: {answer}",
            PROMPT_FORMAT_VARIATIONS: [FORMAT_STRUCTURE_VARIATION],
            'question': [FORMAT_STRUCTURE_VARIATION],
            'paragraph_titles': [ENUMERATE_VARIATION],  # For title-only variations (no shuffle since gold is not an index)
            GOLD_KEY: 'answer',  # Simple format for text answers
            FEW_SHOT_KEY: {
                'count': 3,  # Use 3 few-shot examples
                'format': 'random_per_row',
                'split': 'train'
            }
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate MuSiQue prompt variations")
    parser.add_argument("--rows", type=int, help="Number of rows to process", default=DEFAULT_MAX_ROWS)
    parser.add_argument("--variations", type=int, help="Number of variations per row", default=DEFAULT_MAX_VARIATIONS_PER_ROW)
    parser.add_argument("--variations_per_field", type=int, default=DEFAULT_VARIATIONS_PER_FIELD)
    parser.add_argument("--api_platform", type=str, default=DEFAULT_PLATFORM)
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--random_seed", type=int, help="Random seed for generation", default=DEFAULT_RANDOM_SEED)
    args = parser.parse_args()
    
    task = MuSiQueTask(
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