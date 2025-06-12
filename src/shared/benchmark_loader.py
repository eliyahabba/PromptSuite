import json
import logging
import sys
from typing import Optional, List
import click
from datasets import load_dataset
import os
from dotenv import load_dotenv, dotenv_values
import pprint
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("Hugging Face token is not set in the .env file.")

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

BENCHMARK_DATASETS = {
    "ifeval": "google/IFEval",
    "gpqa": "Idavidrein/gpqa",
    "bigbenchhard": "maveriq/bigbenchhard",
    "math_dataset": "deepmind/math_dataset",
    "musr": "TAUR-Lab/MuSR",
}

@click.command()
@click.argument('benchmark_name', type=click.Choice(BENCHMARK_DATASETS.keys(), case_sensitive=False))
@click.option('--num-examples', '-n', type=int, default=5, help='Number of examples to display (default: 5)')
@click.option('--save-path', '-o', type=str, help='Path to save examples as JSON (optional)')
@click.option('--split', '-s', type=str, default="train", help='Split to choose')
@click.option('--config', '-c', type=str, help='Configuration name for the dataset (if required)')
@click.option('--columns', '-l', multiple=True, type=str, help='List of columns to extract (e.g., -l question -l answer)')
def load_benchmark(benchmark_name: str, num_examples: int, save_path: Optional[str], split: str, config: Optional[str], columns: Optional[List[str]]):
    """
    Loads a benchmark dataset from Hugging Face Datasets and optionally displays/saves specific columns from examples.
    Handles datasets with required configurations.
    """
    dataset_identifier = BENCHMARK_DATASETS[benchmark_name]
    logger.info(f"Loading dataset: {dataset_identifier}, split: {split}, config: {config}, columns: {columns}")
    try:
        if config:
            dataset = load_dataset(dataset_identifier, name=config, split=split)
        else:
            dataset = load_dataset(dataset_identifier, split=split)
        logger.info(f"Dataset loaded successfully. Found {len(dataset)} examples in the '{split}' split.")

        if dataset:
            logger.info(f"\n--- First {min(num_examples, len(dataset))} Examples (Extracted Columns) ---")
            for i in range(min(num_examples, len(dataset))):
                example = dataset[i]
                if columns:
                    extracted_data = {col: example.get(col) for col in columns}
                    logger.info(f"Example {i + 1}: {extracted_data}")
                else:
                    logger.info(f"Example {i + 1}: {pprint.pformat(example)}")

            if save_path:
                logger.info(f"\nSaving first {min(100, len(dataset))} examples (Extracted Columns) to: {save_path}")
                examples_to_save = []
                for i in range(min(100, len(dataset))):
                    example = dataset[i]
                    if columns:
                        extracted_data = {col: example.get(col) for col in columns}
                        examples_to_save.append(extracted_data)
                    else:
                        examples_to_save.append(example)
                with open(save_path, 'w') as f:
                    json.dump(examples_to_save, f, indent=4)
                logger.info("Examples saved.")
        else:
            logger.warning(f"The split '{split}' is empty or not found in the dataset.")

    except Exception as e:
        logger.error(f"An error occurred while loading the dataset: {e}")

if __name__ == "__main__":
    load_benchmark()
