#!/usr/bin/env python3
"""
Base Task Class
This module provides a base class for all MultiPromptify tasks.
"""

import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any

# Add the project root to the path to import multipromptify
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Add current directory to path for local imports
current_dir = Path(__file__).parent.parent
sys.path.insert(0, str(current_dir))

from multipromptify import MultiPromptifier
from multipromptify_tasks.constants import (
    DEFAULT_VARIATIONS_PER_FIELD, DEFAULT_PLATFORM, DEFAULT_MODEL_NAME,
    DEFAULT_MAX_VARIATIONS_PER_ROW, DEFAULT_MAX_ROWS, DEFAULT_RANDOM_SEED
)


class BaseTask(ABC):
    """
    Base class for all MultiPromptify tasks.
    Allows configuration of variations_per_field, api_platform, model_name, max_rows, max_variations_per_row, random_seed via __init__ arguments.
    """

    def __init__(self, task_name: str, output_filename: str,
                 variations_per_field: int = DEFAULT_VARIATIONS_PER_FIELD,
                 api_platform: str = DEFAULT_PLATFORM,
                 model_name: str = DEFAULT_MODEL_NAME,
                 max_rows: int = DEFAULT_MAX_ROWS,
                 max_variations_per_row: int = DEFAULT_MAX_VARIATIONS_PER_ROW,
                 random_seed: int = DEFAULT_RANDOM_SEED):
        """
        Initialize the base task.
        Args:
            task_name: Name of the task for display
            output_filename: Name of the output file
            variations_per_field: Number of variations per field (default: from constants)
            api_platform: API platform to use (default: from constants)
            model_name: Model name to use (default: from constants)
            max_rows: Max rows to process (default: from constants)
            max_variations_per_row: Max variations per row (default: from constants)
            random_seed: Random seed (default: from constants)
        """
        self.task_name = task_name
        self.output_filename = output_filename
        self.variations_per_field = variations_per_field
        self.api_platform = api_platform
        self.model_name = model_name
        self.max_rows = max_rows
        self.max_variations_per_row = max_variations_per_row
        self.random_seed = random_seed
        self.mp = MultiPromptifier()

    @abstractmethod
    def load_data(self) -> None:
        """Load the dataset for this task."""
        pass

    @abstractmethod
    def get_template(self) -> Dict[str, Any]:
        """Get the template configuration for this task."""
        pass

    def override_config(self, rows: int = None, variations: int = None) -> None:
        """
        Override the default configuration with command line arguments.
        Args:
            rows: Number of rows to process (overrides max_rows)
            variations: Number of variations per row (overrides max_variations_per_row)
        """
        if rows is not None:
            self.max_rows = rows
            print(f"   Overriding rows: {rows} (default: {DEFAULT_MAX_ROWS})")
        if variations is not None:
            self.max_variations_per_row = variations
            print(f"   Overriding variations: {variations} (default: {DEFAULT_MAX_VARIATIONS_PER_ROW})")

    def generate(self) -> str:
        """
        Generate variations for this task.
        Returns:
            Path to the output file
        """
        print(f"🚀 Starting {self.task_name}")
        print("=" * 60)

        # Load data
        print("\n1. Loading data...")
        self.load_data()

        # Configure template
        print("\n2. Setting up template...")
        template = self.get_template()
        self.mp.set_template(template)
        print("✅ Template configured")

        # Configure generation parameters
        print(
            f"\n3. Configuring generation ({self.max_variations_per_row} variations per row, {self.max_rows} rows)...")
        print(f"   Variations per field: {self.variations_per_field}")
        print(f"   API Platform: {self.api_platform}")
        print(f"   Model: {self.model_name}")
        print(f"   Random seed: {self.random_seed}")

        self.mp.configure(
            max_rows=self.max_rows,
            variations_per_field=self.variations_per_field,
            max_variations_per_row=self.max_variations_per_row,
            random_seed=self.random_seed,
            api_platform=self.api_platform,
            model_name=self.model_name
        )

        # Generate variations
        print("\n4. Generating prompt variations...")
        variations = self.mp.generate(verbose=True)

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

        # Export results
        output_file = Path(__file__).parent.parent / "tasks_data" / "generated_data" / self.output_filename
        print(f"\n6. Exporting results to {output_file}...")
        self.mp.export(str(output_file), format="json")
        print("✅ Export completed!")

        # Show final statistics
        print("\n7. Final statistics:")
        self.mp.info()

        return output_file
