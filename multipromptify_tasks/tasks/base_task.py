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
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from multipromptify import MultiPromptifier
from multipromptify_tasks.constants import VARIATIONS_PER_ROW, MAX_ROWS_PER_DATASET


class BaseTask(ABC):
    """Base class for all MultiPromptify tasks."""

    def __init__(self, task_name: str, output_filename: str):
        """
        Initialize the base task.
        
        Args:
            task_name: Name of the task for display
            output_filename: Name of the output file
        """
        self.task_name = task_name
        self.output_filename = output_filename
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
            rows: Number of rows to process (overrides MAX_ROWS_PER_DATASET)
            variations: Number of variations per row (overrides VARIATIONS_PER_ROW)
        """
        self._override_rows = rows
        self._override_variations = variations
        if rows is not None:
            print(f"   Overriding rows: {rows} (default: {MAX_ROWS_PER_DATASET})")
        if variations is not None:
            print(f"   Overriding variations: {variations} (default: {VARIATIONS_PER_ROW})")

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

        # Get configuration values (use overrides if provided)
        max_rows = getattr(self, '_override_rows', MAX_ROWS_PER_DATASET)
        variations_per_row = getattr(self, '_override_variations', VARIATIONS_PER_ROW)
        
        # Configure generation parameters
        print(f"\n3. Configuring generation ({variations_per_row} variations per row, {max_rows} rows)...")
        self.mp.configure(
            max_rows=max_rows,
            variations_per_field=4,
            max_variations_per_row=variations_per_row,
            random_seed=42,
            api_platform="TogetherAI",
            model_name="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
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
        output_file = Path(__file__).parent.parent / "data" / self.output_filename
        print(f"\n6. Exporting results to {output_file}...")
        self.mp.export(str(output_file), format="json")
        print("✅ Export completed!")

        # Show final statistics
        print("\n7. Final statistics:")
        self.mp.info()

        return output_file
