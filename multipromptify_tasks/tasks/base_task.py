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
from multipromptify_tasks.constants import (
    VARIATIONS_PER_ROW, MAX_ROWS_PER_DATASET, 
    DEFAULT_PLATFORM, MODELS
)


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

    def get_variations_per_field(self) -> int:
        """
        Get the number of variations per field.
        Child classes can override this to provide custom values.
        
        Returns:
            Number of variations per field (default: 4)
        """
        return 4

    def get_api_platform(self) -> str:
        """
        Get the API platform to use.
        Child classes can override this to provide custom values.
        
        Returns:
            API platform name (default: "TogetherAI")
        """
        return DEFAULT_PLATFORM

    def get_model_name(self) -> str:
        """
        Get the model name to use.
        Child classes can override this to provide custom values.
        
        Returns:
            Model name (default: "default" model for the platform)
        """
        platform = self.get_api_platform()
        return MODELS[platform]["default"]

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
        variations_per_field = self.get_variations_per_field()
        api_platform = self.get_api_platform()
        model_name = self.get_model_name()
        
        # Configure generation parameters
        print(f"\n3. Configuring generation ({variations_per_row} variations per row, {max_rows} rows)...")
        print(f"   Variations per field: {variations_per_field}")
        print(f"   API Platform: {api_platform}")
        print(f"   Model: {model_name}")
        
        self.mp.configure(
            max_rows=max_rows,
            variations_per_field=variations_per_field,
            max_variations_per_row=variations_per_row,
            random_seed=42,
            api_platform=api_platform,
            model_name=model_name
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
