# UI Components

This directory contains reusable UI components for the MultiPromptify application.

## Components

### `results_display.py`

Contains all functionality for displaying generated prompt variations in a consistent way across the application.

#### Main Functions:

- `display_full_results()` - Main function that displays complete results with summary, variations, and optional export
- `display_enhanced_summary_metrics()` - Shows generation statistics with styled cards
- `display_enhanced_variations()` - Shows paginated variations with highlighting
- `display_single_variation()` - Shows individual variation with field comparison
- `highlight_prompt_fields()` - Highlights field values within generated prompts
- `export_interface()` - Provides download options in multiple formats
- `display_simple_download_options()` - Simplified export for generation page

#### Usage:

```python
from src.ui.components.results_display import display_full_results

# Display complete results
display_full_results(
    variations=variations,
    original_data=original_data,
    stats=stats,
    generation_time=generation_time,
    show_export=True,
    show_header=True
)
```

## Benefits

- **Code Reusability**: Shared display logic between pages
- **Consistency**: Same look and feel across the application
- **Maintainability**: Single source of truth for display functions
- **Modularity**: Easy to extend and modify individual components 