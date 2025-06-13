"""
Command-line interface for MultiPromptify.
"""

import json
import os
import sys
from typing import Optional

import click

from src.multipromptify import MultiPromptify


@click.command()
@click.option(
    '--template', '-t',
    required=True,
    help='Template string with variation annotations (e.g., "{instruction:semantic}: {question:paraphrase}")'
)
@click.option(
    '--data', '-d',
    required=True,
    help='Input data file path (CSV, JSON) or directory path'
)
@click.option(
    '--instruction', '-i',
    help='Static instruction text to use across all prompts'
)
@click.option(
    '--few-shot',
    help='Few-shot examples as JSON string (e.g., \'["Example 1", "Example 2"]\')'
)
@click.option(
    '--few-shot-file',
    help='Path to file containing few-shot examples (one per line or JSON)'
)
@click.option(
    '--few-shot-count',
    type=int,
    default=3,
    help='Number of few-shot examples to use per prompt (default: 3)'
)
@click.option(
    '--output', '-o',
    help='Output file path (default: stdout)'
)
@click.option(
    '--output-format',
    type=click.Choice(['json', 'csv', 'hf'], case_sensitive=False),
    default='json',
    help='Output format (default: json)'
)
@click.option(
    '--max-variations',
    type=int,
    default=100,
    help='Maximum number of variations to generate (default: 100)'
)
@click.option(
    '--variations-per-field',
    type=int,
    default=3,
    help='Number of variations to generate per field (default: 3)'
)
@click.option(
    '--seed',
    type=int,
    help='Random seed for reproducible results'
)
@click.option(
    '--verbose', '-v',
    is_flag=True,
    help='Enable verbose output'
)
@click.option(
    '--dry-run',
    is_flag=True,
    help='Show what would be generated without actually generating'
)
@click.option(
    '--validate-only',
    is_flag=True,
    help='Only validate the template and data, don\'t generate variations'
)
@click.option(
    '--show-stats',
    is_flag=True,
    help='Show statistics about generated variations'
)
def main(
        template: str,
        data: str,
        instruction: Optional[str],
        few_shot: Optional[str],
        few_shot_file: Optional[str],
        few_shot_count: int,
        output: Optional[str],
        output_format: str,
        max_variations: int,
        variations_per_field: int,
        seed: Optional[int],
        verbose: bool,
        dry_run: bool,
        validate_only: bool,
        show_stats: bool
):
    """
    MultiPromptify - Generate multi-prompt datasets from single-prompt datasets.
    
    Examples:
    
    Basic usage:
    multipromptify --template "{instruction:semantic}: {question:paraphrase}" --data data.csv --instruction "Answer this"
    
    With few-shot examples:
    multipromptify --template "{instruction}: {few_shot}\\n{question:paraphrase}" --data data.csv --instruction "Answer" --few-shot '["Q: 1+1? A: 2"]'
    
    Save to file:
    multipromptify --template "{instruction:semantic}: {question}" --data data.csv --instruction "Solve" --output variations.json
    """
    try:
        # Set random seed if provided
        if seed is not None:
            import random
            random.seed(seed)

        # Initialize MultiPromptify
        mp = MultiPromptify(max_variations=max_variations)

        if verbose:
            click.echo(f"Initialized MultiPromptify with max_variations={max_variations}")

        # Validate template
        is_valid, errors = mp.template_parser.validate_template(template)
        if not is_valid:
            click.echo("Template validation failed:", err=True)
            for error in errors:
                click.echo(f"  - {error}", err=True)
            sys.exit(1)

        if verbose:
            click.echo("Template validated successfully")
            parsed_fields = mp.parse_template(template)
            click.echo(f"Template fields: {parsed_fields}")

        # Check if data file exists
        if not os.path.exists(data):
            click.echo(f"Error: Data file '{data}' not found", err=True)
            sys.exit(1)

        if validate_only:
            click.echo("✓ Template is valid")
            click.echo(f"✓ Data file '{data}' exists")
            required_columns = mp.template_parser.get_required_columns()
            if required_columns:
                click.echo(f"Required columns: {', '.join(required_columns)}")
            sys.exit(0)

        # Process few-shot examples
        few_shot_examples = None
        if few_shot:
            try:
                few_shot_examples = json.loads(few_shot)
            except json.JSONDecodeError:
                click.echo("Error: Invalid JSON format for --few-shot", err=True)
                sys.exit(1)
        elif few_shot_file:
            if not os.path.exists(few_shot_file):
                click.echo(f"Error: Few-shot file '{few_shot_file}' not found", err=True)
                sys.exit(1)

            with open(few_shot_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                try:
                    # Try to parse as JSON first
                    few_shot_examples = json.loads(content)
                except json.JSONDecodeError:
                    # If not JSON, treat as line-separated examples
                    few_shot_examples = [line.strip() for line in content.split('\n') if line.strip()]

        if verbose and few_shot_examples:
            click.echo(f"Few-shot examples loaded: {len(few_shot_examples)} examples")

        # Dry run mode
        if dry_run:
            click.echo("=== DRY RUN MODE ===")
            click.echo(f"Template: {template}")
            click.echo(f"Data file: {data}")
            click.echo(f"Instruction: {instruction}")
            click.echo(f"Few-shot examples: {few_shot_examples}")
            click.echo(f"Max variations: {max_variations}")
            click.echo("Would generate variations with these settings.")
            sys.exit(0)

        # Generate variations
        if verbose:
            click.echo("Generating variations...")

        try:
            variations = mp.generate_variations(
                template=template,
                data=data,
                instruction=instruction,
                few_shot=few_shot_examples
            )
        except Exception as e:
            click.echo(f"Error generating variations: {str(e)}", err=True)
            if verbose:
                import traceback
                traceback.print_exc()
            sys.exit(1)

        if verbose:
            click.echo(f"Generated {len(variations)} variations")

        # Show statistics if requested
        if show_stats or verbose:
            stats = mp.get_stats(variations)
            click.echo("\n=== STATISTICS ===")
            for key, value in stats.items():
                click.echo(f"{key.replace('_', ' ').title()}: {value}")

        # Output results
        if output:
            # Save to file
            try:
                mp.save_variations(variations, output, format=output_format)
                click.echo(f"Variations saved to {output} ({output_format} format)")
            except Exception as e:
                click.echo(f"Error saving variations: {str(e)}", err=True)
                sys.exit(1)
        else:
            # Output to stdout
            if output_format.lower() == 'json':
                click.echo(json.dumps(variations, indent=2, ensure_ascii=False))
            elif output_format.lower() == 'csv':
                # Simple CSV output to stdout
                import pandas as pd
                flattened = []
                for var in variations:
                    flat_var = {
                        'prompt': var['prompt'],
                        'original_row_index': var.get('original_row_index', ''),
                    }
                    # Add field values
                    for key, value in var.get('field_values', {}).items():
                        flat_var[f'field_{key}'] = value
                    flattened.append(flat_var)

                df = pd.DataFrame(flattened)
                click.echo(df.to_csv(index=False))
            else:
                # Default to showing just the prompts
                for i, var in enumerate(variations, 1):
                    click.echo(f"--- Variation {i} ---")
                    click.echo(var['prompt'])
                    click.echo()

    except KeyboardInterrupt:
        click.echo("\nOperation cancelled by user", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Unexpected error: {str(e)}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


# Additional utility commands
@click.group()
def cli():
    """MultiPromptify CLI utilities."""
    pass


@cli.command()
@click.argument('template')
def validate_template(template: str):
    """Validate a template string."""
    mp = MultiPromptify()
    is_valid, errors = mp.template_parser.validate_template(template)

    if is_valid:
        click.echo("✓ Template is valid")
        parsed_fields = mp.parse_template(template)
        if parsed_fields:
            click.echo("Fields:")
            for field, variation_type in parsed_fields.items():
                if variation_type:
                    click.echo(f"  - {field} ({variation_type})")
                else:
                    click.echo(f"  - {field} (no variation)")
    else:
        click.echo("✗ Template is invalid:")
        for error in errors:
            click.echo(f"  - {error}")


@cli.command()
@click.argument('data_file')
def inspect_data(data_file: str):
    """Inspect a data file to see its structure."""
    if not os.path.exists(data_file):
        click.echo(f"Error: File '{data_file}' not found", err=True)
        return

    try:
        mp = MultiPromptify()
        df = mp._load_data(data_file)

        click.echo(f"Data file: {data_file}")
        click.echo(f"Rows: {len(df)}")
        click.echo(f"Columns: {len(df.columns)}")
        click.echo(f"Column names: {', '.join(df.columns)}")

        # Show sample data
        click.echo("\nFirst 3 rows:")
        click.echo(df.head(3).to_string())

    except Exception as e:
        click.echo(f"Error loading data file: {str(e)}", err=True)


if __name__ == '__main__':
    main()
