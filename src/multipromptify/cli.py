"""
Command-line interface for MultiPromptify.
"""

import json
from pathlib import Path

import click
import pandas as pd

from multipromptify import __version__
from multipromptify.core.engine import MultiPromptify
from multipromptify.shared.constants import GenerationDefaults


@click.command()
@click.option('--template', '-t', required=True, help='Template dictionary as JSON string or file path')
@click.option('--data', '-d', required=True, help='Input data file (CSV or JSON)')
@click.option('--output', '-o', default='variations.json', help='Output file path')
@click.option('--format', '-f', type=click.Choice(['json', 'csv', 'txt']), default='json', help='Output format')
@click.option('--max-variations', '-m', default=100, help='Maximum number of variations per row (use 0 for unlimited)')
@click.option('--variations-per-field', '-v', default=GenerationDefaults.VARIATIONS_PER_FIELD,
              help='Number of variations per field')
@click.option('--api-key', '-k', envvar='TOGETHER_API_KEY', help='API key for paraphrase generation')
@click.version_option(version=__version__)
def main(template, data, output, format, max_variations_per_row, variations_per_field, api_key):
    """MultiPromptify - Generate prompt variations from templates."""

    click.echo(f"MultiPromptify v{__version__}")
    click.echo("=" * 40)

    try:
        # Load template
        if template.startswith('{'):
            # Direct JSON string
            template_dict = json.loads(template)
        elif Path(template).exists():
            # File path
            with open(template, 'r') as f:
                template_dict = json.load(f)
        else:
            raise click.BadParameter("Template must be a JSON string or valid file path")

        # Load data
        if data.endswith('.csv'):
            df = pd.read_csv(data)
        elif data.endswith('.json'):
            with open(data, 'r') as f:
                json_data = json.load(f)
            df = pd.DataFrame(json_data)
        else:
            raise click.BadParameter("Data file must be CSV or JSON")

        click.echo(f"Loaded {len(df)} rows from {data}")

        # Convert max_variations_per_row: 0 means unlimited (None)
        effective_max_variations_per_row = None if max_variations_per_row == 0 else max_variations_per_row

        # Initialize MultiPromptify
        mp = MultiPromptify(max_variations_per_row=effective_max_variations_per_row)

        # Generate variations
        click.echo("Generating variations...")
        variations = mp.generate_variations(
            template=template_dict,
            data=df,
            variations_per_field=variations_per_field,
            api_key=api_key
        )

        click.echo(f"Generated {len(variations)} variations")

        # Save output
        mp.save_variations(variations, output, format=format)
        click.echo(f"Saved to {output}")

        # Show statistics
        stats = mp.get_stats(variations)
        click.echo("\nStatistics:")
        for key, value in stats.items():
            click.echo(f"  {key}: {value}")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()


if __name__ == '__main__':
    main()
