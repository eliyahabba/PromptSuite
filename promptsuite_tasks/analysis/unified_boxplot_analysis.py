#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
import json
warnings.filterwarnings('ignore')

# Set font to Times New Roman for academic papers
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']  # Common in academic papers
plt.rcParams['mathtext.fontset'] = 'dejavuserif'

# ====== VISUALIZATION PARAMETERS (Magic Numbers) ======
# 
# Graph appearance customization - these values can be easily changed:
#
# BOX_WIDTH: Width of each box (0.1-1.0)
#   - 0.8 = relatively wide boxes
#   - 0.5 = narrower boxes
#   - 0.3 = very narrow boxes
#
# BOX_SPACING: Spacing between boxes (position multiplier)
#   - 1.0 = normal spacing
#   - 0.8 = boxes closer together (less spacing)
#   - 1.2 = boxes farther apart (more spacing)
#
# FIGURE_WIDTH_PER_BOX: Graph width per dataset
#   - 1.5 = relatively narrow graph
#   - 2.0 = wider graph
#
BOX_WIDTH = 0.5          # Width of each box (0.1-1.0, smaller = narrower boxes)
BOX_SPACING = 0.68        # Spacing between boxes (positions multiplier, smaller = closer together)
FIGURE_WIDTH_PER_BOX = 1.5  # Figure width multiplier per dataset
MIN_FIGURE_WIDTH = 12    # Minimum figure width
FIGURE_HEIGHT = 8        # Figure height

# Colors and transparency
BOX_ALPHA = 0.7         # Box transparency (0.0-1.0)
MEDIAN_LINE_WIDTH = 2.5 # Thickness of median line in box plots (1.0 = thin, 3.0 = thick)

# Display options
SHOW_INFO_BOX = False    # Whether to show the gray box with question count details
SHOW_MODEL_AS_TITLE = False  # Whether to show model name as title (False = show below graph)

# Font sizes
TITLE_FONT_SIZE = 16
AXIS_LABEL_FONT_SIZE = 16      # Font size for axis titles (like "Metric (%)")
TICK_LABEL_FONT_SIZE = 14      # Font size for task names on X-axis
Y_TICK_LABEL_FONT_SIZE = 15    # Font size for numbers on Y-axis (0, 20, 40, 60...)
INFO_TEXT_FONT_SIZE = 11
MODEL_NAME_FONT_SIZE = 18      # Font size for model name when shown below graph

# Dataset-metric pairs explicitly defined
DATASET_METRICS = {
    'mmlu': 'is_correct',
    'math': 'is_correct',
    'code_generation': 'pass@1',  # Use pass@1 from evaluation JSON files
    'sentiment': 'mae',  # Mean Absolute Error
    'translation': 'bleu',
    'summarization': 'rouge1',
    'musique': 'word_f1',
    'qa': 'word_f1',  # Use word_f1 from F1 metrics files
    'gpqa': 'extracted_is_correct'
}

# Models to analyze
MODELS = ['gpt_4o_mini', 'llama_3_3_70b']

# Model display names mapping
MODEL_DISPLAY_NAMES = {
    'gpt_4o_mini': 'GPT-4o mini',
    'llama_3_3_70b': 'Llama-3.3-70B'
}

# Dataset display names mapping (dataset_name, task_description)
# First tuple element: dataset name (will be displayed in bold)
# Second tuple element: task description (will be displayed in regular font)
DATASET_DISPLAY_NAMES = {
    'mmlu': ('MMLU', 'Multiple\nChoice'),
    'translation': ('WMT14', 'Translation'), 
    'sentiment': ('SST', 'Sentiment\nAnalysis'),
    'math': ('GSM8K', 'Open\nMath\nProblems'),
    'gpqa': ('GPQA-\nDiamond', 'Google-Proof\nMath'),
    'code_generation': ('HumanEval', 'Code\nGeneration'),
    'musique': ('MuSiQue', 'Multihop\nQuestions'),
    'summarization': ('CNN-\nDaily-Mail', 'Summarization'),
    'qa': ('SQuAD', 'Reading\nComprehension')
}

# Metrics that need to be scaled to 0-100 (multiply by 100)
# This includes:
# 1. Boolean metrics (is_correct) - originally 0-1 when averaged, multiply by 100 for percentage
# 2. Ratio metrics (bleu, rouge) - originally 0-1, multiply by 100 for percentage
# 3. Pass@k metrics - originally 0-1, multiply by 100 for percentage
SCALE_TO_100_METRICS = {
    # Boolean accuracy metrics (0-1 -> 0-100%)
    'is_correct': 'percentage',
    'extracted_is_correct': 'percentage',  # For GPQA, assuming it's a boolean metric
    # Text similarity metrics (0-1 -> 0-100)
    'bleu': 'percentage',
    'rouge1': 'percentage',
    'rouge2': 'percentage',
    'rougeL': 'percentage',
    'word_f1': 'percentage',  # Word F1 score (0-1 -> 0-100%)
    # Code generation metrics (0-1 -> 0-100%)
    'pass@1': 'percentage',
    'pass@5': 'percentage',
    'pass@10': 'percentage',
    'pass@20': 'percentage',
    # Error metrics that we want to convert to percentage scale
    'mae': 'percentage',  # Convert MAE to percentage scale for consistency
}

# Metrics that should stay in their original scale
KEEP_ORIGINAL_SCALE_METRICS = {
    'sacrebleu': 'score'  # SacreBleu is already 0-100 scale
}

def get_metric_scaling_info(metric_name: str) -> Tuple[bool, str]:
    """
    Get scaling information for a metric.
    
    Returns:
        Tuple of (should_scale_to_100, metric_type)
    """
    if metric_name in SCALE_TO_100_METRICS:
        return True, SCALE_TO_100_METRICS[metric_name]
    elif metric_name in KEEP_ORIGINAL_SCALE_METRICS:
        return False, KEEP_ORIGINAL_SCALE_METRICS[metric_name]
    else:
        # Default: assume it needs scaling
        print(f"Warning: Unknown metric '{metric_name}', assuming it needs 0-100 scaling")
        return True, 'unknown'

def extract_variation_data_from_sample_key(sample_key: str) -> tuple:
    """Extract row_index and variation_index from sample key like '0_1'."""
    parts = sample_key.split('_')
    if len(parts) >= 2:
        return int(parts[0]), int(parts[1])
    return 0, 0

def load_code_generation_evaluation_data(model_dir: Path, metric_name: str) -> Optional[pd.DataFrame]:
    """
    Load code generation evaluation data from JSON files with pass@k metrics.
    """
    evaluation_files = list(model_dir.glob("*_evaluation.json"))

    if not evaluation_files:
        return None

    all_data = []

    for eval_file in evaluation_files:
        try:
            with open(eval_file, 'r', encoding='utf-8') as f:
                eval_results = json.load(f)

            sample_results = eval_results.get('sample_results', {})

            rows = []
            for sample_key, metrics in sample_results.items():
                row_idx, var_idx = extract_variation_data_from_sample_key(sample_key)

                # Check if the metric exists
                if metric_name not in metrics:
                    continue

                row_data = {
                    'sample_key': sample_key,
                    'original_row_index': row_idx,
                    'variation_index': var_idx,
                    'task_identifier': eval_file.stem.replace('_evaluation', ''),
                    metric_name: metrics[metric_name]
                }

                rows.append(row_data)

            if rows:
                df = pd.DataFrame(rows)
                all_data.append(df)

        except Exception as e:
            print(f"Error reading {eval_file.name}: {e}")
            continue

    if not all_data:
        return None

    return pd.concat(all_data, ignore_index=True)

def analyze_dataset_variations(model_dir: Path, dataset_name: str, metric_name: str) -> Optional[Tuple[pd.DataFrame, Dict]]:
    """
    Analyze a single dataset variations using the same logic as individual analysis files.
    This function combines multiple CSV files for a dataset (like MMLU subjects or translation pairs).
    For code generation, it loads from evaluation JSON files.
    
    Returns:
        Tuple of (variation_scores_df, stats_dict) or None if no data found
    """
    # Special handling for code generation with evaluation JSON files
    if dataset_name == 'code_generation' and metric_name.startswith('pass@'):
        combined_df = load_code_generation_evaluation_data(model_dir, metric_name)
        if combined_df is None:
            return None

        files_with_metrics = list(model_dir.glob("*_evaluation.json"))
        files_without_metrics = []

    # Special handling for qa with F1 metrics files
    elif dataset_name == 'qa' and metric_name == 'word_f1':
        # Look for QA CSV files
        csv_files = list(model_dir.glob("*.csv"))
        
        if not csv_files:
            return None
        
        all_data = []
        files_with_metrics = []
        files_without_metrics = []

        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)

                # Check if the required metric exists
                has_metric = metric_name in df.columns

                if not has_metric:
                    files_without_metrics.append(csv_file.name)
                    continue
                else:
                    files_with_metrics.append(csv_file.name)

                # Add task identifier
                df['task_identifier'] = csv_file.stem
                all_data.append(df)

            except Exception as e:
                print(f"Error reading {csv_file.name}: {e}")
                continue

        if not all_data:
            return None

        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)

    else:
        # Regular CSV file processing
        all_data = []
        files_with_metrics = []
        files_without_metrics = []

        # Load all CSV files
        csv_files = list(model_dir.glob("*.csv"))
        if not csv_files:
            return None

        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)

                # Check if the required metric exists
                has_metric = metric_name in df.columns

                if not has_metric:
                    files_without_metrics.append(csv_file.name)
                    continue
                else:
                    files_with_metrics.append(csv_file.name)

                # Add task identifier (like subject for MMLU, language pair for translation)
                df['task_identifier'] = csv_file.stem
                all_data.append(df)

            except Exception as e:
                print(f"Error reading {csv_file.name}: {e}")
                continue

        if not all_data:
            return None

        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)

    if metric_name not in combined_df.columns:
        return None

    # Find questions/samples that appear in all variations
    question_counts = combined_df.groupby(['task_identifier', 'original_row_index']).size().reset_index(name='count')
    total_variations = combined_df['variation_index'].nunique()

    # Filter to only include questions that appear in all variations
    shared_questions = question_counts[question_counts['count'] == total_variations]
    shared_questions = shared_questions[['task_identifier', 'original_row_index']]

    # Filter data to only shared questions
    filtered_df = pd.merge(combined_df, shared_questions, on=['task_identifier', 'original_row_index'], how='inner')

    if filtered_df.empty:
        return None

    # Get scaling information for this metric
    should_scale, metric_type = get_metric_scaling_info(metric_name)

    # Apply metric-specific transformations BEFORE calculating variation performance
    if should_scale and metric_name not in ['is_correct', 'extracted_is_correct']:
        # For non-boolean metrics that need scaling (like bleu, rouge, pass@k)
        filtered_df[metric_name] = filtered_df[metric_name] * 100

    # Calculate variation performance
    variation_scores = filtered_df.groupby('variation_index')[metric_name].agg(['mean', 'count']).reset_index()
    variation_scores.columns = ['variation_index', f'average_{metric_name}', 'question_count']

    # Apply scaling for boolean metrics AFTER calculating mean (since mean of booleans gives 0-1 ratio)
    if should_scale and metric_name in ['is_correct', 'extracted_is_correct']:
        variation_scores[f'average_{metric_name}'] = variation_scores[f'average_{metric_name}'] * 100

    # Prepare statistics
    total_questions = len(shared_questions)
    unique_tasks = filtered_df['task_identifier'].nunique()

    stats = {
        'total_questions': total_questions,
        'total_variations': total_variations,
        'unique_tasks': unique_tasks,
        'files_processed': len(files_with_metrics),
        'total_files': len(files_with_metrics) + len(files_without_metrics),
        'mean_score': variation_scores[f'average_{metric_name}'].mean(),
        'std_score': variation_scores[f'average_{metric_name}'].std(),
        'min_score': variation_scores[f'average_{metric_name}'].min(),
        'max_score': variation_scores[f'average_{metric_name}'].max(),
        'metric_type': metric_type,
        'scaled_to_100': should_scale
    }

    return variation_scores, stats

def create_unified_boxplot(model_name: str, dataset_results: Dict, output_dir: Path):
    """
    Create a unified box plot for all datasets for a specific model.
    """
    if not dataset_results:
        print(f"No data available for model {model_name}")
        return

    # Prepare data for plotting
    box_data = []
    labels = []
    # Use bold rainbow colors for maximum visual distinction
    # Define vivid rainbow colors manually for better contrast
    rainbow_colors = [
        '#FF0000',  # Red
        '#FF8000',  # Orange  
        '#FFFF00',  # Yellow
        '#80FF00',  # Lime Green
        '#00FF00',  # Green
        '#00FF80',  # Spring Green
        '#00FFFF',  # Cyan
        '#0080FF',  # Sky Blue
        '#0000FF',  # Blue
        '#8000FF',  # Purple
        '#FF00FF',  # Magenta
        '#FF0080'   # Pink
    ]
    
    # Cycle through colors if we have more datasets than colors
    colors = [rainbow_colors[i % len(rainbow_colors)] for i in range(len(dataset_results))]

    # Information for display below the plot
    info_lines = []

    # Check if we have any metrics that are not scaled to 0-100 for y-axis adjustment
    has_non_scaled_metrics = any(not stats['scaled_to_100'] for _, (_, stats, _) in dataset_results.items())

    for i, (dataset_name, (variation_scores, stats, metric_name)) in enumerate(dataset_results.items()):
        # Prepare box plot data
        metric_col = f'average_{metric_name}'
        box_data.append(variation_scores[metric_col].values)

                # Get display name for dataset
        display_info = DATASET_DISPLAY_NAMES.get(dataset_name, (dataset_name.upper(), ""))
        
        # Create formatted label with different styles:
        # Dataset name: Bold, Task name: Regular, Metric name: Italic
        # Extract dataset name and task description from tuple
        dataset_part = display_info[0]  # Dataset name (e.g., "MMLU", "GPQA-Diamond")
        task_part = display_info[1]     # Task description (e.g., "Multiple\nChoice")

        # Create metric description based on scaling
        if stats['scaled_to_100']:
            if metric_name == 'is_correct':
                metric_part = "Accuracy"
            elif metric_name == 'extracted_is_correct':
                metric_part = "Accuracy"
            elif metric_name == 'bleu':
                metric_part = "BLEU"
            elif metric_name == 'rouge1':
                metric_part = "ROUGE-1"
            elif metric_name == 'word_f1':
                metric_part = "Word F1"
            elif metric_name.startswith('pass@'):
                k = metric_name.split('@')[1]
                metric_part = f"Pass@{k}"
            elif metric_name == 'mae':
                metric_part = "MAE"
            else:
                metric_part = metric_name.upper()
        else:
            metric_part = metric_name.upper()
        
        metric_part = f"({metric_part})"
        
        # Create formatted label with proper formatting
        # Dataset name: Bold, Task name: Regular, Metric name: Italic
        # Handle newlines in dataset names properly
        if '\n' in dataset_part:
            # Split dataset name lines and make each line bold
            dataset_lines = dataset_part.split('\n')
            formatted_dataset = '\n'.join([f"$\\bf{{{line}}}$" for line in dataset_lines])
        else:
            formatted_dataset = f"$\\bf{{{dataset_part}}}$"
        
        if task_part:
            label = f"{formatted_dataset}\n{task_part}\n$\\it{{{metric_part}}}$"
        else:
            label = f"{formatted_dataset}\n$\\it{{{metric_part}}}$"

        labels.append(label)

        # Prepare info line - combine dataset name and task for info display
        combined_display_name = f"{dataset_part} {task_part.replace(chr(10), ' ')}"  # Replace newlines with spaces
        info_lines.append(f"{combined_display_name}: {stats['total_questions']} questions, {stats['total_variations']} variations")

    # Create the plot with configurable dimensions
    figure_width = max(MIN_FIGURE_WIDTH, len(dataset_results) * FIGURE_WIDTH_PER_BOX)
    fig, ax = plt.subplots(1, 1, figsize=(figure_width, FIGURE_HEIGHT))

    # Create box plot with configurable width and spacing
    positions = np.arange(1, len(dataset_results) + 1) * BOX_SPACING
    bp = ax.boxplot(box_data, patch_artist=True, labels=labels, widths=BOX_WIDTH, positions=positions)

    # Customize appearance with rainbow colors
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(colors[i])
        patch.set_alpha(BOX_ALPHA)
    
    # # Make median lines black and thicker
    # for median in bp['medians']:
    #     median.set_color('black')
    #     median.set_linewidth(MEDIAN_LINE_WIDTH)

        # Customize plot with configurable font sizes
    ax.set_ylabel('Metric (%)', fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_xlabel('', fontsize=AXIS_LABEL_FONT_SIZE)  # No x-axis label needed
    
    # Set title based on configuration
    if SHOW_MODEL_AS_TITLE:
        display_model_name = MODEL_DISPLAY_NAMES.get(model_name, model_name.replace("_", "-").upper())
        ax.set_title(display_model_name,
                     fontsize=TITLE_FONT_SIZE, fontweight='bold', fontfamily='serif')
    else:
        ax.set_title('')  # No title

    # Add horizontal grid lines only
    ax.grid(True, axis='y', alpha=0.3, linestyle='-', linewidth=0.5)

    # Set y-axis limits based on whether we have non-scaled metrics
    if not has_non_scaled_metrics:
        # All metrics are 0-100 scale
        ax.set_ylim(0, 105)
    else:
        # We have mixed scales, let matplotlib auto-scale but start from 0
        ax.set_ylim(bottom=0)

    # Keep x-axis labels straight (no rotation) with configurable font size
    plt.xticks(positions, labels, rotation=0, ha='center', fontsize=TICK_LABEL_FONT_SIZE)
    
    # Set Y-axis tick labels font size
    plt.yticks(fontsize=Y_TICK_LABEL_FONT_SIZE)

    # Add model name and/or info text below the plot based on configuration
    text_elements = []
    
    # Add model name if not shown as title
    if not SHOW_MODEL_AS_TITLE:
        display_model_name = MODEL_DISPLAY_NAMES.get(model_name)
        model_text = display_model_name
        text_elements.append(model_text)
    
    # Add info box if enabled
    if SHOW_INFO_BOX:
        info_text = '\n'.join(info_lines)
        text_elements.append(info_text)
    
    # Display text elements if any exist
    if text_elements:
        combined_text = '\n\n'.join(text_elements)
        
        # Determine font size based on content
        display_model_name = MODEL_DISPLAY_NAMES.get(model_name)
        
        if not SHOW_MODEL_AS_TITLE and SHOW_INFO_BOX:
            # Both model name and info - use different formatting
            model_text = display_model_name
            final_text = f'{model_text}\n\n{info_text}'
            fig.text(0.1, 0.02, final_text, fontsize=INFO_TEXT_FONT_SIZE, verticalalignment='bottom',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8), weight='bold')
        elif not SHOW_MODEL_AS_TITLE and not SHOW_INFO_BOX:
            # Only model name - center it and make it bold
            fig.text(0.5, 0.05, display_model_name,
                     fontsize=MODEL_NAME_FONT_SIZE, verticalalignment='bottom', 
                     horizontalalignment='center', fontweight='bold')
        else:
            # Only info box
            fig.text(0.1, 0.02, combined_text, fontsize=INFO_TEXT_FONT_SIZE, verticalalignment='bottom',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))

    # Adjust layout to make room for the text
    plt.tight_layout()
    if text_elements:
        plt.subplots_adjust(bottom=0.25)
    else:
        plt.subplots_adjust(bottom=0.1)

    # Save the plot in both PNG and PDF formats
    output_filename_png = f'{model_name}_unified_performance_boxplot.png'
    output_filename_pdf = f'{model_name}_unified_performance_boxplot.pdf'
    
    plt.savefig(output_dir / output_filename_png, dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / output_filename_pdf, dpi=300, bbox_inches='tight')
    
    print(f"  💾 Plot saved to: {output_dir / output_filename_png}")
    print(f"  💾 Plot saved to: {output_dir / output_filename_pdf}")
    
    # show the plot
    plt.show()
    plt.close()

    # Print summary statistics
    display_model_name = MODEL_DISPLAY_NAMES.get(model_name)
    print(f"\n=== {display_model_name.upper()} UNIFIED ANALYSIS ===")
    print(f"Datasets analyzed: {len(dataset_results)}")

    for dataset_name, (variation_scores, stats, metric_name) in dataset_results.items():
        print(f"\n{dataset_name.upper()} ({metric_name.upper()}):")
        print(f"  Questions: {stats['total_questions']}")
        print(f"  Variations: {stats['total_variations']}")
        print(f"  Files processed: {stats['files_processed']}/{stats['total_files']}")
        print(f"  Scaled to 0-100: {stats['scaled_to_100']} ({stats['metric_type']})")

        if stats['scaled_to_100']:
            if metric_name in ['is_correct', 'extracted_is_correct']:
                print(f"  Mean accuracy: {stats['mean_score']:.2f}%")
                print(f"  Std dev: {stats['std_score']:.2f}%")
                print(f"  Range: {stats['min_score']:.2f}% - {stats['max_score']:.2f}%")
            elif metric_name.startswith('pass@'):
                print(f"  Mean {metric_name.upper()}: {stats['mean_score']:.2f}%")
                print(f"  Std dev: {stats['std_score']:.2f}%")
                print(f"  Range: {stats['min_score']:.2f}% - {stats['max_score']:.2f}%")
            else:
                print(f"  Mean {metric_name.upper()}: {stats['mean_score']:.2f}")
                print(f"  Std dev: {stats['std_score']:.2f}")
                print(f"  Range: {stats['min_score']:.2f} - {stats['max_score']:.2f}")
        else:
            print(f"  Mean score: {stats['mean_score']:.4f}")
            print(f"  Std dev: {stats['std_score']:.4f}")
            print(f"  Range: {stats['min_score']:.4f} - {stats['max_score']:.4f}")

def main():
    """
    Main function to run the unified analysis for all models and datasets.
    """
    # Set up paths
    results_base_dir = Path(__file__).parent.parent / "tasks_data" / "results"
    figures_dir = Path(__file__).parent.parent / "tasks_data" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("Starting Unified Box Plot Analysis")
    print("=" * 50)
    print(f"Metrics scaled to 0-100: {list(SCALE_TO_100_METRICS.keys())}")
    print(f"Metrics keeping original scale: {list(KEEP_ORIGINAL_SCALE_METRICS.keys())}")
    print()

    for model_name in MODELS:
        print(f"\nAnalyzing model: {model_name}")
        print("-" * 30)

        model_results = {}

        for dataset_name, metric_name in DATASET_METRICS.items():
            dataset_dir = results_base_dir / dataset_name / model_name

            if not dataset_dir.exists():
                print(f"  ⚠️  {dataset_name}: Directory not found - {dataset_dir}")
                continue

            should_scale, metric_type = get_metric_scaling_info(metric_name)
            scale_info = "→ 0-100%" if should_scale else "→ original scale"

            # Special info for different datasets
            if dataset_name == 'code_generation':
                print(f"  🔍 Processing {dataset_name} ({metric_name} from evaluation JSON {scale_info})...")
            elif dataset_name == 'qa':
                print(f"  🔍 Processing {dataset_name} ({metric_name} from F1 metrics CSV {scale_info})...")
            else:
                print(f"  🔍 Processing {dataset_name} ({metric_name} {scale_info})...")

            result = analyze_dataset_variations(dataset_dir, dataset_name, metric_name)

            if result is None:
                print(f"  ❌ {dataset_name}: No valid data found")
                continue

            variation_scores, stats = result
            model_results[dataset_name] = (variation_scores, stats, metric_name)
            print(f"  ✅ {dataset_name}: {stats['total_questions']} questions, {stats['total_variations']} variations, {stats['files_processed']} files")

        # Create unified box plot for this model
        if model_results:
            print(f"\n  📊 Creating unified box plot for {model_name}...")
            create_unified_boxplot(model_name, model_results, figures_dir)
        else:
            print(f"  ❌ No data found for model {model_name}")

    print(f"\n✅ Analysis complete! Plots saved to: {figures_dir}")

if __name__ == "__main__":
    main()