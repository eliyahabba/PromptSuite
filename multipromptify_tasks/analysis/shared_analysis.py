#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional


def analyze_task_variations(
    model_dir: Path,
    task_type: str,
    metric_name: str,
    file_pattern: str = "*.csv",
    subject_column: Optional[str] = None,
    combine_all_files: bool = False
) -> None:
    """
    Analyze task variations for different task types with specified metrics.
    
    Args:
        model_dir: Directory containing the model results
        task_type: Type of task ('mmlu', 'translation', etc.)
        metric_name: Name of the metric to analyze ('is_correct', 'bleu', 'rouge1', etc.)
        file_pattern: Pattern to match CSV files
        subject_column: Column name for subject/task grouping (optional)
        combine_all_files: If True, combine all files and analyze together regardless of individual file metrics
    """
    all_data = []
    files_with_metrics = []
    files_without_metrics = []
    
    # Load all CSV files
    csv_files = list(model_dir.glob(file_pattern))
    if not csv_files:
        print(f"No CSV files found in {model_dir}")
        return
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            
            # Check if the required metric exists
            has_metric = metric_name in df.columns
            
            if not has_metric:
                files_without_metrics.append(csv_file.name)
                if not combine_all_files:
                    print(f"Warning: Metric '{metric_name}' not found in {csv_file.name}")
                    continue
            else:
                files_with_metrics.append(csv_file.name)
                
            # Add subject/task identifier if not provided
            if subject_column is None:
                df['task_identifier'] = csv_file.stem
            else:
                if subject_column not in df.columns:
                    df['task_identifier'] = csv_file.stem
                else:
                    df['task_identifier'] = df[subject_column]
            
            all_data.append(df)
            
        except Exception as e:
            print(f"Error reading {csv_file.name}: {e}")
            continue
    
    if not all_data:
        print("No valid data files found")
        return
    
    # Print summary of files
    if files_with_metrics:
        print(f"Files with {metric_name} metrics: {len(files_with_metrics)}")
        for f in files_with_metrics:
            print(f"  ✓ {f}")
    
    if files_without_metrics:
        print(f"Files without {metric_name} metrics: {len(files_without_metrics)}")
        for f in files_without_metrics:
            print(f"  ✗ {f}")
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # If combining all files but some don't have metrics, filter to only files with metrics
    if combine_all_files and files_without_metrics:
        print(f"\nFiltering to only analyze files with {metric_name} metrics...")
        files_with_metrics_stems = [Path(f).stem for f in files_with_metrics]
        combined_df = combined_df[combined_df['task_identifier'].isin(files_with_metrics_stems)]
    
    # Check if we have the metric in the filtered data
    if metric_name not in combined_df.columns:
        print(f"Error: No data with metric '{metric_name}' found")
        return
    
    # Find questions/samples that appear in all variations
    question_counts = combined_df.groupby(['task_identifier', 'original_row_index']).size().reset_index(name='count')
    total_variations = combined_df['variation_index'].nunique()
    
    # Filter to only include questions that appear in all variations
    shared_questions = question_counts[question_counts['count'] == total_variations]
    shared_questions = shared_questions[['task_identifier', 'original_row_index']]
    
    # Filter data to only shared questions
    filtered_df = pd.merge(combined_df, shared_questions, on=['task_identifier', 'original_row_index'], how='inner')
    
    if filtered_df.empty:
        print("No shared questions found across all variations")
        return
    # before calculating performance, if metric is blue we need to multiply it by 100
    if metric_name in ['bleu', 'rouge1', 'rouge2', 'rougeL']:
        filtered_df[metric_name] = filtered_df[metric_name] * 100
    # Calculate variation performance
    variation_scores = filtered_df.groupby('variation_index')[metric_name].agg(['mean', 'count']).reset_index()
    variation_scores.columns = ['variation_index', f'average_{metric_name}', 'question_count']
    
    total_questions = len(shared_questions)
    unique_tasks = filtered_df['task_identifier'].nunique()
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.scatter(variation_scores['variation_index'], variation_scores[f'average_{metric_name}'], 
                s=50, alpha=0.7)
    
    plt.xlabel('Variation Index')
    plt.ylabel(f'Average {metric_name.title()}')
    
    # Create appropriate title based on task type
    if task_type.lower() == 'mmlu':
        title = f'MMLU Performance by Variation Index\nModel: {model_dir.name}\nTotal Questions: {total_questions} (from {unique_tasks} subjects)'
    elif task_type.lower() == 'translation':
        title = f'Translation Performance ({metric_name.upper()}) by Variation Index\nModel: {model_dir.name}\nTotal Questions: {total_questions} (from {unique_tasks} language pairs)'
    else:
        title = f'{task_type.title()} Performance ({metric_name.title()}) by Variation Index\nModel: {model_dir.name}\nTotal Questions: {total_questions} (from {unique_tasks} tasks)'
    
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot to figures directory
    figures_dir = Path(__file__).parent.parent / "tasks_data" /"figures" / task_type.lower()
    figures_dir.mkdir(parents=True, exist_ok=True)
    output_filename = f'{model_dir.name}_{task_type}_{metric_name}_variation_performance.png'
    plt.savefig(figures_dir / output_filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create box plot
    variation_scores_dict = {metric_name: variation_scores}
    create_box_plots(variation_scores_dict, task_type, model_dir.name, figures_dir, total_variations, total_questions)
    
    # Print statistics
    avg_score = variation_scores[f'average_{metric_name}'].mean()
    max_score = variation_scores[f'average_{metric_name}'].max()
    min_score = variation_scores[f'average_{metric_name}'].min()
    best_variation = variation_scores.loc[variation_scores[f'average_{metric_name}'].idxmax(), 'variation_index']
    worst_variation = variation_scores.loc[variation_scores[f'average_{metric_name}'].idxmin(), 'variation_index']
    
    print(f"\n=== {task_type.upper()} Analysis Results ({metric_name.upper()}) ===")
    print(f"Total variations: {len(variation_scores)}")
    print(f"Total shared questions: {total_questions}")
    print(f"Tasks/subjects analyzed: {unique_tasks}")
    print(f"Average {metric_name}: {avg_score:.4f}")
    print(f"Best variation: {best_variation} (score: {max_score:.4f})")
    print(f"Worst variation: {worst_variation} (score: {min_score:.4f})")
    print(f"Score range: {max_score - min_score:.4f}")
    
    # Additional statistics for different metrics
    if metric_name == 'is_correct':
        print(f"Best accuracy: {max_score:.1%}")
        print(f"Worst accuracy: {min_score:.1%}")
    elif metric_name in ['bleu', 'rouge1', 'rouge2', 'rougeL']:
        print(f"Best {metric_name.upper()}: {max_score:.4f}")
        print(f"Worst {metric_name.upper()}: {min_score:.4f}")
    elif metric_name == 'sacrebleu':
        print(f"Best SacreBlEU: {max_score:.2f}")
        print(f"Worst SacreBlEU: {min_score:.2f}")


def analyze_multiple_metrics(
    model_dir: Path,
    task_type: str,
    metrics: List[str],
    file_pattern: str = "*.csv",
    subject_column: Optional[str] = None,
    combine_all_files: bool = False
) -> None:
    """
    Analyze multiple metrics for a task type and create subplots.
    
    Args:
        model_dir: Directory containing the model results
        task_type: Type of task ('mmlu', 'translation', etc.)
        metrics: List of metric names to analyze
        file_pattern: Pattern to match CSV files
        subject_column: Column name for subject/task grouping (optional)
        combine_all_files: If True, combine all files and analyze together regardless of individual file metrics
    """
    all_data = []
    files_with_any_metrics = []
    files_without_any_metrics = []
    
    # Load all CSV files
    csv_files = list(model_dir.glob(file_pattern))
    if not csv_files:
        print(f"No CSV files found in {model_dir}")
        return
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            
            # Check if any of the required metrics exist
            available_metrics = [m for m in metrics if m in df.columns]
            
            if not available_metrics:
                files_without_any_metrics.append(csv_file.name)
                if not combine_all_files:
                    print(f"Warning: None of the metrics {metrics} found in {csv_file.name}")
                    continue
            else:
                files_with_any_metrics.append(csv_file.name)
                
            # Add subject/task identifier if not provided
            if subject_column is None:
                df['task_identifier'] = csv_file.stem
            else:
                if subject_column not in df.columns:
                    df['task_identifier'] = csv_file.stem
                else:
                    df['task_identifier'] = df[subject_column]
            
            all_data.append(df)
            
        except Exception as e:
            print(f"Error reading {csv_file.name}: {e}")
            continue
    
    if not all_data:
        print("No valid data files found")
        return
    
    # Print summary of files
    if files_with_any_metrics:
        print(f"Files with metrics: {len(files_with_any_metrics)}")
        for f in files_with_any_metrics:
            print(f"  ✓ {f}")
    
    if files_without_any_metrics:
        print(f"Files without metrics: {len(files_without_any_metrics)}")
        for f in files_without_any_metrics:
            print(f"  ✗ {f}")
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # If combining all files but some don't have metrics, filter to only files with metrics
    if combine_all_files and files_without_any_metrics:
        print(f"\nFiltering to only analyze files with metrics...")
        files_with_metrics_stems = [Path(f).stem for f in files_with_any_metrics]
        combined_df = combined_df[combined_df['task_identifier'].isin(files_with_metrics_stems)]
    
    # Find available metrics in the combined data
    available_metrics = [m for m in metrics if m in combined_df.columns]
    if not available_metrics:
        print(f"None of the requested metrics {metrics} are available in the data")
        return
    
    # Find questions/samples that appear in all variations
    question_counts = combined_df.groupby(['task_identifier', 'original_row_index']).size().reset_index(name='count')
    total_variations = combined_df['variation_index'].nunique()
    
    shared_questions = question_counts[question_counts['count'] == total_variations]
    shared_questions = shared_questions[['task_identifier', 'original_row_index']]
    
    # Filter data to only shared questions
    filtered_df = pd.merge(combined_df, shared_questions, on=['task_identifier', 'original_row_index'], how='inner')
    for metric_name in ['bleu', 'rouge1', 'rouge2', 'rougeL']:
        if metric_name in available_metrics:
            filtered_df[metric_name] = filtered_df[metric_name] * 100

    if filtered_df.empty:
        print("No shared questions found across all variations")
        return
    
    unique_tasks = filtered_df['task_identifier'].nunique()
    
    # Create subplots
    n_metrics = len(available_metrics)
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    if n_metrics == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes if n_metrics > 1 else [axes]
    else:
        axes = axes.flatten()
    
    for i, metric_name in enumerate(available_metrics):
        ax = axes[i]
        
        # Calculate variation performance for this metric
        variation_scores = filtered_df.groupby('variation_index')[metric_name].agg(['mean', 'count']).reset_index()
        variation_scores.columns = ['variation_index', f'average_{metric_name}', 'question_count']
        
        # Create scatter plot
        ax.scatter(variation_scores['variation_index'], variation_scores[f'average_{metric_name}'], 
                  s=30, alpha=0.7)
        
        ax.set_xlabel('Variation Index')
        ax.set_ylabel(f'Average {metric_name.title()}')
        ax.set_title(f'{metric_name.upper()} Performance')
        ax.grid(True, alpha=0.3)
    
    # Hide empty subplots
    for i in range(n_metrics, len(axes)):
        axes[i].set_visible(False)
    
    if task_type.lower() == 'translation':
        suptitle = f'Translation Performance by Variation Index\nModel: {model_dir.name}\nTotal Questions: {len(shared_questions)} (from {unique_tasks} language pairs)'
    else:
        suptitle = f'{task_type.title()} Performance by Variation Index\nModel: {model_dir.name}\nTotal Questions: {len(shared_questions)} (from {unique_tasks} tasks)'
    
    plt.suptitle(suptitle, fontsize=14)
    plt.tight_layout()
    
    # Save the plot to figures directory
    figures_dir = Path(__file__).parent.parent / "tasks_data" /"figures" / task_type.lower()
    figures_dir.mkdir(parents=True, exist_ok=True)
    output_filename = f'{model_dir.name}_{task_type}_multi_metric_variation_performance.png'
    plt.savefig(figures_dir / output_filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create box plots for all metrics
    variation_scores_dict = {}
    for metric_name in available_metrics:
        variation_scores = filtered_df.groupby('variation_index')[metric_name].agg(['mean', 'count']).reset_index()
        variation_scores.columns = ['variation_index', f'average_{metric_name}', 'question_count']
        variation_scores_dict[metric_name] = variation_scores
    
    create_box_plots(variation_scores_dict, task_type, model_dir.name, figures_dir, total_variations, len(shared_questions))
    
    # Print statistics for all metrics
    print(f"\n=== {task_type.upper()} Multi-Metric Analysis Results ===")
    print(f"Total variations: {total_variations}")
    print(f"Total shared questions: {len(shared_questions)}")
    print(f"Tasks/subjects analyzed: {unique_tasks}")
    
    for metric_name in available_metrics:
        variation_scores = filtered_df.groupby('variation_index')[metric_name].agg(['mean']).reset_index()
        variation_scores.columns = ['variation_index', f'average_{metric_name}']
        
        avg_score = variation_scores[f'average_{metric_name}'].mean()
        max_score = variation_scores[f'average_{metric_name}'].max()
        min_score = variation_scores[f'average_{metric_name}'].min()
        
        print(f"\n{metric_name.upper()}:")
        print(f"  Average: {avg_score:.4f}")
        print(f"  Best: {max_score:.4f}")
        print(f"  Worst: {min_score:.4f}")
        print(f"  Range: {max_score - min_score:.4f}")


def analyze_math_variations(model_dir: Path) -> None:
    """
    Analyze math problem solving variations for accuracy.
    
    Args:
        model_dir: Directory containing the model results for math problems
    """
    analyze_task_variations(
        model_dir=model_dir,
        task_type="math",
        metric_name="is_correct",
        file_pattern="*.csv",
        subject_column=None,
        combine_all_files=True
    )


def analyze_code_generation_variations(model_dir: Path) -> None:
    """
    Analyze code generation variations for correctness and syntactic accuracy.
    
    Args:
        model_dir: Directory containing the model results for code generation
    """
    analyze_task_variations(
        model_dir=model_dir,
        task_type="code_generation",
        metric_name="is_correct",
        file_pattern="*.csv",
        subject_column=None,
        combine_all_files=True
    )


def analyze_code_generation_multiple_metrics(model_dir: Path) -> None:
    """
    Analyze code generation variations with multiple metrics.
    
    Args:
        model_dir: Directory containing the model results for code generation
    """
    analyze_multiple_metrics(
        model_dir=model_dir,
        task_type="code_generation",
        metrics=["is_correct", "functionally_correct", "syntactically_correct", "pass_at_1", "bleu", "rouge1", "rougeL"],
        file_pattern="*.csv",
        subject_column=None,
        combine_all_files=True
    )


def analyze_code_generation_functional_correctness(model_dir: Path) -> None:
    """
    Analyze code generation variations for functional correctness.
    
    Args:
        model_dir: Directory containing the model results for code generation
    """
    analyze_task_variations(
        model_dir=model_dir,
        task_type="code_generation",
        metric_name="functionally_correct",
        file_pattern="*.csv",
        subject_column=None,
        combine_all_files=True
    )


def analyze_code_generation_pass_at_1(model_dir: Path) -> None:
    """
    Analyze code generation variations for pass@1 scores.
    
    Args:
        model_dir: Directory containing the model results for code generation
    """
    analyze_task_variations(
        model_dir=model_dir,
        task_type="code_generation",
        metric_name="pass_at_1",
        file_pattern="*.csv",
        subject_column=None,
        combine_all_files=True
    )


def analyze_musique_variations(model_dir: Path) -> None:
    """
    Analyze MuSiQue multi-hop question answering variations for exact match accuracy.
    
    Args:
        model_dir: Directory containing the model results for MuSiQue
    """
    analyze_task_variations(
        model_dir=model_dir,
        task_type="musique",
        metric_name="is_correct",
        file_pattern="*.csv",
        subject_column=None,
        combine_all_files=True
    )


def analyze_musique_multiple_metrics(model_dir: Path) -> None:
    """
    Analyze MuSiQue variations with multiple metrics including F1, precision, recall, and text generation metrics.
    
    Args:
        model_dir: Directory containing the model results for MuSiQue
    """
    analyze_multiple_metrics(
        model_dir=model_dir,
        task_type="musique",
        metrics=["is_correct", "word_f1", "word_precision", "word_recall", "bleu", "rouge1", "rouge2", "rougeL"],
        file_pattern="*.csv",
        subject_column=None,
        combine_all_files=True
    )


def analyze_musique_word_f1(model_dir: Path) -> None:
    """
    Analyze MuSiQue variations for word-level F1 scores.
    
    Args:
        model_dir: Directory containing the model results for MuSiQue
    """
    analyze_task_variations(
        model_dir=model_dir,
        task_type="musique",
        metric_name="word_f1",
        file_pattern="*.csv",
        subject_column=None,
        combine_all_files=True
    )


def analyze_qa_variations(model_dir: Path) -> None:
    """
    Analyze Question Answering variations for exact match accuracy.
    
    Args:
        model_dir: Directory containing the model results for Question Answering
    """
    analyze_task_variations(
        model_dir=model_dir,
        task_type="question_answering",
        metric_name="is_correct",
        file_pattern="*.csv",
        subject_column=None,
        combine_all_files=True
    )


def analyze_qa_multiple_metrics(model_dir: Path) -> None:
    """
    Analyze Question Answering variations with multiple metrics including exact match, F1, precision, recall, and text generation metrics.
    
    Args:
        model_dir: Directory containing the model results for Question Answering
    """
    analyze_multiple_metrics(
        model_dir=model_dir,
        task_type="question_answering",
        metrics=["is_correct", "exact_match", "word_f1", "word_precision", "word_recall", "bleu", "rouge1", "rouge2", "rougeL"],
        file_pattern="*.csv",
        subject_column=None,
        combine_all_files=True
    )


def analyze_qa_word_f1(model_dir: Path) -> None:
    """
    Analyze Question Answering variations for word-level F1 scores.
    
    Args:
        model_dir: Directory containing the model results for Question Answering
    """
    analyze_task_variations(
        model_dir=model_dir,
        task_type="question_answering",
        metric_name="word_f1",
        file_pattern="*.csv",
        subject_column=None,
        combine_all_files=True
    )


def create_box_plots(variation_scores_dict: Dict[str, pd.DataFrame], task_type: str, model_name: str, figures_dir: Path, total_variations: int = 0, total_questions: int = 0) -> None:
    """
    Create box plots for variation performance metrics - all metrics on the same plot.
    
    Args:
        variation_scores_dict: Dictionary mapping metric names to variation scores DataFrames
        task_type: Type of task ('mmlu', 'translation', etc.)
        model_name: Name of the model
        figures_dir: Directory to save the figure
        total_variations: Total number of variations analyzed
        total_questions: Total number of questions/samples analyzed
    """
    metrics = list(variation_scores_dict.keys())
    n_metrics = len(metrics)
    
    if n_metrics == 0:
        return
    
    # Create single figure for all metrics
    fig, ax = plt.subplots(1, 1, figsize=(max(8, 2 + n_metrics * 1.5), 6))
    
    # Prepare data for all metrics
    box_data = []
    labels = []
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink', 'lightgray']
    
    for i, metric_name in enumerate(metrics):
        variation_scores = variation_scores_dict[metric_name]
        box_data.append(variation_scores[f'average_{metric_name}'].values)
        labels.append(metric_name.upper())
    
    # Create box plot with all metrics
    bp = ax.boxplot(box_data, patch_artist=True, labels=labels)
    
    # Customize box plot appearance with different colors for each metric
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(colors[i % len(colors)])
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Performance Score')
    ax.set_xlabel('Metrics')
    
    # Set main title with sample information
    sample_info = ""
    if total_variations > 0 and total_questions > 0:
        sample_info = f"\nVariations: {total_variations} | Questions: {total_questions}"
    elif total_variations > 0:
        sample_info = f"\nVariations: {total_variations}"
    elif total_questions > 0:
        sample_info = f"\nQuestions: {total_questions}"
    
    if task_type.lower() == 'translation':
        title = f'Translation Performance Distribution\nModel: {model_name}{sample_info}'
    else:
        title = f'{task_type.title()} Performance Distribution\nModel: {model_name}{sample_info}'
    
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Add statistics text for each metric
    stats_text = []
    for i, metric_name in enumerate(metrics):
        variation_scores = variation_scores_dict[metric_name]
        mean_val = variation_scores[f'average_{metric_name}'].mean()
        std_val = variation_scores[f'average_{metric_name}'].std()
        stats_text.append(f'{metric_name.upper()}: μ={mean_val:.4f}, σ={std_val:.4f}')
    
    # Place statistics text
    stats_str = '\n'.join(stats_text)
    ax.text(0.02, 0.98, stats_str, 
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    if n_metrics == 1:
        output_filename = f'{model_name}_{task_type}_{metrics[0]}_boxplot.png'
    else:
        output_filename = f'{model_name}_{task_type}_combined_metrics_boxplot.png'
    
    plt.savefig(figures_dir / output_filename, dpi=300, bbox_inches='tight')
    plt.show()