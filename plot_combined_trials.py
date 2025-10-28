#!/usr/bin/env python3
"""
Plot combined HPO checkpoint and time penalty trial data showing generations vs fitness evolution.

This script loads both checkpoint files from HPO folders and history JSON files from rerun_outputs,
then creates comprehensive plots showing how fitness metrics evolve over generations.
"""

import os
import pickle
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse


def _label_prefix(folder_name: str) -> str:
    """Return a short prefix based on the trial type."""
    lowered = folder_name.lower()
    if 'time_penalty' in lowered or 'time penalty' in lowered:
        return 'tp'
    elif 'hpo_ckpts_full_2' in lowered:
        return 'fid'  # No time penalty (blue)
    elif 'hpo_ckpts' in lowered and 'full' not in lowered:
        return 'tp'   # Time penalty (red)
    return 'fid'


def _format_run_label(folder_name: str, run_id: str) -> str:
    """Generate a concise legend label for the given run."""
    prefix = _label_prefix(folder_name)
    label = run_id
    if label.startswith('ckpt_'):
        label = label[len('ckpt_'):]
    if not label.startswith(f'{prefix}_'):
        label = f'{prefix}_{label}'
    return label


def _ensure_unique_label(label: str, existing: Dict[str, Dict]) -> str:
    """Ensure legend labels remain unique when added to a mapping."""
    if label not in existing:
        return label

    index = 2
    while f"{label}_{index}" in existing:
        index += 1
    return f"{label}_{index}"


def load_checkpoint_data(checkpoint_path: str) -> Dict:
    """Load checkpoint data from a pickle file."""
    try:
        with open(checkpoint_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading {checkpoint_path}: {e}")
        return None


def load_history_data(history_path: str) -> Dict:
    """Load history data from a JSON file."""
    try:
        with open(history_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {history_path}: {e}")
        return None


def extract_fitness_history(ckpt_data: Dict) -> Dict:
    """Extract fitness history from checkpoint data."""
    if not ckpt_data or 'history' not in ckpt_data:
        return {}
    
    history = ckpt_data['history']
    if not history:
        return {}
    
    # Extract fitness metrics from history
    generations = []
    best_fitness = []
    avg_fitness = []
    avg_cost = []
    front0_size = []
    
    for entry in history:
        generations.append(entry.get('gen', 0))
        best_fitness.append(entry.get('best_scalar_fitness', 0))
        avg_fitness.append(entry.get('avg_adjusted_raw', 0))
        avg_cost.append(entry.get('avg_cost', 0))
        front0_size.append(entry.get('front0_size', 0))
    
    return {
        'generations': generations,
        'best_fitness': best_fitness,
        'avg_fitness': avg_fitness,
        'avg_cost': avg_cost,
        'front0_size': front0_size,
        'final_generation': ckpt_data.get('generation', 0)
    }


def extract_fitness_from_history(history_data: List[Dict]) -> Dict:
    """Extract fitness metrics from history data."""
    if not history_data:
        return {}
    
    # Extract fitness metrics from history
    generations = []
    best_fitness = []
    avg_fitness = []
    avg_cost = []
    front0_size = []
    
    for entry in history_data:
        generations.append(entry.get('gen', 0))
        best_fitness.append(entry.get('best_scalar_fitness', 0))
        avg_fitness.append(entry.get('avg_adjusted_raw', 0))
        avg_cost.append(entry.get('avg_cost', 0))
        front0_size.append(entry.get('front0_size', 0))
    
    return {
        'generations': generations,
        'best_fitness': best_fitness,
        'avg_fitness': avg_fitness,
        'avg_cost': avg_cost,
        'front0_size': front0_size,
        'final_generation': max(generations) if generations else 0
    }


def plot_combined_evolution(checkpoint_folders: List[str], output_dir: str = "combined_plots"):
    """Plot fitness evolution for both HPO checkpoints and time penalty trials."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect data from all sources
    all_data = {}
    
    # Process HPO checkpoint folders
    for folder in checkpoint_folders:
        if not os.path.exists(folder):
            print(f"Warning: Folder {folder} does not exist")
            continue
            
        print(f"Processing HPO folder: {folder}")
        folder_data = {}
        
        # Find all checkpoint files in the folder
        checkpoint_files = [f for f in os.listdir(folder) if f.startswith('ckpt_') and f.endswith('.pkl')]
        
        for ckpt_file in checkpoint_files:
            ckpt_path = os.path.join(folder, ckpt_file)
            ckpt_data = load_checkpoint_data(ckpt_path)
            
            if ckpt_data:
                fitness_data = extract_fitness_history(ckpt_data)
                if fitness_data:
                    # Extract run identifier from filename
                    run_id = ckpt_file.replace('ckpt_', '').replace('.pkl', '')
                    folder_data[run_id] = fitness_data
                    print(f"  Loaded HPO {run_id}: {len(fitness_data['generations'])} generations")
        
        if folder_data:
            all_data[f"HPO_{folder}"] = folder_data
    
    
    # Create plots
    create_combined_fitness_plots(all_data, output_dir)
    create_combined_comparison_plots(all_data, output_dir)


def create_combined_fitness_plots(all_data: Dict, output_dir: str):
    """Create combined fitness plots showing both HPO and time penalty trials."""
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Combined HPO and Time Penalty Trials - Fitness Evolution', fontsize=16)
    
    # Define colors for different trial types
    no_time_penalty_color = 'blue'  # hpo_ckpts_full_2
    time_penalty_color = 'red'      # hpo_ckpts
    
    # Plot 1: Best Fitness over Generations
    ax1 = axes[0, 0]
    for folder_name, folder_data in all_data.items():
        color = no_time_penalty_color if 'hpo_ckpts_full_2' in folder_name else time_penalty_color

        for run_id, data in folder_data.items():
            label = _format_run_label(folder_name, run_id)
            # Apply transformation: best_fitness / 47.6 * 100
            if label.startswith('tp_'):
                transformed_fitness = [f / 47.6 * 100 for f in data['best_fitness']]
            else:
                transformed_fitness = data['best_fitness']
            ax1.plot(
                data['generations'],
                transformed_fitness,
                marker='o',
                label=label,
                alpha=0.7,
                color=color,
                linewidth=2,
            )
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Best Fitness (Scaled %)')
    ax1.set_title('Best Fitness Evolution')
    ax1.legend(fontsize=10, ncol=2)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Average Fitness over Generations
    ax2 = axes[0, 1]
    for folder_name, folder_data in all_data.items():
        color = no_time_penalty_color if 'hpo_ckpts_full_2' in folder_name else time_penalty_color

        for run_id, data in folder_data.items():
            label = _format_run_label(folder_name, run_id)
            if label.startswith('tp_'):
                transformed_fitness = [f / 47.6 * 100 for f in data['avg_fitness']]
            else:
                transformed_fitness = data['avg_fitness']
            ax2.plot(
                data['generations'],
                transformed_fitness,
                marker='s',
                label=label,
                alpha=0.7,
                color=color,
                linewidth=2,
            )
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Average Fitness (Scaled %)')
    ax2.set_title('Average Fitness Evolution')
    ax2.legend(fontsize=10, ncol=2)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Average Cost over Generations
    ax3 = axes[1, 0]
    for folder_name, folder_data in all_data.items():
        color = no_time_penalty_color if 'hpo_ckpts_full_2' in folder_name else time_penalty_color

        for run_id, data in folder_data.items():
            label = _format_run_label(folder_name, run_id)
            ax3.plot(
                data['generations'],
                data['avg_cost'],
                marker='^',
                label=label,
                alpha=0.7,
                color=color,
                linewidth=2,
            )
    ax3.set_xlabel('Generation')
    ax3.set_ylabel('Average Cost')
    ax3.set_title('Average Cost Evolution')
    ax3.legend(fontsize=10, ncol=2)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Pareto Front Size over Generations
    ax4 = axes[1, 1]
    for folder_name, folder_data in all_data.items():
        color = no_time_penalty_color if 'hpo_ckpts_full_2' in folder_name else time_penalty_color

        for run_id, data in folder_data.items():
            label = _format_run_label(folder_name, run_id)
            ax4.plot(
                data['generations'],
                data['front0_size'],
                marker='d',
                label=label,
                alpha=0.7,
                color=color,
                linewidth=2,
            )
    ax4.set_xlabel('Generation')
    ax4.set_ylabel('Pareto Front Size')
    ax4.set_title('Pareto Front Size Evolution')
    ax4.legend(fontsize=10, ncol=2)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_dir, 'combined_fitness_evolution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved combined plot: {output_path}")
    plt.close()


def create_combined_comparison_plots(all_data: Dict, output_dir: str):
    """Create comparison plots across all trial types."""
    
    # Collect all runs for comparison
    all_runs = {}
    for folder_name, folder_data in all_data.items():
        for run_id, data in folder_data.items():
            label = _format_run_label(folder_name, run_id)
            label = _ensure_unique_label(label, all_runs)
            all_runs[label] = data
    
    if not all_runs:
        print("No data to plot")
        return
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(24, 14))
    fig.suptitle('Combined HPO and Time Penalty Trials - Comprehensive Comparison', fontsize=16)
    
    # Plot 1: Best Fitness Comparison
    ax1 = axes[0, 0]
    for run_id, data in all_runs.items():
        color = 'blue' if run_id.startswith('fid_') else 'red'
        if run_id.startswith('tp_'):
            transformed_fitness = [f / 47.6 * 100 for f in data['best_fitness']]
        else:
            transformed_fitness = data['best_fitness']
        ax1.plot(
            data['generations'],
            transformed_fitness,
            marker='o',
            label=run_id,
            alpha=0.8,
            color=color,
            linewidth=2,
        )
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Best Fitness (Scaled %)')
    ax1.set_title('Best Fitness Evolution - All Trials')
    ax1.legend(fontsize=10, ncol=2)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Average Fitness Comparison
    ax2 = axes[0, 1]
    for run_id, data in all_runs.items():
        color = 'blue' if run_id.startswith('fid_') else 'red'
        # if run_id.startswith('tp_'):
        #     transformed_fitness = [f / 47.6 * 100 for f in data['avg_fitness']]
        # else:
        #     transformed_fitness = data['avg_fitness']
        transformed_fitness = [(f * 100) for f in data['avg_fitness']]
        ax2.plot(
            data['generations'],
            transformed_fitness,
            marker='s',
            label=run_id,
            alpha=0.8,
            color=color,
            linewidth=2,
        )
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Average Fitness (Scaled %)')
    ax2.set_title('Average Fitness Evolution - All Trials')
    ax2.legend(fontsize=10, ncol=2)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Final Fitness Comparison (bar chart)
    ax3 = axes[1, 0]
    final_fitness = []
    run_names = []
    colors_bar = []
    for run_id, data in all_runs.items():
        if run_id.startswith('tp_'):
            transformed_fitness = [f / 47.6 * 100 for f in data['best_fitness']]
        else:
            transformed_fitness = data['best_fitness']
        if data['best_fitness']:
            final_fitness.append(transformed_fitness[-1])
            run_names.append(run_id)
            colors_bar.append('blue' if run_id.startswith('fid_') else 'red')

    if final_fitness:
        bars = ax3.bar(range(len(run_names)), final_fitness, alpha=0.7, color=colors_bar)
        ax3.set_xlabel('Trial')
        ax3.set_ylabel('Final Best Fitness (Scaled %)')
        ax3.set_title('Final Best Fitness Comparison')
        ax3.set_xticks(range(len(run_names)))
        ax3.set_xticklabels(run_names, rotation=45, ha='right', fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, final_fitness)):
            ax3.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f'{value:.1f}',
                ha='center',
                va='bottom',
                fontsize=10,
            )
    
    # Plot 4: Convergence Analysis
    ax4 = axes[1, 1]
    for run_id, data in all_runs.items():
        if len(data['best_fitness']) > 1:
            color = 'blue' if run_id.startswith('fid_') else 'red'
            # Calculate improvement over time
            improvements = np.diff(data['best_fitness'])
            ax4.plot(
                data['generations'][1:],
                improvements,
                marker='o',
                label=run_id,
                alpha=0.8,
                color=color,
                linewidth=2,
            )
    ax4.set_xlabel('Generation')
    ax4.set_ylabel('Fitness Improvement')
    ax4.set_title('Fitness Improvement per Generation')
    ax4.legend(fontsize=10, ncol=2)
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    # Save comparison plot
    output_path = os.path.join(output_dir, 'combined_comparison_all_trials.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved combined comparison plot: {output_path}")
    plt.close()


def create_combined_summary_table(all_data: Dict, output_dir: str):
    """Create a summary table of final results."""
    
    summary_data = []
    
    for folder_name, folder_data in all_data.items():
        for run_id, data in folder_data.items():
            if data['best_fitness']:
                trial_type = 'HPO' if 'HPO' in folder_name else 'Time Penalty'
                summary_data.append({
                    'Trial Type': trial_type,
                    'Folder': folder_name,
                    'Run': run_id,
                    'Final Generation': data['final_generation'],
                    'Final Best Fitness': data['best_fitness'][-1],
                    'Final Avg Fitness': data['avg_fitness'][-1],
                    'Final Avg Cost': data['avg_cost'][-1],
                    'Final Front Size': data['front0_size'][-1],
                    'Max Fitness': max(data['best_fitness']),
                    'Fitness Improvement': data['best_fitness'][-1] - data['best_fitness'][0] if len(data['best_fitness']) > 1 else 0
                })
    
    if summary_data:
        # Save as CSV
        import pandas as pd
        df = pd.DataFrame(summary_data)
        csv_path = os.path.join(output_dir, 'combined_summary.csv')
        df.to_csv(csv_path, index=False)
        print(f"Saved combined summary table: {csv_path}")
        
        # Print summary
        print("\n" + "="*100)
        print("COMBINED HPO AND TIME PENALTY TRIALS SUMMARY")
        print("="*100)
        print(df.to_string(index=False))
        print("="*100)


def main():
    parser = argparse.ArgumentParser(description='Plot combined HPO and time penalty trial fitness evolution')
    parser.add_argument('--folders', nargs='+', 
                       default=['hpo_ckpts_full_2', 'hpo_ckpts_2'],
                       help='Checkpoint folders to process')
    parser.add_argument('--output', default='combined_plots',
                       help='Output directory for plots')
    parser.add_argument('--summary', action='store_true',
                       help='Generate summary table')
    
    args = parser.parse_args()
    
    print("Combined HPO and Time Penalty Trials Fitness Plotter")
    print("="*60)
    print(f"Processing HPO folders: {args.folders}")
    print(f"Output directory: {args.output}")
    
    # Create plots
    plot_combined_evolution(args.folders, args.output)
    
    # Create summary if requested
    if args.summary:
        # Reload data for summary
        all_data = {}
        
        # Process HPO folders
        for folder in args.folders:
            if os.path.exists(folder):
                folder_data = {}
                checkpoint_files = [f for f in os.listdir(folder) if f.startswith('ckpt_') and f.endswith('.pkl')]
                for ckpt_file in checkpoint_files:
                    ckpt_path = os.path.join(folder, ckpt_file)
                    ckpt_data = load_checkpoint_data(ckpt_path)
                    if ckpt_data:
                        fitness_data = extract_fitness_history(ckpt_data)
                        if fitness_data:
                            run_id = ckpt_file.replace('ckpt_', '').replace('.pkl', '')
                            folder_data[run_id] = fitness_data
                if folder_data:
                    all_data[f"HPO_{folder}"] = folder_data
        
        
        create_combined_summary_table(all_data, args.output)
    
    print(f"\nPlotting complete! Check the '{args.output}' directory for results.")


if __name__ == "__main__":
    main()
