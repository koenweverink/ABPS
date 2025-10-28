#!/usr/bin/env python3
"""
Plot time penalty trial data showing generations vs fitness evolution.

This script loads history JSON files from rerun_outputs folder and creates plots showing
how fitness metrics evolve over generations for different time penalty trials.
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse


def _format_trial_label(trial_id: str) -> str:
    """Return a concise legend label for a time penalty trial."""
    label = trial_id
    if label.startswith('ckpt_'):
        label = label[len('ckpt_'):]
    if not label.startswith('tp_'):
        label = f'tp_{label}'
    return label


def _ensure_unique_label(label: str, existing: Dict[str, Dict]) -> str:
    """Ensure legend labels remain unique when aggregating trials."""
    if label not in existing:
        return label

    index = 2
    while f"{label}_{index}" in existing:
        index += 1
    return f"{label}_{index}"


def load_history_data(history_path: str) -> Dict:
    """Load history data from a JSON file."""
    try:
        with open(history_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {history_path}: {e}")
        return None


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


def plot_time_penalty_evolution(output_dir: str = "time_penalty_plots"):
    """Plot fitness evolution for time penalty trials."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect data from rerun_outputs folder
    all_data = {}
    
    if not os.path.exists('rerun_outputs'):
        print("Warning: rerun_outputs folder does not exist")
        return
    
    print("Processing rerun_outputs folder")
    folder_data = {}
    
    # Find all history JSON files in the folder
    history_files = [f for f in os.listdir('rerun_outputs') if f.endswith('_history.json')]
    
    for history_file in history_files:
        history_path = os.path.join('rerun_outputs', history_file)
        history_data = load_history_data(history_path)
        
        if history_data:
            fitness_data = extract_fitness_from_history(history_data)
            if fitness_data:
                # Extract trial identifier from filename
                trial_id = history_file.replace('_history.json', '')
                folder_data[trial_id] = fitness_data
                print(f"  Loaded {trial_id}: {len(fitness_data['generations'])} generations")
    
    if folder_data:
        all_data['time_penalty_trials'] = folder_data
    
    # Create plots
    create_fitness_plots(all_data, output_dir)
    create_comparison_plots(all_data, output_dir)


def create_fitness_plots(all_data: Dict, output_dir: str):
    """Create individual fitness plots for each folder."""
    
    for folder_name, folder_data in all_data.items():
        if not folder_data:
            continue
            
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Time Penalty Trials - Fitness Evolution', fontsize=16)
        
        # Plot 1: Best Fitness over Generations
        ax1 = axes[0, 0]
        for trial_id, data in folder_data.items():
            final_fitness = [f / 47.6 * 100 for f in data['best_fitness']]
            label = _format_trial_label(trial_id)
            ax1.plot(
                data['generations'],
                final_fitness,
                marker='o',
                label=label,
                alpha=1,
                linewidth=2,
            )
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Best Fitness')
        ax1.set_title('Best Fitness Evolution')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Average Fitness over Generations
        ax2 = axes[0, 1]
        for trial_id, data in folder_data.items():
            final_fitness = [f / 47.6 * 100 for f in data['avg_fitness']]
            label = _format_trial_label(trial_id)
            ax2.plot(
                data['generations'],
                final_fitness,
                marker='s',
                label=label,
                alpha=1,
                linewidth=2,
            )
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Average Fitness')
        ax2.set_title('Average Fitness Evolution')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Average Cost over Generations
        ax3 = axes[1, 0]
        for trial_id, data in folder_data.items():
            label = _format_trial_label(trial_id)
            ax3.plot(
                data['generations'],
                data['avg_cost'],
                marker='^',
                label=label,
                alpha=1,
                linewidth=2,
            )
        ax3.set_xlabel('Generation')
        ax3.set_ylabel('Average Cost')
        ax3.set_title('Average Cost Evolution')
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Pareto Front Size over Generations
        ax4 = axes[1, 1]
        for trial_id, data in folder_data.items():
            label = _format_trial_label(trial_id)
            ax4.plot(
                data['generations'],
                data['front0_size'],
                marker='d',
                label=label,
                alpha=1,
                linewidth=2,
            )
        ax4.set_xlabel('Generation')
        ax4.set_ylabel('Pareto Front Size')
        ax4.set_title('Pareto Front Size Evolution')
        ax4.legend(fontsize=10, ncol=2)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        output_path = os.path.join(output_dir, f'time_penalty_fitness_evolution.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot: {output_path}")
        plt.close()


def create_comparison_plots(all_data: Dict, output_dir: str):
    """Create comparison plots across all trials."""
    
    # Collect all runs for comparison
    all_runs = {}
    for folder_name, folder_data in all_data.items():
        for trial_id, data in folder_data.items():
            label = _format_trial_label(trial_id)
            label = _ensure_unique_label(label, all_runs)
            all_runs[label] = data
    
    if not all_runs:
        print("No data to plot")
        return
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    fig.suptitle('Time Penalty Trials - Fitness Evolution Comparison', fontsize=16)
    
    # Plot 1: Best Fitness Comparison
    ax1 = axes[0, 0]
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_runs)))
    for i, (trial_id, data) in enumerate(all_runs.items()):
        final_fitness = [f / 47.6 * 100 for f in data['best_fitness']]
        ax1.plot(
            data['generations'],
            final_fitness,
            marker='o',
            label=trial_id,
            alpha=0.8,
            color=colors[i],
            linewidth=2,
        )
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Best Fitness')
    ax1.set_title('Best Fitness Evolution - All Trials')
    ax1.legend(fontsize=10, ncol=2)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Average Fitness Comparison
    ax2 = axes[0, 1]
    for i, (trial_id, data) in enumerate(all_runs.items()):
        final_fitness = [f / 47.6 * 100 for f in data['avg_fitness']]
        ax2.plot(
            data['generations'],
            final_fitness,
            marker='s',
            label=trial_id,
            alpha=0.8,
            color=colors[i],
            linewidth=2,
        )
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Average Fitness')
    ax2.set_title('Average Fitness Evolution - All Trials')
    ax2.legend(fontsize=10, ncol=2)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Final Fitness Comparison (bar chart)
    ax3 = axes[1, 0]
    final_fitness = []
    trial_names = []
    for trial_id, data in all_runs.items():
        if data['best_fitness']:
            final_fitness.append(data['best_fitness'][-1])
            trial_names.append(trial_id)
    
    if final_fitness:
        bars = ax3.bar(range(len(trial_names)), final_fitness, alpha=1)
        ax3.set_xlabel('Trial')
        ax3.set_ylabel('Final Best Fitness')
        ax3.set_title('Final Best Fitness Comparison')
        ax3.set_xticks(range(len(trial_names)))
        ax3.set_xticklabels(trial_names, rotation=45, ha='right', fontsize=10)
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
    for i, (trial_id, data) in enumerate(all_runs.items()):
        if len(data['best_fitness']) > 1:
            # Calculate improvement over time
            improvements = np.diff(data['best_fitness'])
            ax4.plot(
                data['generations'][1:],
                improvements,
                marker='o',
                label=trial_id,
                alpha=0.8,
                color=colors[i],
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
    output_path = os.path.join(output_dir, 'time_penalty_comparison_all_trials.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot: {output_path}")
    plt.close()


def create_summary_table(all_data: Dict, output_dir: str):
    """Create a summary table of final results."""
    
    summary_data = []
    
    for folder_name, folder_data in all_data.items():
        for trial_id, data in folder_data.items():
            if data['best_fitness']:
                summary_data.append({
                    'Trial': trial_id,
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
        csv_path = os.path.join(output_dir, 'time_penalty_summary.csv')
        df.to_csv(csv_path, index=False)
        print(f"Saved summary table: {csv_path}")
        
        # Print summary
        print("\n" + "="*80)
        print("TIME PENALTY TRIALS SUMMARY")
        print("="*80)
        print(df.to_string(index=False))
        print("="*80)


def main():
    parser = argparse.ArgumentParser(description='Plot time penalty trial fitness evolution')
    parser.add_argument('--output', default='time_penalty_plots',
                       help='Output directory for plots')
    parser.add_argument('--summary', action='store_true',
                       help='Generate summary table')
    
    args = parser.parse_args()
    
    print("Time Penalty Trials Fitness Plotter")
    print("="*40)
    print(f"Output directory: {args.output}")
    
    # Create plots
    plot_time_penalty_evolution(args.output)
    
    # Create summary if requested
    if args.summary:
        # Reload data for summary
        all_data = {}
        if os.path.exists('rerun_outputs'):
            folder_data = {}
            history_files = [f for f in os.listdir('rerun_outputs') if f.endswith('_history.json')]
            for history_file in history_files:
                history_path = os.path.join('rerun_outputs', history_file)
                history_data = load_history_data(history_path)
                if history_data:
                    fitness_data = extract_fitness_from_history(history_data)
                    if fitness_data:
                        trial_id = history_file.replace('_history.json', '')
                        folder_data[trial_id] = fitness_data
            if folder_data:
                all_data['time_penalty_trials'] = folder_data
        create_summary_table(all_data, args.output)
    
    print(f"\nPlotting complete! Check the '{args.output}' directory for results.")


if __name__ == "__main__":
    main()
