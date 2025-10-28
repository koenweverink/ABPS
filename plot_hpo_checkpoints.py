#!/usr/bin/env python3
"""
Plot HPO checkpoint data showing generations vs fitness evolution.

This script loads checkpoint files from HPO folders and creates plots showing
how fitness metrics evolve over generations for different hyperparameter
optimization runs.
"""

import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse


def load_checkpoint_data(checkpoint_path: str) -> Dict:
    """Load checkpoint data from a pickle file."""
    try:
        with open(checkpoint_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading {checkpoint_path}: {e}")
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


def plot_fitness_evolution(checkpoint_folders: List[str], output_dir: str = "hpo_plots"):
    """Plot fitness evolution for all checkpoint folders."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect data from all checkpoint folders
    all_data = {}
    
    for folder in checkpoint_folders:
        if not os.path.exists(folder):
            print(f"Warning: Folder {folder} does not exist")
            continue
            
        print(f"Processing folder: {folder}")
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
                    print(f"  Loaded {run_id}: {len(fitness_data['generations'])} generations")
        
        if folder_data:
            all_data[folder] = folder_data
    
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
        fig.suptitle(f'Fitness Evolution - {folder_name}', fontsize=16)
        
        # Plot 1: Best Fitness over Generations
        ax1 = axes[0, 0]
        for run_id, data in folder_data.items():
            normalized_fitness = [f / 47.6 * 100 for f in data['best_fitness']]
            ax1.plot(data['generations'], normalized_fitness, 
                    marker='o', label=run_id, alpha=1)
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Best Fitness')
        ax1.set_title('Best Fitness Evolution')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Average Fitness over Generations
        ax2 = axes[0, 1]
        for run_id, data in folder_data.items():
            final_fitness = [f * 100 for f in data['avg_fitness']]
            ax2.plot(data['generations'], final_fitness, 
                    marker='s', label=run_id, alpha=1)
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Average Fitness')
        ax2.set_title('Average Fitness Evolution')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Average Cost over Generations
        ax3 = axes[1, 0]
        for run_id, data in folder_data.items():
            ax3.plot(data['generations'], data['avg_cost'], 
                    marker='^', label=run_id, alpha=1)
        ax3.set_xlabel('Generation')
        ax3.set_ylabel('Average Cost')
        ax3.set_title('Average Cost Evolution')
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Pareto Front Size over Generations
        ax4 = axes[1, 1]
        for run_id, data in folder_data.items():
            ax4.plot(data['generations'], data['front0_size'], 
                    marker='d', label=run_id, alpha=1)
        ax4.set_xlabel('Generation')
        ax4.set_ylabel('Pareto Front Size')
        ax4.set_title('Pareto Front Size Evolution')
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        safe_folder_name = folder_name.replace('/', '_').replace('\\', '_')
        output_path = os.path.join(output_dir, f'fitness_evolution_{safe_folder_name}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot: {output_path}")
        plt.close()


def create_comparison_plots(all_data: Dict, output_dir: str):
    """Create comparison plots across all folders."""
    
    # Collect all runs for comparison
    all_runs = {}
    for folder_name, folder_data in all_data.items():
        for run_id, data in folder_data.items():
            full_id = f"{folder_name}_{run_id}"
            all_runs[full_id] = data
    
    if not all_runs:
        print("No data to plot")
        return
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    fig.suptitle('HPO Fitness Evolution Comparison', fontsize=16)
    
    # Plot 1: Best Fitness Comparison
    ax1 = axes[0, 0]
    colors = plt.cm.tab20(np.linspace(0, 1, len(all_runs)))
    for i, (run_id, data) in enumerate(all_runs.items()):
        normalized_fitness = [f / 47.6 * 100 for f in data['best_fitness']]
        ax1.plot(data['generations'], normalized_fitness, 
                marker='o', label=run_id, alpha=1, color=colors[i])
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Best Fitness')
    ax1.set_title('Best Fitness Evolution - All Runs')
    ax1.legend(fontsize=6, ncol=2)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Average Fitness Comparison
    ax2 = axes[0, 1]
    for i, (run_id, data) in enumerate(all_runs.items()):
        ax2.plot(data['generations'], data['avg_fitness'], 
                marker='s', label=run_id, alpha=1, color=colors[i])
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Average Fitness')
    ax2.set_title('Average Fitness Evolution - All Runs')
    ax2.legend(fontsize=6, ncol=2)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Final Fitness Comparison (bar chart)
    ax3 = axes[1, 0]
    final_fitness = []
    run_names = []
    for run_id, data in all_runs.items():
        if data['best_fitness']:
            normalized_final = data['best_fitness'][-1] / 47.6 * 100
            final_fitness.append(normalized_final)
            run_names.append(run_id)
    
    if final_fitness:
        bars = ax3.bar(range(len(run_names)), final_fitness, alpha=1)
        ax3.set_xlabel('Run')
        ax3.set_ylabel('Final Best Fitness')
        ax3.set_title('Final Best Fitness Comparison')
        ax3.set_xticks(range(len(run_names)))
        ax3.set_xticklabels(run_names, rotation=45, ha='right')
        ax3.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, final_fitness)):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{value:.2f}', ha='center', va='bottom', fontsize=10)
    
    # Plot 4: Convergence Analysis
    ax4 = axes[1, 1]
    for i, (run_id, data) in enumerate(all_runs.items()):
        if len(data['best_fitness']) > 1:
            # Calculate improvement over time using normalized values
            normalized_fitness = [f / 47.6 * 100 for f in data['best_fitness']]
            improvements = np.diff(normalized_fitness)
            ax4.plot(data['generations'][1:], improvements, 
                    marker='o', label=run_id, alpha=1, color=colors[i])
    ax4.set_xlabel('Generation')
    ax4.set_ylabel('Fitness Improvement')
    ax4.set_title('Fitness Improvement per Generation')
    ax4.legend(fontsize=6, ncol=2)
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    # Save comparison plot
    output_path = os.path.join(output_dir, 'fitness_comparison_all_runs.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot: {output_path}")
    plt.close()


def create_summary_table(all_data: Dict, output_dir: str):
    """Create a summary table of final results."""
    
    summary_data = []
    
    for folder_name, folder_data in all_data.items():
        for run_id, data in folder_data.items():
            if data['best_fitness']:
                summary_data.append({
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
        csv_path = os.path.join(output_dir, 'hpo_summary.csv')
        df.to_csv(csv_path, index=False)
        print(f"Saved summary table: {csv_path}")
        
        # Print summary
        print("\n" + "="*80)
        print("HPO SUMMARY")
        print("="*80)
        print(df.to_string(index=False))
        print("="*80)


def main():
    parser = argparse.ArgumentParser(description='Plot HPO checkpoint fitness evolution')
    # parser.add_argument('--folders', nargs='+', 
    #                    default=['hpo_ckpts_full_2', 'hpo_ckpts_full', 'hpo_ckpts_2', 'hpo_ckpts'],
    #                    help='Checkpoint folders to process')
    parser.add_argument('--folders', nargs='+', 
                       default=['hpo_ckpts_2'],
                       help='Checkpoint folders to process')
    parser.add_argument('--output', default='hpo_plots',
                       help='Output directory for plots')
    parser.add_argument('--summary', action='store_true',
                       help='Generate summary table')
    
    args = parser.parse_args()
    
    print("HPO Checkpoint Fitness Plotter")
    print("="*40)
    print(f"Processing folders: {args.folders}")
    print(f"Output directory: {args.output}")
    
    # Create plots
    plot_fitness_evolution(args.folders, args.output)
    
    # Create summary if requested
    if args.summary:
        # Reload data for summary
        all_data = {}
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
                    all_data[folder] = folder_data
        create_summary_table(all_data, args.output)
    
    print(f"\nPlotting complete! Check the '{args.output}' directory for results.")


if __name__ == "__main__":
    main()
