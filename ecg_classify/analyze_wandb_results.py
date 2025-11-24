#!/usr/bin/env python3
"""
Script to fetch best test AUC from wandb experiments and create a bar graph.
Groups results by seed and averages across augmentation conditions.

Usage:
    python analyze_wandb_results.py --project ecg-classify --entity your_entity --output results.png
    
    # If entity is set in wandb config, you can omit it:
    python analyze_wandb_results.py --project ecg-classify --output results.png
    
    # Save results to CSV as well:
    python analyze_wandb_results.py --project ecg-classify --output results.png --csv results.csv

Requirements:
    - wandb API key must be set (wandb login)
    - wandb package installed
    - matplotlib and pandas installed
"""

import wandb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from collections import defaultdict

def fetch_best_test_auc(run):
    """
    Fetch the best (maximum) test AUC from a wandb run.
    
    Args:
        run: wandb run object
        
    Returns:
        Best test AUC value or None if not found
    """
    try:
        # Get all logged values for 'auc test' metric
        history = run.history(keys=['auc test'])
        if history.empty:
            return None
        
        # Get the maximum value (best AUC)
        best_auc = history['auc test'].max()
        return best_auc
    except Exception as e:
        print(f"Error fetching data for run {run.name}: {e}")
        return None

def fetch_experiments(project_name="ecg-classify", entity=None):
    """
    Fetch all experiments from wandb project.
    
    Args:
        project_name: Name of wandb project
        entity: wandb entity/username (optional)
        
    Returns:
        List of runs with their config and best test AUC
    """
    # Initialize wandb API
    api = wandb.Api()
    
    # Fetch all runs from the project
    runs = api.runs(f"{entity}/{project_name}" if entity else project_name)
    
    results = []
    for run in runs:
        try:
            # Get run configuration
            config = run.config
            
            # Extract relevant config values
            seed = config.get('seed', None)
            use_augmented = config.get('use_augmented', None)
            num_gen_samples = config.get('num_gen_samples', None)
            
            # Get best test AUC
            best_auc = fetch_best_test_auc(run)
            
            if best_auc is not None:
                results.append({
                    'run_name': run.name,
                    'run_id': run.id,
                    'seed': seed,
                    'use_augmented': use_augmented,
                    'num_gen_samples': num_gen_samples,
                    'best_test_auc': best_auc,
                    'state': run.state  # 'finished', 'running', 'crashed', etc.
                })
        except Exception as e:
            print(f"Error processing run {run.name}: {e}")
            continue
    
    return results

def analyze_results(results_df):
    """
    Analyze results and group by seed and augmentation condition.
    
    Args:
        results_df: DataFrame with experiment results
        
    Returns:
        Dictionary with aggregated results
    """
    # Filter only finished runs
    results_df = results_df[results_df['state'] == 'finished'].copy()
    
    # Separate augmented and non-augmented experiments
    no_aug = results_df[results_df['use_augmented'] == False].copy()
    aug = results_df[results_df['use_augmented'] == True].copy()
    
    # Group by seed and calculate statistics
    analysis = {
        'no_augmentation': {},
        'augmentation': {}
    }
    
    # Process non-augmented experiments
    if not no_aug.empty:
        for seed in no_aug['seed'].unique():
            seed_data = no_aug[no_aug['seed'] == seed]
            analysis['no_augmentation'][seed] = {
                'mean': seed_data['best_test_auc'].mean(),
                'std': seed_data['best_test_auc'].std(),
                'count': len(seed_data),
                'values': seed_data['best_test_auc'].tolist()
            }
    
    # Process augmented experiments
    if not aug.empty:
        for seed in aug['seed'].unique():
            seed_data = aug[aug['seed'] == seed]
            analysis['augmentation'][seed] = {
                'mean': seed_data['best_test_auc'].mean(),
                'std': seed_data['best_test_auc'].std(),
                'count': len(seed_data),
                'values': seed_data['best_test_auc'].tolist(),
                'num_gen_samples': seed_data['num_gen_samples'].tolist()
            }
    
    return analysis

def create_bar_graph(analysis, output_path='wandb_results_barplot.png', figsize=(10, 6)):
    """
    Create a bar graph comparing augmented vs non-augmented results by seed.
    
    Args:
        analysis: Dictionary with analyzed results
        output_path: Path to save the figure
        figsize: Figure size tuple
    """
    # Get all unique seeds
    all_seeds = set()
    if analysis['no_augmentation']:
        all_seeds.update(analysis['no_augmentation'].keys())
    if analysis['augmentation']:
        all_seeds.update(analysis['augmentation'].keys())
    all_seeds = sorted(all_seeds)
    
    if not all_seeds:
        print("No data to plot!")
        return
    
    # Check if we have any data
    has_no_aug = len(analysis['no_augmentation']) > 0
    has_aug = len(analysis['augmentation']) > 0
    
    if not has_no_aug and not has_aug:
        print("No valid data to plot!")
        return
    
    # Prepare data for plotting
    seeds = []
    no_aug_means = []
    no_aug_stds = []
    aug_means = []
    aug_stds = []
    
    for seed in all_seeds:
        seeds.append(f"Seed {seed}")
        
        # Non-augmented
        if seed in analysis['no_augmentation']:
            no_aug_means.append(analysis['no_augmentation'][seed]['mean'])
            no_aug_stds.append(analysis['no_augmentation'][seed]['std'])
        else:
            no_aug_means.append(None)  # Use None to skip plotting
            no_aug_stds.append(None)
        
        # Augmented
        if seed in analysis['augmentation']:
            aug_means.append(analysis['augmentation'][seed]['mean'])
            aug_stds.append(analysis['augmentation'][seed]['std'])
        else:
            aug_means.append(None)  # Use None to skip plotting
            aug_stds.append(None)
    
    # Create bar plot
    x = np.arange(len(seeds))
    
    # Adjust width based on number of conditions
    if has_no_aug and has_aug:
        width = 0.35
        offset1 = -width/2
        offset2 = width/2
    else:
        width = 0.6
        offset1 = 0 if has_no_aug else width/2
        offset2 = 0 if has_aug else -width/2
    
    fig, ax = plt.subplots(figsize=figsize)
    
    bars1 = None
    bars2 = None
    
    # Convert None to 0 for plotting (matplotlib doesn't handle None well)
    no_aug_means_plot = [m if m is not None else 0 for m in no_aug_means]
    no_aug_stds_plot = [s if s is not None else 0 for s in no_aug_stds]
    aug_means_plot = [m if m is not None else 0 for m in aug_means]
    aug_stds_plot = [s if s is not None else 0 for s in aug_stds]
    
    if has_no_aug:
        bars1 = ax.bar(x + offset1, no_aug_means_plot, width, yerr=no_aug_stds_plot, 
                       label='No Augmentation', alpha=0.8, capsize=5)
        # Hide bars where value is 0 (meaning no data)
        for i, (bar, val) in enumerate(zip(bars1, no_aug_means)):
            if val is None:
                bar.set_alpha(0)  # Make invisible
                bar.set_height(0)  # Set height to 0
    if has_aug:
        bars2 = ax.bar(x + offset2, aug_means_plot, width, yerr=aug_stds_plot,
                       label='With Augmentation', alpha=0.8, capsize=5)
        # Hide bars where value is 0 (meaning no data)
        for i, (bar, val) in enumerate(zip(bars2, aug_means)):
            if val is None:
                bar.set_alpha(0)  # Make invisible
                bar.set_height(0)  # Set height to 0
    
    # Add value labels on bars
    def add_value_labels(bars, values, stds):
        if bars is None:
            return
        for bar, val, std in zip(bars, values, stds):
            if val is not None and val > 0:
                height = bar.get_height()
                std_val = std if std is not None else 0
                ax.text(bar.get_x() + bar.get_width()/2., height + std_val + 0.01,
                       f'{val:.4f}',
                       ha='center', va='bottom', fontsize=9)
    
    if bars1 is not None:
        add_value_labels(bars1, no_aug_means, no_aug_stds)
    if bars2 is not None:
        add_value_labels(bars2, aug_means, aug_stds)
    
    # Customize plot
    ax.set_xlabel('Seed', fontsize=12, fontweight='bold')
    ax.set_ylabel('Best Test AUC', fontsize=12, fontweight='bold')
    ax.set_title('Best Test AUC: Augmented vs Non-Augmented by Seed', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(seeds)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    # Calculate y-axis limit
    all_means = []
    all_stds = []
    if has_no_aug:
        all_means.extend([m for m in no_aug_means if m is not None and m > 0])
        all_stds.extend([s for s in no_aug_stds if s is not None and s > 0])
    if has_aug:
        all_means.extend([m for m in aug_means if m is not None and m > 0])
        all_stds.extend([s for s in aug_stds if s is not None and s > 0])
    
    if all_means:
        max_val = max(all_means)
        max_std = max(all_stds) if all_stds else 0
        ax.set_ylim([0, min(max_val + max_std + 0.1, 1.1)])
    else:
        ax.set_ylim([0, 1.0])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Bar graph saved to {output_path}")
    plt.close()

def print_summary(results_df, analysis):
    """
    Print summary statistics.
    
    Args:
        results_df: DataFrame with all results
        analysis: Dictionary with analyzed results
    """
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    
    # Overall statistics
    print(f"\nTotal experiments fetched: {len(results_df)}")
    print(f"Finished experiments: {len(results_df[results_df['state'] == 'finished'])}")
    
    # Non-augmented experiments
    no_aug = results_df[results_df['use_augmented'] == False]
    if not no_aug.empty:
        print(f"\nNon-Augmented Experiments: {len(no_aug)}")
        for seed in sorted(no_aug['seed'].unique()):
            seed_data = no_aug[no_aug['seed'] == seed]
            print(f"  Seed {seed}: {len(seed_data)} experiment(s)")
            for _, row in seed_data.iterrows():
                print(f"    - {row['run_name']}: AUC = {row['best_test_auc']:.4f}")
            if seed in analysis['no_augmentation']:
                print(f"    Average AUC: {analysis['no_augmentation'][seed]['mean']:.4f} ± {analysis['no_augmentation'][seed]['std']:.4f}")
    
    # Augmented experiments
    aug = results_df[results_df['use_augmented'] == True]
    if not aug.empty:
        print(f"\nAugmented Experiments: {len(aug)}")
        for seed in sorted(aug['seed'].unique()):
            seed_data = aug[aug['seed'] == seed]
            print(f"  Seed {seed}: {len(seed_data)} experiment(s)")
            for _, row in seed_data.iterrows():
                gen_samples = row['num_gen_samples'] if pd.notna(row['num_gen_samples']) else 'N/A'
                print(f"    - {row['run_name']}: AUC = {row['best_test_auc']:.4f} (gen_samples: {gen_samples})")
            if seed in analysis['augmentation']:
                print(f"    Average AUC: {analysis['augmentation'][seed]['mean']:.4f} ± {analysis['augmentation'][seed]['std']:.4f}")
    
    # Overall averages
    print("\n" + "-"*80)
    print("OVERALL AVERAGES BY CONDITION:")
    print("-"*80)
    
    if analysis['no_augmentation']:
        no_aug_all_means = [v['mean'] for v in analysis['no_augmentation'].values()]
        print(f"No Augmentation: {np.mean(no_aug_all_means):.4f} ± {np.std(no_aug_all_means):.4f}")
    
    if analysis['augmentation']:
        aug_all_means = [v['mean'] for v in analysis['augmentation'].values()]
        print(f"With Augmentation: {np.mean(aug_all_means):.4f} ± {np.std(aug_all_means):.4f}")
    
    print("="*80 + "\n")

def main():
    parser = argparse.ArgumentParser(description='Analyze wandb experiment results and create bar graph')
    parser.add_argument('--project', type=str, default='ecg-classify',
                       help='wandb project name (default: ecg-classify)')
    parser.add_argument('--entity', type=str, default=None,
                       help='wandb entity/username (optional)')
    parser.add_argument('--output', type=str, default='wandb_results_barplot.png',
                       help='Output path for bar graph (default: wandb_results_barplot.png)')
    parser.add_argument('--csv', type=str, default=None,
                       help='Optional: Save results to CSV file')
    
    args = parser.parse_args()
    
    print("Fetching experiments from wandb...")
    print(f"Project: {args.project}")
    if args.entity:
        print(f"Entity: {args.entity}")
    print()
    
    # Fetch experiments
    results = fetch_experiments(project_name=args.project, entity=args.entity)
    
    if not results:
        print("No experiments found!")
        return
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save to CSV if requested
    if args.csv:
        results_df.to_csv(args.csv, index=False)
        print(f"Results saved to {args.csv}")
    
    # Analyze results
    analysis = analyze_results(results_df)
    
    # Print summary
    print_summary(results_df, analysis)
    
    # Create bar graph
    create_bar_graph(analysis, output_path=args.output)
    
    print("Analysis complete!")

if __name__ == "__main__":
    main()

