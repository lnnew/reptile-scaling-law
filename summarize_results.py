#!/usr/bin/env python3
"""
Summarize Reptile Scaling Law Experiments Results
"""
import json
import os
from pathlib import Path
import pandas as pd

def load_results(results_dir='results'):
    """Load all experiment results"""
    data = []
    
    for ntasks_dir in Path(results_dir).glob('ntasks_*_gpu*'):
        # Extract N_tasks from directory name
        ntasks_str = ntasks_dir.name.split('_')[1]
        try:
            n_tasks = int(ntasks_str)
        except:
            continue
            
        # Load final results
        final_result_path = ntasks_dir / f'ntasks_{n_tasks}_seed_100' / 'final_result.json'
        if not final_result_path.exists():
            continue
            
        with open(final_result_path, 'r') as f:
            result = json.load(f)
            
        # Load baseline results
        baseline_path = ntasks_dir / 'baseline_results.json'
        if baseline_path.exists():
            with open(baseline_path, 'r') as f:
                baseline = json.load(f)
        else:
            baseline = {}
        
        # Extract key metrics
        data.append({
            'N_tasks': n_tasks,
            'Meta_Test_Loss': result.get('meta_test_loss', None),
            'Meta_Test_Acc': result.get('meta_test_acc', None),
            'Zero_Shot_Loss': baseline.get('zero_shot', {}).get('loss', None),
            'Zero_Shot_Acc': baseline.get('zero_shot', {}).get('acc', None),
            'No_Meta_Loss': baseline.get('no_meta_learning', {}).get('loss', None),
            'No_Meta_Acc': baseline.get('no_meta_learning', {}).get('acc', None),
            'Final_Train_Loss': result.get('final_meta_train_loss', None),
            'GPU': ntasks_dir.name.split('_gpu')[1],
        })
    
    return pd.DataFrame(data).sort_values('N_tasks')

def print_summary(df):
    """Print formatted summary"""
    print("="*80)
    print("REPTILE SCALING LAW EXPERIMENTS - FINAL RESULTS")
    print("="*80)
    print()
    
    print("Configuration:")
    print("  Model: TinyLlama-1.1B-Chat-v1.0")
    print("  Meta-steps: 2000 (optimized from 10000)")
    print("  Inner steps (k_inner): 3 (optimized from 5)")
    print("  Meta batch size: 2")
    print("  Inner batch size: 16")
    print("="*80)
    print()
    
    # Performance Summary
    print("PERFORMANCE SUMMARY:")
    print("-"*80)
    print(f"{'N_tasks':<10} {'Meta-Test':<20} {'Zero-Shot':<20} {'No-Meta':<20}")
    print(f"{'':10} {'Loss':<10} {'Acc':<10} {'Loss':<10} {'Acc':<10} {'Loss':<10} {'Acc':<10}")
    print("-"*80)
    
    for _, row in df.iterrows():
        def fmt(val):
            return f"{val:.4f}" if val is not None else "N/A"
        
        print(f"{row['N_tasks']:<10} "
              f"{fmt(row['Meta_Test_Loss']):<10} {fmt(row['Meta_Test_Acc']):<10} "
              f"{fmt(row['Zero_Shot_Loss']):<10} {fmt(row['Zero_Shot_Acc']):<10} "
              f"{fmt(row['No_Meta_Loss']):<10} {fmt(row['No_Meta_Acc']):<10}")
    
    print("-"*80)
    print()
    
    # Improvement Analysis
    print("IMPROVEMENT ANALYSIS (Meta-Learning vs Baselines):")
    print("-"*80)
    print(f"{'N_tasks':<10} {'vs Zero-Shot':<25} {'vs No-Meta':<25}")
    print(f"{'':10} {'Acc Gain':<12} {'Loss Reduc':<12} {'Acc Gain':<12} {'Loss Reduc':<12}")
    print("-"*80)
    
    for _, row in df.iterrows():
        if row['Meta_Test_Acc'] is None or row['Zero_Shot_Acc'] is None or row['No_Meta_Acc'] is None:
            continue
            
        acc_gain_zs = (row['Meta_Test_Acc'] - row['Zero_Shot_Acc']) * 100
        loss_reduc_zs = ((row['Zero_Shot_Loss'] - row['Meta_Test_Loss']) / row['Zero_Shot_Loss']) * 100
        acc_gain_nm = (row['Meta_Test_Acc'] - row['No_Meta_Acc']) * 100
        loss_reduc_nm = ((row['No_Meta_Loss'] - row['Meta_Test_Loss']) / row['No_Meta_Loss']) * 100
        
        print(f"{row['N_tasks']:<10} "
              f"+{acc_gain_zs:>5.2f}%     {loss_reduc_zs:>5.1f}%       "
              f"+{acc_gain_nm:>5.2f}%     {loss_reduc_nm:>5.1f}%")
    
    print("-"*80)
    print()
    
    # Scaling Analysis
    print("SCALING ANALYSIS:")
    print("-"*80)
    best_idx = df['Meta_Test_Acc'].idxmax()
    best_row = df.loc[best_idx]
    
    print(f"Best Performance: N_tasks={best_row['N_tasks']}")
    print(f"  Meta-Test Accuracy: {best_row['Meta_Test_Acc']:.4f}")
    print(f"  Meta-Test Loss: {best_row['Meta_Test_Loss']:.4f}")
    print()
    
    # Performance gain from N_tasks=50 to best
    first_row = df.iloc[0]
    acc_improvement = (best_row['Meta_Test_Acc'] - first_row['Meta_Test_Acc']) * 100
    
    print(f"Performance Gain (N_tasks={first_row['N_tasks']} → {best_row['N_tasks']}):")
    print(f"  Accuracy: +{acc_improvement:.2f}%")
    print(f"  From {first_row['Meta_Test_Acc']:.4f} to {best_row['Meta_Test_Acc']:.4f}")
    
    print("="*80)
    print()
    
    return df

def save_csv(df, output_file='results_summary.csv'):
    """Save results to CSV"""
    df.to_csv(output_file, index=False)
    print(f"Results saved to: {output_file}")

if __name__ == '__main__':
    # Load and process results
    df = load_results()
    
    if len(df) == 0:
        print("No results found!")
        exit(1)
    
    # Print summary
    df_summary = print_summary(df)
    
    # Save to CSV
    save_csv(df_summary, 'results_summary.csv')
    
    print("\nAll experiments completed successfully!")
    print(f"Total experiments: {len(df)}")
    print(f"N_tasks values tested: {sorted(df['N_tasks'].tolist())}")
