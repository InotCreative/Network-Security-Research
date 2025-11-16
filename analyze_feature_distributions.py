#!/usr/bin/env python3
"""
Analyze feature distributions between successful and failing BoT-IoT files
to understand why some generalize well and others don't
"""

import pandas as pd
import numpy as np
import os
import glob

def analyze_file_features(file_path):
    """Analyze feature statistics for a single file"""
    try:
        df = pd.read_csv(file_path)
        
        # Get numeric columns only
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        stats = {
            'filename': os.path.basename(file_path),
            'total_samples': len(df),
            'num_features': len(df.columns),
            'num_numeric_features': len(numeric_cols)
        }
        
        # Attack distribution
        if 'attack' in df.columns:
            attack_dist = df['attack'].value_counts()
            stats['normal_count'] = attack_dist.get(0, 0)
            stats['attack_count'] = attack_dist.get(1, 0) if 1 in attack_dist else sum(attack_dist) - attack_dist.get(0, 0)
            stats['attack_ratio'] = stats['attack_count'] / len(df) if len(df) > 0 else 0
        
        # Feature statistics
        if len(numeric_cols) > 0:
            stats['mean_of_means'] = df[numeric_cols].mean().mean()
            stats['mean_of_stds'] = df[numeric_cols].std().mean()
            stats['zero_variance_features'] = (df[numeric_cols].std() == 0).sum()
            stats['missing_values'] = df[numeric_cols].isnull().sum().sum()
            
            # Check for extreme values
            stats['has_inf'] = np.isinf(df[numeric_cols]).any().any()
            stats['max_value'] = df[numeric_cols].max().max()
            stats['min_value'] = df[numeric_cols].min().min()
        
        return stats
        
    except Exception as e:
        print(f"Error analyzing {file_path}: {e}")
        return None

def main():
    # Files that performed well (>75% accuracy)
    good_files = ['data_2.csv', 'data_3.csv', 'data_4.csv', 'data_5.csv']
    
    # Files that performed poorly (~50% accuracy)
    poor_files = ['data_1.csv', 'data_55.csv', 'data_69.csv', 'data_27.csv', 
                  'data_61.csv', 'data_44.csv', 'data_13.csv', 'data_64.csv']
    
    data_dir = '../../Downloads/archive'
    
    print("🔍 ANALYZING FEATURE DISTRIBUTIONS")
    print("=" * 80)
    
    # Analyze good files
    print("\n✅ GOOD PERFORMING FILES (>75% accuracy):")
    print("-" * 80)
    good_stats = []
    for filename in good_files:
        file_path = os.path.join(data_dir, filename)
        if os.path.exists(file_path):
            stats = analyze_file_features(file_path)
            if stats:
                good_stats.append(stats)
                print(f"\n{filename}:")
                print(f"  Samples: {stats['total_samples']:,}")
                print(f"  Normal: {stats.get('normal_count', 0):,}, Attack: {stats.get('attack_count', 0):,}")
                print(f"  Attack ratio: {stats.get('attack_ratio', 0):.4f}")
                print(f"  Zero variance features: {stats.get('zero_variance_features', 0)}")
                print(f"  Mean of means: {stats.get('mean_of_means', 0):.2f}")
                print(f"  Mean of stds: {stats.get('mean_of_stds', 0):.2f}")
    
    # Analyze poor files
    print("\n\n❌ POOR PERFORMING FILES (~50% accuracy):")
    print("-" * 80)
    poor_stats = []
    for filename in poor_files:
        file_path = os.path.join(data_dir, filename)
        if os.path.exists(file_path):
            stats = analyze_file_features(file_path)
            if stats:
                poor_stats.append(stats)
                print(f"\n{filename}:")
                print(f"  Samples: {stats['total_samples']:,}")
                print(f"  Normal: {stats.get('normal_count', 0):,}, Attack: {stats.get('attack_count', 0):,}")
                print(f"  Attack ratio: {stats.get('attack_ratio', 0):.4f}")
                print(f"  Zero variance features: {stats.get('zero_variance_features', 0)}")
                print(f"  Mean of means: {stats.get('mean_of_means', 0):.2f}")
                print(f"  Mean of stds: {stats.get('mean_of_stds', 0):.2f}")
    
    # Compare statistics
    if good_stats and poor_stats:
        print("\n\n📊 COMPARISON:")
        print("=" * 80)
        
        good_df = pd.DataFrame(good_stats)
        poor_df = pd.DataFrame(poor_stats)
        
        print("\nGood files average:")
        print(f"  Samples: {good_df['total_samples'].mean():.0f}")
        print(f"  Attack ratio: {good_df['attack_ratio'].mean():.4f}")
        print(f"  Zero variance features: {good_df['zero_variance_features'].mean():.1f}")
        print(f"  Mean of means: {good_df['mean_of_means'].mean():.2f}")
        print(f"  Mean of stds: {good_df['mean_of_stds'].mean():.2f}")
        
        print("\nPoor files average:")
        print(f"  Samples: {poor_df['total_samples'].mean():.0f}")
        print(f"  Attack ratio: {poor_df['attack_ratio'].mean():.4f}")
        print(f"  Zero variance features: {poor_df['zero_variance_features'].mean():.1f}")
        print(f"  Mean of means: {poor_df['mean_of_means'].mean():.2f}")
        print(f"  Mean of stds: {poor_df['mean_of_stds'].mean():.2f}")
        
        print("\n💡 KEY DIFFERENCES:")
        print("-" * 80)
        
        # Sample size difference
        sample_diff = good_df['total_samples'].mean() - poor_df['total_samples'].mean()
        print(f"Sample size difference: {sample_diff:+.0f}")
        
        # Attack ratio difference
        ratio_diff = good_df['attack_ratio'].mean() - poor_df['attack_ratio'].mean()
        print(f"Attack ratio difference: {ratio_diff:+.4f}")
        
        # Zero variance difference
        zv_diff = good_df['zero_variance_features'].mean() - poor_df['zero_variance_features'].mean()
        print(f"Zero variance features difference: {zv_diff:+.1f}")

if __name__ == '__main__':
    main()
