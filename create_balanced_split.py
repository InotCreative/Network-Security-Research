#!/usr/bin/env python3
"""
Create properly balanced train/test split from UNSW-NB15 data
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

def create_balanced_split():
    print("🔧 CREATING BALANCED TRAIN/TEST SPLIT")
    print("=" * 50)
    
    import os
    
    # Load files from provided paths
    print("📊 Loading original files...")
    
    # Check for files in multiple possible locations
    possible_paths = [
        '',  # Current directory
        '../Network Analysis/UNSW_NB15 Dataset/',  # From error message
        './UNSW_NB15 Dataset/',
        './data/',
        '../data/'
    ]
    
    train_files = ['UNSW_NB15_training-set.csv', 'training-set.csv', 'train.csv']
    test_files = ['UNSW_NB15_testing-set.csv', 'testing-set.csv', 'test.csv']
    
    train_df = None
    test_df = None
    found_train_path = None
    found_test_path = None
    
    # Search for training file
    for path in possible_paths:
        for train_file in train_files:
            full_path = os.path.join(path, train_file)
            if os.path.exists(full_path):
                train_df = pd.read_csv(full_path)
                found_train_path = full_path
                print(f"Found training file: {full_path}")
                break
        if train_df is not None:
            break
    
    # Search for test file
    for path in possible_paths:
        for test_file in test_files:
            full_path = os.path.join(path, test_file)
            if os.path.exists(full_path):
                test_df = pd.read_csv(full_path)
                found_test_path = full_path
                print(f"Found test file: {full_path}")
                break
        if test_df is not None:
            break
    
    if train_df is None or test_df is None:
        print("\n[ERROR] UNSW-NB15 dataset files not found!")
        print("Please ensure you have the UNSW-NB15 dataset files:")
        print("  - UNSW_NB15_training-set.csv")
        print("  - UNSW_NB15_testing-set.csv")
        print("\nSearched in these locations:")
        for path in possible_paths:
            search_path = path if path else "current directory"
            print(f"  - {search_path}")
        print("\nPlease download the UNSW-NB15 dataset and place the files in one of these locations.")
        raise FileNotFoundError("Required UNSW-NB15 dataset files not found")
    
    print(f"Training file: {train_df.shape}")
    print(f"Test file: {test_df.shape}")
    
    # Check label distributions
    target_col = 'label' if 'label' in train_df.columns else 'attack'
    
    train_dist = train_df[target_col].value_counts().sort_index()
    test_dist = test_df[target_col].value_counts().sort_index()
    
    print(f"\nOriginal distributions:")
    print(f"Training file - Normal: {train_dist[0]}, Attack: {train_dist[1]} ({train_dist[1]/(train_dist[0]+train_dist[1])*100:.1f}% attack)")
    print(f"Test file - Normal: {test_dist[0]}, Attack: {test_dist[1]} ({test_dist[1]/(test_dist[0]+test_dist[1])*100:.1f}% attack)")
    
    # Combine datasets
    print(f"\n🔄 Combining datasets...")
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    combined_dist = combined_df[target_col].value_counts().sort_index()
    
    print(f"Combined dataset: {combined_df.shape}")
    print(f"Combined - Normal: {combined_dist[0]}, Attack: {combined_dist[1]} ({combined_dist[1]/(combined_dist[0]+combined_dist[1])*100:.1f}% attack)")
    
    # Create stratified split based on data size
    print(f"\n✂️  Creating stratified split...")
    
    # Determine split ratio based on combined data size
    total_samples = len(combined_df)
    if total_samples < 10000:
        test_ratio = 0.2  # Smaller test set for small datasets
    elif total_samples < 100000:
        test_ratio = 0.25  # Medium test set for medium datasets  
    else:
        test_ratio = 0.3  # Larger test set for large datasets
    
    print(f"Using {test_ratio:.0%} for test set based on data size ({total_samples} samples)")
    
    train_new, test_new = train_test_split(
        combined_df, 
        test_size=test_ratio, 
        random_state=42, 
        stratify=combined_df[target_col]
    )
    
    # Check new distributions
    train_new_dist = train_new[target_col].value_counts().sort_index()
    test_new_dist = test_new[target_col].value_counts().sort_index()
    
    print(f"\nNew balanced distributions:")
    print(f"New training: {train_new.shape}")
    print(f"  Normal: {train_new_dist[0]}, Attack: {train_new_dist[1]} ({train_new_dist[1]/(train_new_dist[0]+train_new_dist[1])*100:.1f}% attack)")
    print(f"New test: {test_new.shape}")
    print(f"  Normal: {test_new_dist[0]}, Attack: {test_new_dist[1]} ({test_new_dist[1]/(test_new_dist[0]+test_new_dist[1])*100:.1f}% attack)")
    
    # Save balanced datasets
    print(f"\n💾 Saving balanced datasets...")
    train_new.to_csv('UNSW_balanced_train.csv', index=False)
    test_new.to_csv('UNSW_balanced_test.csv', index=False)
    
    print(f"✅ Created balanced datasets:")
    print(f"   📁 UNSW_balanced_train.csv ({len(train_new)} samples)")
    print(f"   📁 UNSW_balanced_test.csv ({len(test_new)} samples)")
    
    # Verify the split is good
    train_attack_ratio = train_new_dist[1] / (train_new_dist[0] + train_new_dist[1])
    test_attack_ratio = test_new_dist[1] / (test_new_dist[0] + test_new_dist[1])
    ratio_diff = abs(train_attack_ratio - test_attack_ratio)
    
    print(f"\n🎯 Quality Check:")
    print(f"   Train attack ratio: {train_attack_ratio:.3f}")
    print(f"   Test attack ratio: {test_attack_ratio:.3f}")
    print(f"   Difference: {ratio_diff:.3f}")
    
    if ratio_diff < 0.01:
        print(f"   ✅ Excellent balance (difference < 1%)")
    elif ratio_diff < 0.05:
        print(f"   ✅ Good balance (difference < 5%)")
    else:
        print(f"   ⚠️  Moderate balance (difference > 5%)")
    
    return 'UNSW_balanced_train.csv', 'UNSW_balanced_test.csv'

if __name__ == "__main__":
    train_file, test_file = create_balanced_split()
    
    print(f"\n🚀 Now run your experiment with balanced data:")
    print(f"python run_novel_ml.py --dataset {train_file} --test-dataset {test_file} --compare-baseline --analyze-components")