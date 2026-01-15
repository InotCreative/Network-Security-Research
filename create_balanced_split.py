#!/usr/bin/env python3
"""
Preprocess UNSW-NB15 Official Train/Test Split

This script uses the OFFICIAL UNSW-NB15 train/test split as-is (NO re-splitting).
All preprocessing statistics are computed from training data only to prevent data leakage.

Output Files:
- preprocessed_train.csv: Official training set with imputation + encoding
- preprocessed_test.csv: Official test set with imputation + encoding (using TRAIN statistics)

What's Included:
✅ Missing values imputed (using train statistics)
✅ Categorical features label-encoded (fit on train)
✅ Infinite values replaced with 0

What's NOT Included (done per-experiment):
❌ Feature engineering
❌ Scaling
❌ Feature selection
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os


def preprocess_official_split():
    """
    Preprocess the official UNSW-NB15 train/test split WITHOUT re-splitting.
    
    This prevents data leakage by:
    1. Using official split as-is (no concatenation)
    2. Computing imputation statistics from training data only
    3. Fitting label encoders on training data only
    """
    print("🔧 PREPROCESSING OFFICIAL UNSW-NB15 SPLIT")
    print("=" * 60)
    print("⚠️  Using OFFICIAL split - NO re-splitting (prevents data leakage)")
    print("=" * 60)
    
    # Check for files in multiple possible locations
    possible_paths = [
        '',  # Current directory
        '../Network Analysis/UNSW_NB15 Dataset/',
        './UNSW_NB15 Dataset/',
        './data/',
        '../data/'
    ]
    
    train_files = ['UNSW_NB15_training-set.csv', 'training-set.csv', 'train.csv']
    test_files = ['UNSW_NB15_testing-set.csv', 'testing-set.csv', 'test.csv']
    
    df_train = None
    df_test = None
    found_train_path = None
    found_test_path = None
    
    # Search for training file
    for path in possible_paths:
        for train_file in train_files:
            full_path = os.path.join(path, train_file)
            if os.path.exists(full_path):
                df_train = pd.read_csv(full_path)
                found_train_path = full_path
                print(f"✅ Found training file: {full_path}")
                break
        if df_train is not None:
            break
    
    # Search for test file
    for path in possible_paths:
        for test_file in test_files:
            full_path = os.path.join(path, test_file)
            if os.path.exists(full_path):
                df_test = pd.read_csv(full_path)
                found_test_path = full_path
                print(f"✅ Found test file: {full_path}")
                break
        if df_test is not None:
            break
    
    if df_train is None or df_test is None:
        print("\n❌ ERROR: UNSW-NB15 dataset files not found!")
        print("Please ensure you have the UNSW-NB15 dataset files:")
        print("  - UNSW_NB15_training-set.csv")
        print("  - UNSW_NB15_testing-set.csv")
        print("\nSearched in these locations:")
        for path in possible_paths:
            search_path = path if path else "current directory"
            print(f"  - {search_path}")
        print("\nPlease download the UNSW-NB15 dataset and place the files in one of these locations.")
        raise FileNotFoundError("Required UNSW-NB15 dataset files not found")
    
    print(f"\n📊 Dataset Sizes:")
    print(f"   Training: {df_train.shape[0]:,} samples, {df_train.shape[1]} features")
    print(f"   Test: {df_test.shape[0]:,} samples, {df_test.shape[1]} features")
    
    # Check label distributions
    target_col = 'label' if 'label' in df_train.columns else 'attack'
    
    train_dist = df_train[target_col].value_counts().sort_index()
    test_dist = df_test[target_col].value_counts().sort_index()
    
    print(f"\n📊 Official Split Distributions:")
    print(f"   Training - Normal: {train_dist[0]:,}, Attack: {train_dist[1]:,} ({train_dist[1]/(train_dist[0]+train_dist[1])*100:.1f}% attack)")
    print(f"   Test - Normal: {test_dist[0]:,}, Attack: {test_dist[1]:,} ({test_dist[1]/(test_dist[0]+test_dist[1])*100:.1f}% attack)")
    
    # =========================================================================
    # IMPUTATION: Compute statistics from TRAINING data only
    # =========================================================================
    print(f"\n🔧 IMPUTATION (using TRAINING statistics only)")
    print("-" * 40)
    
    # Identify numeric columns
    numeric_cols = df_train.select_dtypes(include=[np.number]).columns.tolist()
    
    # Compute medians from training data only
    train_medians = df_train[numeric_cols].median()
    
    # Count missing values before imputation
    train_missing = df_train[numeric_cols].isnull().sum().sum()
    test_missing = df_test[numeric_cols].isnull().sum().sum()
    
    print(f"   Missing values - Train: {train_missing:,}, Test: {test_missing:,}")
    
    # Apply imputation using TRAINING medians to BOTH sets
    df_train[numeric_cols] = df_train[numeric_cols].fillna(train_medians)
    df_test[numeric_cols] = df_test[numeric_cols].fillna(train_medians)  # Uses TRAIN statistics!
    
    print(f"   ✅ Imputed using training medians")
    
    # =========================================================================
    # HANDLE INFINITE VALUES
    # =========================================================================
    print(f"\n🔧 HANDLING INFINITE VALUES")
    print("-" * 40)
    
    train_inf = np.isinf(df_train[numeric_cols]).sum().sum()
    test_inf = np.isinf(df_test[numeric_cols]).sum().sum()
    
    print(f"   Infinite values - Train: {train_inf:,}, Test: {test_inf:,}")
    
    df_train[numeric_cols] = df_train[numeric_cols].replace([np.inf, -np.inf], 0)
    df_test[numeric_cols] = df_test[numeric_cols].replace([np.inf, -np.inf], 0)
    
    print(f"   ✅ Replaced infinite values with 0")
    
    # =========================================================================
    # LABEL ENCODING: Fit on TRAINING data only
    # =========================================================================
    print(f"\n🔧 LABEL ENCODING (fit on TRAINING only)")
    print("-" * 40)
    
    categorical_cols = ['proto', 'service', 'state']
    label_encoders = {}
    
    for col in categorical_cols:
        if col in df_train.columns:
            le = LabelEncoder()
            
            # Fit on training data only
            le.fit(df_train[col].astype(str))
            
            # Transform training data
            df_train[col] = le.transform(df_train[col].astype(str))
            
            # Transform test data - handle unseen categories
            test_values = df_test[col].astype(str)
            unseen_mask = ~test_values.isin(le.classes_)
            unseen_count = unseen_mask.sum()
            
            if unseen_count > 0:
                print(f"   ⚠️  {col}: {unseen_count} unseen categories in test set → mapped to -1 (unknown)")
                # Map known categories normally, unseen get -1
                test_values_mapped = test_values.copy()
                test_values_mapped[unseen_mask] = le.classes_[0]  # Temporarily map to known class for transform
                df_test[col] = le.transform(test_values_mapped)
                df_test.loc[unseen_mask, col] = -1  # Then set unseen to -1 (tree-friendly unknown marker)
            else:
                df_test[col] = le.transform(test_values)
            
            label_encoders[col] = le
            print(f"   ✅ {col}: {len(le.classes_)} categories encoded (unseen → -1)")
    
    # =========================================================================
    # SAVE PREPROCESSED FILES
    # =========================================================================
    print(f"\n💾 SAVING PREPROCESSED FILES")
    print("-" * 40)
    
    df_train.to_csv('preprocessed_train.csv', index=False)
    df_test.to_csv('preprocessed_test.csv', index=False)
    
    print(f"   ✅ preprocessed_train.csv ({len(df_train):,} samples)")
    print(f"   ✅ preprocessed_test.csv ({len(df_test):,} samples)")
    
    # =========================================================================
    # VALIDATION SUMMARY
    # =========================================================================
    print(f"\n✅ PREPROCESSING COMPLETE")
    print("=" * 60)
    print("📋 Validation Checklist:")
    print("   ✅ Training CSV loaded independently")
    print("   ✅ Test CSV loaded independently (NO concatenation)")
    print("   ✅ Imputation statistics computed from train only")
    print("   ✅ Label encoders fit on train only")
    print("   ✅ Test set preserved for final evaluation only")
    print("   ✅ NO train_test_split() called on combined data")
    
    print(f"\n📊 Final Statistics:")
    final_train_dist = df_train[target_col].value_counts().sort_index()
    final_test_dist = df_test[target_col].value_counts().sort_index()
    
    print(f"   Train - Normal: {final_train_dist[0]:,}, Attack: {final_train_dist[1]:,}")
    print(f"   Test - Normal: {final_test_dist[0]:,}, Attack: {final_test_dist[1]:,}")
    
    return 'preprocessed_train.csv', 'preprocessed_test.csv'


if __name__ == "__main__":
    train_file, test_file = preprocess_official_split()
    
    print(f"\n🚀 Now run your experiment with preprocessed data:")
    print(f"   python run_novel_ml.py --dataset {train_file} --test-dataset {test_file} --compare-baseline --analyze-components")
