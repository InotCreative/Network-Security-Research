#!/usr/bin/env python3
"""
Feature Separability Analysis using Random Forest
Demonstrates that certain features in the dataset provide perfect or near-perfect class separation
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


def load_dataset(train_file, test_file):
    """Load train and test datasets from provided file paths"""
    print(f"📂 Loading training data: {train_file}")
    train_df = pd.read_csv(train_file)
    print(f"   Shape: {train_df.shape}")
    
    print(f"📂 Loading test data: {test_file}")
    test_df = pd.read_csv(test_file)
    print(f"   Shape: {test_df.shape}")
    
    return train_df, test_df


def preprocess_data(train_df, test_df):
    """Preprocess train and test datasets"""
    train_df = train_df.copy()
    test_df = test_df.copy()
    
    # Identify target column
    target_col = 'label' if 'label' in train_df.columns else 'attack'
    
    # Drop non-feature columns
    drop_cols = ['id', 'attack_cat'] if 'attack_cat' in train_df.columns else ['id']
    drop_cols = [c for c in drop_cols if c in train_df.columns]
    
    if drop_cols:
        train_df = train_df.drop(columns=drop_cols)
        test_df = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns])
    
    # Encode categorical columns (fit on train, transform both)
    categorical_cols = train_df.select_dtypes(include=['object']).columns.tolist()
    if target_col in categorical_cols:
        categorical_cols.remove(target_col)
    
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        # Fit on combined unique values to handle unseen categories
        all_values = pd.concat([train_df[col], test_df[col]]).astype(str).unique()
        le.fit(all_values)
        train_df[col] = le.transform(train_df[col].astype(str))
        test_df[col] = le.transform(test_df[col].astype(str))
        label_encoders[col] = le
    
    # Separate features and target
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]
    
    # Handle any remaining NaN values
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)
    
    return X_train, y_train, X_test, y_test, label_encoders


def analyze_single_feature_separability(X_train, y_train, X_test, y_test, feature_name):
    """Analyze how well a single feature separates classes"""
    # Use unconstrained RF to find true separability potential
    rf = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42, n_jobs=-1)
    
    X_train_single = X_train[[feature_name]].values
    X_test_single = X_test[[feature_name]].values
    
    rf.fit(X_train_single, y_train)
    y_pred = rf.predict(X_test_single)
    accuracy = accuracy_score(y_test, y_pred)
    
    return accuracy


def find_perfectly_separable_features(X_train, y_train, X_test, y_test, threshold=0.95):
    """Find features that provide near-perfect class separation"""
    print("\n🔍 ANALYZING SINGLE-FEATURE SEPARABILITY")
    print("=" * 60)
    print(f"Testing each feature individually with Random Forest (100 trees, unlimited depth)")
    print(f"Threshold for 'perfect' separability: {threshold*100:.0f}%\n")
    
    results = []
    
    for feature in X_train.columns:
        accuracy = analyze_single_feature_separability(X_train, y_train, X_test, y_test, feature)
        results.append({'feature': feature, 'accuracy': accuracy})
        
        if accuracy >= threshold:
            print(f"   🎯 {feature}: {accuracy*100:.2f}% (PERFECT SEPARABILITY)")
        elif accuracy >= 0.85:
            print(f"   ✅ {feature}: {accuracy*100:.2f}% (High separability)")
    
    results_df = pd.DataFrame(results).sort_values('accuracy', ascending=False)
    
    perfect_features = results_df[results_df['accuracy'] >= threshold]
    high_sep_features = results_df[(results_df['accuracy'] >= 0.85) & (results_df['accuracy'] < threshold)]
    
    print(f"\n📊 SUMMARY:")
    print(f"   Features with perfect separability (≥{threshold*100:.0f}%): {len(perfect_features)}")
    print(f"   Features with high separability (85-{threshold*100:.0f}%): {len(high_sep_features)}")
    
    return results_df


def demonstrate_perfect_separation(X_train, y_train, X_test, y_test, top_n):
    """Demonstrate perfect separation using top features"""
    print("\n\n🎯 DEMONSTRATING PERFECT FEATURE SEPARATION")
    print("=" * 60)
    
    # Get feature importances from full RF model
    rf_full = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_full.fit(X_train, y_train)
    
    # Get ALL features ranked by importance
    importances = pd.DataFrame({
        'feature': X_train.columns,
        'importance': rf_full.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nAll {len(importances)} features ranked by importance:")
    for _, row in importances.iterrows():
        print(f"   {row['feature']}: {row['importance']*100:.2f}%")
    
    # Test with all features (same as full model)
    y_pred_full = rf_full.predict(X_test)
    accuracy_full = accuracy_score(y_test, y_pred_full)
    print(f"\n🧪 Accuracy with ALL {top_n} features: {accuracy_full*100:.2f}%")
    
    if accuracy_full >= 0.99:
        print(f"\n   ⚠️  WARNING: Near-perfect accuracy!")
        print(f"   This indicates potential data leakage or trivially separable classes.")
    
    return importances, accuracy_full, accuracy_full


def test_minimal_tree_depth(X_train, y_train, X_test, y_test):
    """Test if perfect separation is achievable with minimal tree depth"""
    print("\n\n🌳 TESTING MINIMAL TREE DEPTH FOR PERFECT SEPARATION")
    print("=" * 60)
    
    results = []
    for depth in [1, 2, 3, 5, 10, None]:
        rf = RandomForestClassifier(n_estimators=50, max_depth=depth, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        depth_str = str(depth) if depth else "unlimited"
        results.append({'depth': depth_str, 'accuracy': accuracy})
        
        marker = "🎯" if accuracy >= 0.99 else "✅" if accuracy >= 0.95 else "📊"
        print(f"   {marker} Depth={depth_str:>10}: {accuracy*100:.2f}%")
    
    # Check if depth=1 achieves high accuracy
    depth_1_acc = results[0]['accuracy']
    if depth_1_acc >= 0.90:
        print(f"\n   ⚠️  CRITICAL: {depth_1_acc*100:.1f}% accuracy with depth=1!")
        print(f"   This means a SINGLE decision split can nearly perfectly separate classes.")
        print(f"   Strong indicator of trivial separability or data leakage.")
    
    return results


def visualize_separability(X_train, y_train, importances, output_file='feature_separability.png'):
    """Create visualization of feature separability"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Top feature importances
    ax1 = axes[0, 0]
    top_20 = importances.head(20)
    ax1.barh(range(len(top_20)), top_20['importance'].values)
    ax1.set_yticks(range(len(top_20)))
    ax1.set_yticklabels(top_20['feature'].values)
    ax1.set_xlabel('Importance')
    ax1.set_title('Top 20 Feature Importances (Random Forest)')
    ax1.invert_yaxis()
    
    # Plot 2: Distribution of top feature by class
    ax2 = axes[0, 1]
    top_feature = importances.iloc[0]['feature']
    for label in y_train.unique():
        ax2.hist(X_train[top_feature][y_train == label], bins=50, alpha=0.5, label=f'Class {label}')
    ax2.set_xlabel(top_feature)
    ax2.set_ylabel('Frequency')
    ax2.set_title(f'Distribution of Top Feature: {top_feature}')
    ax2.legend()
    
    # Plot 3: Second top feature distribution
    ax3 = axes[1, 0]
    if len(importances) > 1:
        second_feature = importances.iloc[1]['feature']
        for label in y_train.unique():
            ax3.hist(X_train[second_feature][y_train == label], bins=50, alpha=0.5, label=f'Class {label}')
        ax3.set_xlabel(second_feature)
        ax3.set_ylabel('Frequency')
        ax3.set_title(f'Distribution of 2nd Feature: {second_feature}')
        ax3.legend()
    
    # Plot 4: Scatter of top 2 features
    ax4 = axes[1, 1]
    if len(importances) > 1:
        # Sample for visualization if dataset is large
        sample_size = min(5000, len(X_train))
        idx = np.random.choice(len(X_train), sample_size, replace=False)
        
        colors = ['blue' if label == 0 else 'red' for label in y_train.iloc[idx]]
        ax4.scatter(X_train[top_feature].iloc[idx], X_train[second_feature].iloc[idx], 
                   c=colors, alpha=0.3, s=10)
        ax4.set_xlabel(top_feature)
        ax4.set_ylabel(second_feature)
        ax4.set_title('Top 2 Features Scatter (Blue=Normal, Red=Attack)')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n📊 Visualization saved to: {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Feature Separability Analysis using Random Forest')
    parser.add_argument('--train', required=True, help='Path to training CSV file')
    parser.add_argument('--test', required=True, help='Path to test CSV file')
    parser.add_argument('--threshold', type=float, default=0.95, help='Threshold for perfect separability (default: 0.95)')
    parser.add_argument('--output', default='feature_separability.png', help='Output visualization file')
    args = parser.parse_args()
    
    print("=" * 70)
    print("   FEATURE SEPARABILITY ANALYSIS - RANDOM FOREST")
    print("   Demonstrating Perfect/Near-Perfect Class Separation")
    print("=" * 70)
    
    # Load and preprocess data
    train_df, test_df = load_dataset(args.train, args.test)
    X_train, y_train, X_test, y_test, _ = preprocess_data(train_df, test_df)
    
    # Use all features for top-n analysis
    num_features = len(X_train.columns)
    
    print(f"\n📊 Dataset Info:")
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    print(f"   Features: {num_features}")
    print(f"   Train class distribution: {dict(y_train.value_counts())}")
    print(f"   Test class distribution: {dict(y_test.value_counts())}")
    
    # Find perfectly separable features
    separability_results = find_perfectly_separable_features(X_train, y_train, X_test, y_test, threshold=args.threshold)
    
    # Demonstrate perfect separation with ALL features ranked by importance
    importances, acc_top, acc_full = demonstrate_perfect_separation(X_train, y_train, X_test, y_test, top_n=num_features)
    
    # Test minimal tree depth
    depth_results = test_minimal_tree_depth(X_train, y_train, X_test, y_test)
    
    # Create visualization
    visualize_separability(X_train, y_train, importances, output_file=args.output)
    
    # Final summary
    print("\n" + "=" * 70)
    print("   CONCLUSION: FEATURE SEPARABILITY ANALYSIS")
    print("=" * 70)
    
    perfect_count = len(separability_results[separability_results['accuracy'] >= args.threshold])
    
    if perfect_count > 0:
        print(f"\n   ⚠️  FOUND {perfect_count} FEATURES WITH PERFECT SEPARABILITY (≥{args.threshold*100:.0f}%)")
        print(f"\n   Top perfectly separable features:")
        for _, row in separability_results[separability_results['accuracy'] >= args.threshold].head(10).iterrows():
            print(f"      • {row['feature']}: {row['accuracy']*100:.2f}%")
        
        print(f"\n   IMPLICATIONS:")
        print(f"   1. The dataset contains features that trivially separate classes")
        print(f"   2. This could indicate:")
        print(f"      - Data leakage (target information encoded in features)")
        print(f"      - Synthetic/artificial data patterns")
        print(f"      - Features derived from the target variable")
        print(f"   3. High ML accuracy on this dataset may not generalize to real-world data")
    else:
        print(f"\n   ✅ No single feature provides perfect separability")
        print(f"   The classification task requires combining multiple features")
    
    # Save detailed results
    separability_results.to_csv('feature_separability_results.csv', index=False)
    print(f"\n📁 Detailed results saved to: feature_separability_results.csv")


if __name__ == "__main__":
    main()
