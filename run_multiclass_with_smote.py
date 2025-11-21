#!/usr/bin/env python3
"""
Run multiclass experiment with SMOTE oversampling
EXACT SAME ARCHITECTURE - only adds SMOTE to training data
"""

# Patch the NovelEnsembleMLSystem.train() method to add SMOTE before classifier training
from novel_ensemble_ml import NovelEnsembleMLSystem, AdaptiveEnsembleClassifier
from imblearn.over_sampling import SMOTE, ADASYN
import numpy as np

# Save original fit method
original_classifier_fit = AdaptiveEnsembleClassifier.fit

def fit_with_smote(self, X, y, attack_types=None, cache_dir="Models", classification_type="binary", force_retrain_models=None, use_smote=True, use_adasyn=False):
    """
    Wrapper around AdaptiveEnsembleClassifier.fit() that applies SMOTE to training data
    
    EVERYTHING ELSE IS IDENTICAL - same preprocessing, same models, same training
    Only difference: Training data is balanced with SMOTE
    """
    
    # Only apply SMOTE for multiclass (not binary - binary is already done)
    if classification_type == 'multiclass' and use_smote:
        print(f"\n🎯 APPLYING SMOTE OVERSAMPLING TO TRAINING DATA")
        print("=" * 60)
        
        print(f"\n📊 Class Distribution BEFORE SMOTE:")
        unique, counts = np.unique(y, return_counts=True)
        for cls, count in zip(unique, counts):
            pct = count / len(y) * 100
            print(f"   Class {cls}: {count:,} ({pct:.1f}%)")
        
        try:
            # Calculate k_neighbors based on smallest class
            min_samples = min(counts)
            k_neighbors = min(5, min_samples - 1)
            
            if k_neighbors < 1:
                print(f"   ⚠️  Some classes have too few samples - skipping SMOTE")
                X_resampled = X
                y_resampled = y
            else:
                # Identify majority and minority classes
                majority_class = unique[np.argmax(counts)]
                majority_count = max(counts)
                
                print(f"\n   🔍 SMOTE Configuration:")
                print(f"      Majority class: {majority_class} ({majority_count:,} samples)")
                print(f"      Strategy: 'auto' (oversample minorities to match majority)")
                print(f"      ✅ Majority class will be LEFT UNTOUCHED")
                
                if use_adasyn:
                    sampler = ADASYN(random_state=42, n_neighbors=k_neighbors, sampling_strategy='auto')
                    print(f"      Using ADASYN with {k_neighbors} neighbors...")
                else:
                    sampler = SMOTE(random_state=42, k_neighbors=k_neighbors, sampling_strategy='auto')
                    print(f"      Using SMOTE with {k_neighbors} neighbors...")
                
                X_resampled, y_resampled = sampler.fit_resample(X, y)
                
                print(f"\n📊 Class Distribution AFTER SMOTE:")
                unique_resampled, counts_resampled = np.unique(y_resampled, return_counts=True)
                for cls, count in zip(unique_resampled, counts_resampled):
                    pct = count / len(y_resampled) * 100
                    print(f"   Class {cls}: {count:,} ({pct:.1f}%)")
                
                print(f"\n   ✅ SMOTE applied: {len(y):,} → {len(y_resampled):,} samples")
                print(f"   ✅ Now training ensemble on balanced data...")
        
        except Exception as e:
            print(f"   ⚠️  SMOTE failed: {e}")
            print(f"   Using original data")
            X_resampled = X
            y_resampled = y
    else:
        # No SMOTE for binary or if disabled
        X_resampled = X
        y_resampled = y
    
    # Call original fit method with (possibly) resampled data
    # EVERYTHING FROM HERE IS IDENTICAL TO ORIGINAL
    print("\n" + "=" * 60)
    return original_classifier_fit(self, X_resampled, y_resampled, attack_types, cache_dir, classification_type, force_retrain_models)

# Monkey-patch the fit method
AdaptiveEnsembleClassifier.fit = fit_with_smote

# Now import and run the standard experiment
# This ensures EXACT same flow as original
if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Multiclass with SMOTE')
    parser.add_argument('--dataset', type=str, default='UNSW_balanced_train.csv')
    parser.add_argument('--test-dataset', type=str, default='UNSW_balanced_test.csv')
    parser.add_argument('--force-retrain', action='store_true')
    parser.add_argument('--adasyn', action='store_true')
    parser.add_argument('--no-smote', action='store_true')
    
    args = parser.parse_args()
    
    use_smote = not args.no_smote
    use_adasyn = args.adasyn
    
    print("\n" + "="*80)
    print("🔬 MULTICLASS EXPERIMENT WITH SMOTE OVERSAMPLING")
    print("="*80)
    print(f"Training: {args.dataset}")
    print(f"Testing: {args.test_dataset}")
    print(f"SMOTE: {'Enabled' if use_smote else 'Disabled'}")
    print(f"ADASYN: {'Enabled' if use_adasyn else 'Disabled'}")
    print(f"Force retrain: {args.force_retrain}")
    print("="*80)
    
    # Store SMOTE settings globally so fit method can access them
    AdaptiveEnsembleClassifier._use_smote = use_smote
    AdaptiveEnsembleClassifier._use_adasyn = use_adasyn
    
    # Modify the fit wrapper to read these settings
    def fit_with_smote_v2(self, X, y, attack_types=None, cache_dir="Models", classification_type="binary", force_retrain_models=None):
        use_smote = getattr(AdaptiveEnsembleClassifier, '_use_smote', True)
        use_adasyn = getattr(AdaptiveEnsembleClassifier, '_use_adasyn', False)
        return fit_with_smote(self, X, y, attack_types, cache_dir, classification_type, force_retrain_models, use_smote, use_adasyn)
    
    AdaptiveEnsembleClassifier.fit = fit_with_smote_v2
    
    # Now run EXACT same flow as original
    print("\n📊 Initializing system (multiclass mode)...")
    system = NovelEnsembleMLSystem(classification_type='multiclass')
    
    print("\n🎯 Training model with SMOTE-balanced data...")
    print("   (All other steps identical to original)\n")
    
    # Train - EXACT same method call as original
    system.train(
        args.dataset,
        test_csv=args.test_dataset,
        force_retrain=args.force_retrain
    )
    
    # Evaluate - EXACT same method call as original
    print("\n📊 Evaluating on test set...")
    system.evaluate(args.test_dataset)
    
    print("\n" + "="*80)
    print("✅ MULTICLASS WITH SMOTE COMPLETE")
    print("="*80)
    print("\n💡 Compare with baseline (no SMOTE):")
    print(f"   python run_multiclass_experiment.py --force-retrain")
