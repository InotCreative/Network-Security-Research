#!/usr/bin/env python3
"""
Complete Ablation Study Runner

Runs the full ablation study including:
1. Binary classification ablations (B1-B6, E1-E4)
2. Multiclass classification ablations (B1-B6, E1-E4)
3. Attack specialist ablations (S1-S4 per attack type)

Statistical Testing:
- Paired t-test for comparing CV scores between configurations
- Cohen's d effect size for practical significance
- Multiple comparisons tracking

Usage:
    python run_ablation_study.py
    python run_ablation_study.py --train custom_train.csv --test custom_test.csv
    python run_ablation_study.py --skip-specialists
"""

import os
import sys
import argparse
import pickle
import numpy as np
from datetime import datetime


def check_prerequisites():
    """Check if required files and models exist."""
    print("=" * 80)
    print("🔍 CHECKING PREREQUISITES")
    print("=" * 80)
    
    issues = []
    
    # Check for training data
    if not os.path.exists('UNSW_balanced_train.csv'):
        issues.append("UNSW_balanced_train.csv not found - run: python create_balanced_split.py")
    
    # Check for cached models
    binary_cache = os.path.exists('Models/Binary')
    multiclass_cache = os.path.exists('Models/Multiclass')
    main_model = os.path.exists('trained_novel_ensemble_model.pkl')
    
    if not binary_cache and not main_model:
        issues.append("No cached binary models - run: python run_novel_ml.py --dataset UNSW_balanced_train.csv")
    
    if not multiclass_cache:
        issues.append("No cached multiclass models - run: python run_novel_ml.py --dataset UNSW_balanced_train.csv --multiclass")
    
    if issues:
        print("\n⚠️  MISSING PREREQUISITES:")
        for issue in issues:
            print(f"   • {issue}")
        print("\n💡 Run the following commands first:")
        print("   1. python create_balanced_split.py")
        print("   2. python run_novel_ml.py --dataset UNSW_balanced_train.csv")
        print("   3. python run_novel_ml.py --dataset UNSW_balanced_train.csv --multiclass")
        print("   4. python train_improved_specialists.py UNSW_balanced_train.csv")
        return False
    
    print("✅ All prerequisites met!")
    return True


def run_binary_ablation(train_csv, test_csv):
    """Run binary classification ablation study."""
    print("\n" + "=" * 80)
    print("🔬 BINARY CLASSIFICATION ABLATION STUDY")
    print("=" * 80)
    
    from ablation_study import AblationStudy
    
    study = AblationStudy(
        train_csv=train_csv,
        test_csv=test_csv,
        classification_type='binary',
        cache_dir='Models'
    )
    
    # Run all configurations
    study.run_all_baselines()
    study.run_all_ensembles()
    study.run_statistical_comparisons()
    study.generate_summary()
    
    # Save results
    study.save_results('ablation_results_binary.pkl')
    
    return study.results


def run_multiclass_ablation(train_csv, test_csv):
    """Run multiclass classification ablation study."""
    print("\n" + "=" * 80)
    print("🔬 MULTICLASS CLASSIFICATION ABLATION STUDY")
    print("=" * 80)
    
    from ablation_study import AblationStudy
    
    study = AblationStudy(
        train_csv=train_csv,
        test_csv=test_csv,
        classification_type='multiclass',
        cache_dir='Models'
    )
    
    # Run all configurations
    study.run_all_baselines()
    study.run_all_ensembles()
    study.run_statistical_comparisons()
    study.generate_summary()
    
    # Save results
    study.save_results('ablation_results_multiclass.pkl')
    
    return study.results


def run_specialist_ablation(train_csv, test_csv):
    """Run attack specialist ablation study."""
    print("\n" + "=" * 80)
    print("🔬 ATTACK SPECIALIST ABLATION STUDY")
    print("=" * 80)
    
    from ablation_study import AblationStudy
    
    study = AblationStudy(
        train_csv=train_csv,
        test_csv=test_csv,
        classification_type='binary',
        cache_dir='Models'
    )
    
    # Run specialist ablations
    specialist_results = study.run_specialist_ablations()
    
    # Save results
    with open('ablation_results_specialists.pkl', 'wb') as f:
        pickle.dump(specialist_results, f)
    
    print(f"\n💾 Specialist results saved to: ablation_results_specialists.pkl")
    
    return specialist_results


def print_statistical_testing_summary():
    """Print summary of statistical testing methodology."""
    print("\n" + "=" * 80)
    print("📊 ENHANCED STATISTICAL TESTING METHODOLOGY")
    print("=" * 80)
    
    print("""
+-----------------------------------------------------------------------------+
|                        STATISTICAL TESTS PERFORMED                          |
+-----------------------------------------------------------------------------+

  1. SHAPIRO-WILK NORMALITY TEST
     - Tests if CV scores follow normal distribution
     - H0: Data is normally distributed
     - If p > 0.05: Normality assumption holds

  2. ADAPTIVE TEST SELECTION
     - If BOTH groups normal (p > 0.05): Paired t-test
     - Otherwise: Wilcoxon signed-rank test (non-parametric)
     - Ensures valid statistical inference

  3. COHEN'S D EFFECT SIZE
     - Measures practical significance of differences
     - Formula: d = mean(diff) / std(diff)
     - Interpretation:
       |d| < 0.2  : Negligible effect
       |d| < 0.5  : Small effect
       |d| < 0.8  : Medium effect
       |d| >= 0.8 : Large effect

  4. HOLM-BONFERRONI CORRECTION
     - Controls Family-Wise Error Rate (FWER)
     - Less conservative than Bonferroni
     - Applied to all 14 comparisons
     - Reports both raw and corrected p-values

  5. BOOTSTRAP CONFIDENCE INTERVALS (Test Set)
     - 1000 bootstrap samples
     - 95% confidence intervals for accuracy and F1
     - Provides uncertainty estimates for test metrics

  6. CROSS-VALIDATION SETUP
     - 5-fold Stratified K-Fold
     - shuffle=True, random_state=42
     - Consistent across ALL configurations

+-----------------------------------------------------------------------------+
|                         COMPARISONS PERFORMED (14)                          |
+-----------------------------------------------------------------------------+

  Feature Engineering Comparisons (vs B1 baseline):
   1. B2a vs B1a: Statistical features (7) - no selection
   2. B2b vs B1b: Statistical features (7) - with selection
   3. B3a vs B1a: Duration features (5) - no selection
   4. B3b vs B1b: Duration features (5) - with selection
   5. B4a vs B1a: Network features (9) - no selection
   6. B4b vs B1b: Network features (9) - with selection
   7. B5a vs B1a: Interaction features (2) - no selection
   8. B5b vs B1b: Interaction features (2) - with selection
   9. B6a vs B1a: All 23 features - no selection
  10. B6b vs B1b: All 23 features - with selection

  Feature Selection Effect:
  11. B6b vs B6a: Does selection improve full FE pipeline?

  Ensemble Comparisons:
  12. E4 vs B6b: Ensemble vs single GB (main claim)
  13. E2 vs E1: Mixing ratio optimization (no selection)
  14. E4 vs E3: Mixing ratio optimization (with selection)

  Specialist Comparisons (per attack):
  - S1 vs S2: SMOTE effect (with selection)
  - S3 vs S4: SMOTE effect (no selection)
  - S1 vs S3: Selection effect (with SMOTE)
  - S2 vs S4: Selection effect (no SMOTE)

+-----------------------------------------------------------------------------+
""")


def generate_final_report(binary_results, multiclass_results, specialist_results):
    """Generate comprehensive final report."""
    print("\n" + "=" * 80)
    print("📋 FINAL ABLATION STUDY REPORT")
    print("=" * 80)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Binary summary
    if binary_results:
        print("\n" + "-" * 40)
        print("BINARY CLASSIFICATION RESULTS")
        print("-" * 40)
        
        if 'B1a' in binary_results and 'B6b' in binary_results:
            baseline = binary_results['B1a']['test_accuracy']
            full_fe = binary_results['B6b']['test_accuracy']
            print(f"  Baseline (no FE):     {baseline:.4f}")
            print(f"  Full FE + Selection:  {full_fe:.4f}")
            print(f"  Improvement:          {full_fe - baseline:+.4f}")
        
        if 'E4' in binary_results:
            ensemble = binary_results['E4']['test_accuracy']
            print(f"  Ensemble (E4):        {ensemble:.4f}")
    
    # Multiclass summary
    if multiclass_results:
        print("\n" + "-" * 40)
        print("MULTICLASS CLASSIFICATION RESULTS")
        print("-" * 40)
        
        if 'B1a' in multiclass_results and 'B6b' in multiclass_results:
            baseline = multiclass_results['B1a']['test_accuracy']
            full_fe = multiclass_results['B6b']['test_accuracy']
            print(f"  Baseline (no FE):     {baseline:.4f}")
            print(f"  Full FE + Selection:  {full_fe:.4f}")
            print(f"  Improvement:          {full_fe - baseline:+.4f}")
        
        if 'E4' in multiclass_results:
            ensemble = multiclass_results['E4']['test_accuracy']
            print(f"  Ensemble (E4):        {ensemble:.4f}")
    
    # Specialist summary
    if specialist_results:
        print("\n" + "-" * 40)
        print("SPECIALIST ABLATION RESULTS")
        print("-" * 40)
        
        for attack, configs in specialist_results.items():
            if isinstance(configs, dict) and 'S1' in configs:
                s1_f1 = configs['S1'].get('test_f1', 0)
                print(f"  {attack}: S1 F1={s1_f1:.3f}")
    
    # Save combined report
    combined_results = {
        'binary': binary_results,
        'multiclass': multiclass_results,
        'specialists': specialist_results,
        'timestamp': datetime.now().isoformat()
    }
    
    with open('ablation_study_complete_results.pkl', 'wb') as f:
        pickle.dump(combined_results, f)
    
    print("\n" + "=" * 80)
    print("💾 RESULTS SAVED")
    print("=" * 80)
    print("  • ablation_results_binary.pkl")
    print("  • ablation_results_multiclass.pkl")
    print("  • ablation_results_specialists.pkl")
    print("  • ablation_study_complete_results.pkl")


def main():
    parser = argparse.ArgumentParser(description='Complete Ablation Study Runner')
    parser.add_argument('--train', type=str, default='UNSW_balanced_train.csv',
                       help='Training dataset CSV')
    parser.add_argument('--test', type=str, default='UNSW_balanced_test.csv',
                       help='Test dataset CSV')
    parser.add_argument('--skip-binary', action='store_true',
                       help='Skip binary classification ablation')
    parser.add_argument('--skip-multiclass', action='store_true',
                       help='Skip multiclass classification ablation')
    parser.add_argument('--skip-specialists', action='store_true',
                       help='Skip specialist ablation')
    parser.add_argument('--skip-prereq-check', action='store_true',
                       help='Skip prerequisite check')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🔬 COMPLETE ABLATION STUDY")
    print("=" * 80)
    print(f"Training data: {args.train}")
    print(f"Test data: {args.test}")
    print(f"Run binary: {not args.skip_binary}")
    print(f"Run multiclass: {not args.skip_multiclass}")
    print(f"Run specialists: {not args.skip_specialists}")
    
    # Print statistical testing methodology
    print_statistical_testing_summary()
    
    # Check prerequisites
    if not args.skip_prereq_check:
        if not check_prerequisites():
            print("\n❌ Prerequisites not met. Exiting.")
            sys.exit(1)
    
    # Check if data files exist
    if not os.path.exists(args.train):
        print(f"\n❌ Training data not found: {args.train}")
        sys.exit(1)
    
    # Initialize results
    binary_results = None
    multiclass_results = None
    specialist_results = None
    
    # Run ablations
    try:
        if not args.skip_binary:
            binary_results = run_binary_ablation(args.train, args.test)
        
        if not args.skip_multiclass:
            multiclass_results = run_multiclass_ablation(args.train, args.test)
        
        if not args.skip_specialists:
            specialist_results = run_specialist_ablation(args.train, args.test)
        
        # Generate final report
        generate_final_report(binary_results, multiclass_results, specialist_results)
        
        print("\n" + "=" * 80)
        print("✅ ABLATION STUDY COMPLETE!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Error during ablation study: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
