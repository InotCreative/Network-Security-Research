#!/usr/bin/env python3
"""
Master Research Execution Script
Runs core experiments automatically:
1. Binary Classification
2. Multiclass Classification (baseline)
"""

import subprocess
import sys

def main():
    """Run core research experiments"""
    
    print("\n" + "="*80)
    print("🔬 CORE RESEARCH EXPERIMENTS")
    print("="*80)
    print("This will run:")
    print("   1️⃣ Binary Classification (Normal vs Attack)")
    print("   2️⃣ Multiclass Classification (10 attack types - baseline)")
    print("\nYou'll run separately:")
    print("   - Multiclass with SMOTE (run_multiclass_with_smote.py)")
    print("   - Cross-dataset validation (test_cross_dataset.py)")
    print("   - Attack specialists (train_improved_specialists.py)")
    print("="*80)
    
    input("\n⏸️  Press ENTER to start experiments...")
    
    # =========================================================================
    # EXPERIMENT 1: Binary Classification
    # =========================================================================
    print("\n" + "="*80)
    print("1️⃣ BINARY CLASSIFICATION")
    print("="*80)
    print("Training: Normal vs Attack classifier")
    print("Expected: 93-95% accuracy")
    print("Runtime: ~10-15 minutes")
    print("="*80 + "\n")
    
    binary_cmd = [
        sys.executable, 'run_novel_ml.py',
        '--dataset', 'UNSW_balanced_train.csv',
        '--test-dataset', 'UNSW_balanced_test.csv',
        '--force-retrain'
    ]
    
    try:
        subprocess.run(binary_cmd, check=True)
        print("\n✅ Binary classification complete!")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Binary classification failed: {e}")
        print("Fix errors before proceeding to multiclass")
        return
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        return
    
    # =========================================================================
    # EXPERIMENT 2: Multiclass Classification (No SMOTE)
    # =========================================================================
    print("\n" + "="*80)
    print("2️⃣ MULTICLASS CLASSIFICATION (NO SMOTE - BASELINE)")
    print("="*80)
    print("Training: 10-class attack classifier WITHOUT oversampling")
    print("Expected: 83% accuracy, poor minority class recall")
    print("Runtime: ~15-20 minutes")
    print("This is your NULL MODEL for comparison with SMOTE version")
    print("="*80 + "\n")
    
    multiclass_cmd = [
        sys.executable, 'run_multiclass_experiment.py',
        '--force-retrain'
    ]
    
    try:
        subprocess.run(multiclass_cmd, check=True)
        print("\n✅ Multiclass classification (baseline) complete!")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Multiclass classification failed: {e}")
        return
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        return
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*80)
    print("✅ CORE EXPERIMENTS COMPLETE!")
    print("="*80)
    
    print("\n📊 Results Summary:")
    print("   1️⃣ Binary Classification: Check for 93-95% accuracy")
    print("   2️⃣ Multiclass (No SMOTE): Check for 83% accuracy")
    
    print("\n📋 Next Steps - Run These Manually:")
    print("\n   3️⃣ Multiclass WITH SMOTE (compare to baseline above):")
    print("      python run_multiclass_with_smote.py --dataset UNSW_balanced_train.csv --test-dataset UNSW_balanced_test.csv --force-retrain")
    
    print("\n   4️⃣ Cross-Dataset Validation:")
    print("      python test_cross_dataset.py \"/path/to/Bot-IoT\"")
    
    print("\n   5️⃣ Attack Specialists (optional):")
    print("      python train_improved_specialists.py UNSW_balanced_train.csv UNSW_balanced_test.csv")
    
    print("\n🎯 Key Comparison:")
    print("   Compare multiclass results:")
    print("   - Without SMOTE: Good overall accuracy, POOR minority recall")
    print("   - With SMOTE: Slight accuracy drop, MUCH BETTER minority recall")
    print("   - Expected improvement: +30-40% recall on Exploits, Worms, etc.")
    
    print("\n💪 You have MORE than enough novelty for ACM!")
    print("   Submit with confidence! 🚀")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()