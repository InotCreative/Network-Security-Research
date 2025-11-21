#!/usr/bin/env python3
"""
Master Research Execution Script
Runs ALL experiments in order with separate terminals for each section
Each terminal shows exactly which experiment is running
"""

import subprocess
import sys
import time
import os

def run_in_new_terminal(title, command, description):
    """Run a command in a new Windows terminal with a clear title"""
    
    print(f"\n{'='*80}")
    print(f"🚀 {title}")
    print(f"{'='*80}")
    print(f"Description: {description}")
    print(f"Command: {command}")
    print(f"Opening new terminal window...")
    print(f"{'='*80}\n")
    
    # Windows terminal command - opens new window with title
    # The title will show in the terminal window
    full_command = f'start "{title}" cmd /k "title {title} && {command}"'
    
    subprocess.Popen(full_command, shell=True)
    
    # Wait a bit before starting next one
    time.sleep(2)
    
    print(f"✅ Terminal opened for: {title}")
    print(f"   Check the new window titled: '{title}'\n")


def main():
    """Run all research experiments in order with separate terminals"""
    
    print("\n" + "="*80)
    print("🔬 COMPREHENSIVE RESEARCH EXPERIMENT SUITE")
    print("="*80)
    print("This will run ALL experiments in order with separate terminal windows")
    print("Each window will be clearly labeled so you know what's running")
    print("="*80)
    
    input("\n⏸️  Press ENTER to start all experiments...")
    
    # =========================================================================
    # EXPERIMENT 1: Binary Classification (Baseline)
    # =========================================================================
    run_in_new_terminal(
        title="1️⃣ Binary Classification",
        command="python run_novel_ml.py --dataset UNSW_balanced_train.csv --test-dataset UNSW_balanced_test.csv --force-retrain",
        description="Train binary classifier (Normal vs Attack) with all fixes applied"
    )
    
    # =========================================================================
    # EXPERIMENT 2: Multiclass Classification (Baseline - No SMOTE)
    # =========================================================================
    run_in_new_terminal(
        title="2️⃣ Multiclass (No SMOTE)",
        command="python run_multiclass_experiment.py --force-retrain",
        description="Train multiclass classifier (10 attack types) WITHOUT oversampling - NULL MODEL"
    )
    
    # =========================================================================
    # EXPERIMENT 3: Multiclass Classification (With SMOTE)
    # =========================================================================
    run_in_new_terminal(
        title="3️⃣ Multiclass (With SMOTE)",
        command="python run_multiclass_with_smote.py --dataset UNSW_balanced_train.csv --test-dataset UNSW_balanced_test.csv --force-retrain",
        description="Train multiclass classifier WITH SMOTE oversampling - EXPERIMENTAL MODEL"
    )
    
    # =========================================================================
    # EXPERIMENT 4: Cross-Dataset Validation (Bot-IoT)
    # =========================================================================
    
    # Check if Bot-IoT path is set
    print("\n" + "="*80)
    print("4️⃣ Cross-Dataset Validation")
    print("="*80)
    
    botiot_path = input("Enter Bot-IoT dataset path (or press ENTER to skip): ").strip()
    
    if botiot_path and os.path.exists(botiot_path):
        run_in_new_terminal(
            title="4️⃣ Cross-Dataset (Bot-IoT)",
            command=f'python test_cross_dataset.py "{botiot_path}"',
            description="Test UNSW-trained model on Bot-IoT dataset with adaptive thresholding"
        )
    else:
        print("⏭️  Skipping cross-dataset validation (no valid path provided)")
    
    # =========================================================================
    # EXPERIMENT 5: Attack Specialists (Optional)
    # =========================================================================
    
    print("\n" + "="*80)
    print("5️⃣ Attack Specialists (Optional)")
    print("="*80)
    
    train_specialists = input("Train improved attack specialists? (y/N): ").strip().lower()
    
    if train_specialists == 'y':
        run_in_new_terminal(
            title="5️⃣ Attack Specialists",
            command="python train_improved_specialists.py UNSW_balanced_train.csv UNSW_balanced_test.csv",
            description="Train individual specialists for each attack type with SMOTE"
        )
    else:
        print("⏭️  Skipping attack specialists")
    
    # =========================================================================
    # Summary
    # =========================================================================
    
    print("\n" + "="*80)
    print("✅ ALL EXPERIMENTS LAUNCHED!")
    print("="*80)
    print("\n📊 Terminal Windows Opened:")
    print("   1️⃣  Binary Classification")
    print("   2️⃣  Multiclass (No SMOTE) - Baseline")
    print("   3️⃣  Multiclass (With SMOTE) - Experimental")
    
    if botiot_path and os.path.exists(botiot_path):
        print("   4️⃣  Cross-Dataset Validation")
    
    if train_specialists == 'y':
        print("   5️⃣  Attack Specialists")
    
    print("\n💡 Tips:")
    print("   - Check each terminal window by its title")
    print("   - Experiments run in parallel (faster)")
    print("   - Wait for all to complete before comparing results")
    print("   - Look for 'EXPERIMENT COMPLETE' in each window")
    
    print("\n📋 What to Compare:")
    print("   1. Binary vs Multiclass accuracy")
    print("   2. Multiclass without SMOTE vs with SMOTE")
    print("   3. Check minority class recall improvement with SMOTE")
    print("   4. Cross-dataset performance (should be 50-55%)")
    
    print("\n🎯 Expected Results:")
    print("   Binary:                93-95% accuracy")
    print("   Multiclass (No SMOTE): 83% accuracy, poor minority recall")
    print("   Multiclass (SMOTE):    78-82% accuracy, BETTER minority recall")
    print("   Cross-Dataset:         50-55% accuracy (with adaptive threshold)")
    
    print("\n" + "="*80)
    print("🎉 All experiments are running in separate terminal windows!")
    print("="*80)


if __name__ == "__main__":
    main()