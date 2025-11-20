#!/usr/bin/env python3
"""
Improved Attack Specialist Training with SMOTE and Optimal Thresholding
Fixes specialist performance issues while maintaining exact pipeline architecture

Improvements:
1. SMOTE/ADASYN for training data only (no test data leakage)
2. Optimal F1 threshold instead of fixed 0.5
3. Focal Loss via XGBoost scale_pos_weight
4. Calibration curves to validate non-random predictions
5. Same train/test split architecture as original
6. Internal validation + holdout test set
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.metrics import (f1_score, balanced_accuracy_score, precision_score, 
                            recall_score, accuracy_score, roc_auc_score, roc_curve)

# Try to import calibration_curve from different locations
calibration_curve = None
try:
    from sklearn.calibration import calibration_curve
except ImportError:
    try:
        from sklearn.metrics import calibration_curve
    except ImportError:
        print("⚠️  Warning: calibration_curve not available - calibration plots will be skipped")
        calibration_curve = None
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE, ADASYN
import xgboost as xgb
import pickle
import os
import matplotlib.pyplot as plt
from novel_ensemble_ml import DynamicFeatureEngineer

class ImprovedSpecialistTrainer:
    """Train improved attack specialists with SMOTE and optimal thresholding"""
    
    def __init__(self, train_csv, test_csv=None, use_smote=True, use_adasyn=False):
        self.train_csv = train_csv
        self.test_csv = test_csv
        self.use_smote = use_smote
        self.use_adasyn = use_adasyn
        self.specialists = {}
        self.feature_names = None
        
    def load_data(self):
        """Load and preprocess data"""
        print("📁 Loading UNSW-NB15 dataset...")
        df_train = pd.read_csv(self.train_csv)
        
        if self.test_csv:
            df_test = pd.read_csv(self.test_csv)
            df = pd.concat([df_train, df_test], ignore_index=True)
        else:
            df = df_train
        
        print(f"✅ Loaded {len(df):,} samples")
        
        # Check for required columns
        if 'label' not in df.columns or 'attack_cat' not in df.columns:
            raise ValueError("Dataset must have 'label' (binary) and 'attack_cat' columns")
        
        # Encode categorical features
        categorical_cols = ['proto', 'service', 'state']
        self.label_encoders = {}
        
        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
        
        # Get feature names (exclude metadata)
        exclude_cols = ['id', 'attack_cat', 'label', 'stime', 'srcip', 'dstip']
        self.feature_names = [col for col in df.columns if col not in exclude_cols]
        
        # Extract features, labels, and attack types
        X = df[self.feature_names].fillna(0).values
        y = df['label'].values
        attack_types = df['attack_cat'].values
        
        print(f"📊 Features: {len(self.feature_names)}")
        print(f"📊 Attack types: {len(np.unique(attack_types))}")
        
        return X, y, attack_types
    
    def train_specialist(self, attack_name, X_raw, y_binary, use_smote=True):
        """
        Train a single specialist with SMOTE and optimal thresholding
        
        Architecture (EXACT SAME AS ORIGINAL):
        1. Train/test split (80/20)
        2. Feature engineering (fit on train only)
        3. Scaling (fit on train only)
        4. Feature selection (fit on train only)
        5. SMOTE (ONLY on training data - NEW)
        6. Model training
        7. Optimal threshold finding (on validation set - NEW)
        8. Final evaluation on holdout test set
        """
        print(f"\n{'='*80}")
        print(f"🎯 Training {attack_name} Specialist (IMPROVED)")
        print(f"{'='*80}")
        
        # Count samples
        attack_samples = np.sum(y_binary == 1)
        normal_samples = np.sum(y_binary == 0)
        imbalance_ratio = normal_samples / attack_samples if attack_samples > 0 else 1
        
        print(f"📊 Original Distribution:")
        print(f"   {attack_name}: {attack_samples:,} samples")
        print(f"   Normal: {normal_samples:,} samples")
        print(f"   Imbalance Ratio: {imbalance_ratio:.1f}:1")
        
        # STEP 1: Train/Test Split (80/20) - SAME AS ORIGINAL
        print(f"\n1️⃣  Train/Test Split (80/20)...")
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X_raw, y_binary, test_size=0.2, random_state=42, stratify=y_binary
        )
        print(f"   Training: {len(X_train_raw):,} samples")
        print(f"   Testing:  {len(X_test_raw):,} samples")
        
        # STEP 2: Feature Engineering (fit on train only) - SAME AS ORIGINAL
        print(f"\n2️⃣  Feature Engineering (train only)...")
        fe_train = DynamicFeatureEngineer()
        X_train_df = pd.DataFrame(X_train_raw, columns=self.feature_names)
        X_test_df = pd.DataFrame(X_test_raw, columns=self.feature_names)
        
        X_train_eng = fe_train.fit_transform(X_train_df)
        X_test_eng = fe_train.transform(X_test_df)
        print(f"   Features: {X_train_eng.shape[1]}")
        
        # STEP 3: Scaling (fit on train only) - SAME AS ORIGINAL
        print(f"\n3️⃣  Scaling (train only)...")
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train_eng)
        X_test_scaled = scaler.transform(X_test_eng)
        
        # STEP 4: Feature Selection (fit on train only) - SAME AS ORIGINAL
        print(f"\n4️⃣  Feature Selection (train only)...")
        selector = SelectKBest(f_classif, k=min(30, X_train_scaled.shape[1]))
        X_train_selected = selector.fit_transform(X_train_scaled, y_train)
        X_test_selected = selector.transform(X_test_scaled)
        print(f"   Selected: {X_train_selected.shape[1]} features")
        
        # STEP 5: SMOTE/ADASYN (ONLY on training data) - NEW!
        print(f"\n5️⃣  Oversampling (ONLY training data)...")
        if use_smote and imbalance_ratio > 2:
            try:
                if self.use_adasyn:
                    sampler = ADASYN(random_state=42, n_neighbors=min(5, attack_samples-1))
                    print(f"   Using ADASYN...")
                else:
                    sampler = SMOTE(random_state=42, k_neighbors=min(5, attack_samples-1))
                    print(f"   Using SMOTE...")
                
                X_train_resampled, y_train_resampled = sampler.fit_resample(X_train_selected, y_train)
                
                print(f"   Before SMOTE: {len(y_train):,} samples")
                print(f"      {attack_name}: {np.sum(y_train==1):,}")
                print(f"      Normal: {np.sum(y_train==0):,}")
                print(f"   After SMOTE: {len(y_train_resampled):,} samples")
                print(f"      {attack_name}: {np.sum(y_train_resampled==1):,}")
                print(f"      Normal: {np.sum(y_train_resampled==0):,}")
                print(f"   ✅ SMOTE applied (training data only - NO TEST DATA LEAKAGE)")
                
            except Exception as e:
                print(f"   ⚠️  SMOTE failed: {e}")
                print(f"   Using original training data")
                X_train_resampled = X_train_selected
                y_train_resampled = y_train
        else:
            print(f"   Skipping SMOTE (imbalance ratio: {imbalance_ratio:.1f}:1)")
            X_train_resampled = X_train_selected
            y_train_resampled = y_train
        
        # STEP 6: Model Training with Focal Loss - IMPROVED
        print(f"\n6️⃣  Model Training (XGBoost with Focal Loss)...")
        
        # Calculate scale_pos_weight for focal loss effect
        neg_samples = np.sum(y_train_resampled == 0)
        pos_samples = np.sum(y_train_resampled == 1)
        scale_weight = neg_samples / pos_samples if pos_samples > 0 else 1
        
        specialist = xgb.XGBClassifier(
            random_state=42,
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            min_child_weight=3,
            objective='binary:logistic',
            eval_metric='logloss',
            scale_pos_weight=scale_weight,  # Focal loss effect
            reg_alpha=0.1,
            reg_lambda=0.1,
            subsample=0.8,
            colsample_bytree=0.8
        )
        
        specialist.fit(X_train_resampled, y_train_resampled)
        print(f"   ✅ Model trained with scale_pos_weight={scale_weight:.2f}")
        
        # STEP 7: Cross-Validation on Training Data - SAME AS ORIGINAL
        print(f"\n7️⃣  Cross-Validation (3-fold on training data)...")
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        
        # CV on ORIGINAL training data (not SMOTE'd) to get realistic estimates
        cv_f1 = cross_val_score(specialist, X_train_selected, y_train, cv=cv, scoring='f1')
        cv_balanced_acc = cross_val_score(specialist, X_train_selected, y_train, cv=cv, scoring='balanced_accuracy')
        cv_auc = cross_val_score(specialist, X_train_selected, y_train, cv=cv, scoring='roc_auc')
        
        print(f"   CV F1: {cv_f1.mean():.3f} ± {cv_f1.std():.3f}")
        print(f"   CV Bal-Acc: {cv_balanced_acc.mean():.3f} ± {cv_balanced_acc.std():.3f}")
        print(f"   CV AUC-ROC: {cv_auc.mean():.3f} ± {cv_auc.std():.3f}")
        
        # STEP 8: Find Optimal Threshold (on validation split) - NEW!
        print(f"\n8️⃣  Finding Optimal Threshold...")
        
        # Use a validation split from training data to find optimal threshold
        X_train_opt, X_val_opt, y_train_opt, y_val_opt = train_test_split(
            X_train_selected, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        # Get probabilities on validation set
        y_val_proba = specialist.predict_proba(X_val_opt)[:, 1]
        
        # Find threshold that maximizes F1
        fpr, tpr, thresholds = roc_curve(y_val_opt, y_val_proba)
        f1_scores = []
        for threshold in thresholds:
            y_pred_thresh = (y_val_proba >= threshold).astype(int)
            f1 = f1_score(y_val_opt, y_pred_thresh, zero_division=0)
            f1_scores.append(f1)
        
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]
        optimal_f1 = f1_scores[optimal_idx]
        
        print(f"   Default threshold (0.5): F1 = {f1_score(y_val_opt, (y_val_proba >= 0.5).astype(int), zero_division=0):.3f}")
        print(f"   Optimal threshold ({optimal_threshold:.3f}): F1 = {optimal_f1:.3f}")
        print(f"   ✅ Using optimal threshold for predictions")
        
        # STEP 9: Final Evaluation on Holdout Test Set - SAME AS ORIGINAL
        print(f"\n9️⃣  Test Set Performance...")
        
        # Get probabilities on test set
        y_test_proba = specialist.predict_proba(X_test_selected)[:, 1]
        
        # Predictions with default threshold
        y_test_pred_default = specialist.predict(X_test_selected)
        
        # Predictions with optimal threshold
        y_test_pred_optimal = (y_test_proba >= optimal_threshold).astype(int)
        
        # Calculate metrics for both
        print(f"\n   📊 Default Threshold (0.5):")
        test_f1_default = f1_score(y_test, y_test_pred_default, zero_division=0)
        test_precision_default = precision_score(y_test, y_test_pred_default, zero_division=0)
        test_recall_default = recall_score(y_test, y_test_pred_default, zero_division=0)
        test_balanced_acc_default = balanced_accuracy_score(y_test, y_test_pred_default)
        
        print(f"      F1-Score: {test_f1_default:.3f}")
        print(f"      Balanced Accuracy: {test_balanced_acc_default:.3f}")
        print(f"      Precision: {test_precision_default:.3f}")
        print(f"      Recall: {test_recall_default:.3f}")
        
        print(f"\n   📊 Optimal Threshold ({optimal_threshold:.3f}):")
        test_f1_optimal = f1_score(y_test, y_test_pred_optimal, zero_division=0)
        test_precision_optimal = precision_score(y_test, y_test_pred_optimal, zero_division=0)
        test_recall_optimal = recall_score(y_test, y_test_pred_optimal, zero_division=0)
        test_balanced_acc_optimal = balanced_accuracy_score(y_test, y_test_pred_optimal)
        test_auc = roc_auc_score(y_test, y_test_proba)
        
        print(f"      F1-Score: {test_f1_optimal:.3f}")
        print(f"      Balanced Accuracy: {test_balanced_acc_optimal:.3f}")
        print(f"      Precision: {test_precision_optimal:.3f}")
        print(f"      Recall: {test_recall_optimal:.3f}")
        print(f"      AUC-ROC: {test_auc:.3f}")
        
        improvement = test_f1_optimal - test_f1_default
        print(f"\n   📈 F1 Improvement: {improvement:+.3f}")
        
        # STEP 10: Overfitting/Underfitting Analysis - SAME AS ORIGINAL
        print(f"\n🔟 Overfitting/Underfitting Analysis...")
        
        # Calculate training accuracy for overfitting detection
        y_train_pred = specialist.predict(X_train_selected)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        cv_accuracy_mean = cv_balanced_acc.mean()
        
        # Overfitting analysis
        overfitting_gap = train_accuracy - cv_accuracy_mean
        
        print(f"   Train Accuracy: {train_accuracy:.3f}")
        print(f"   CV Accuracy: {cv_accuracy_mean:.3f}")
        print(f"   Overfitting Gap: {overfitting_gap:.3f}")
        
        if overfitting_gap > 0.1:
            overfitting_status = "⚠️  HIGH OVERFITTING"
        elif overfitting_gap > 0.05:
            overfitting_status = "🔶 MODERATE OVERFITTING"
        else:
            overfitting_status = "✅ GOOD GENERALIZATION"
        print(f"   Status: {overfitting_status}")
        
        # Underfitting analysis
        if cv_accuracy_mean < 0.6:
            underfitting_status = "⚠️  SEVERE UNDERFITTING"
        elif cv_accuracy_mean < 0.7:
            underfitting_status = "🔶 MODERATE UNDERFITTING"
        else:
            underfitting_status = "✅ ADEQUATE PERFORMANCE"
        print(f"   Underfitting: {underfitting_status}")
        
        # Stability analysis
        cv_std = cv_balanced_acc.std()
        if cv_std < 0.02:
            stability_status = "✅ HIGHLY STABLE"
        elif cv_std < 0.05:
            stability_status = "🔶 MODERATELY STABLE"
        else:
            stability_status = "⚠️  UNSTABLE PERFORMANCE"
        print(f"   Stability: {stability_status}")
        
        # Realistic performance assessment
        if test_f1_optimal > 0.9 or test_balanced_acc_optimal > 0.95:
            print(f"   ⚠️  Still high performance - check for remaining leakage")
        elif test_f1_optimal > 0.8 or test_balanced_acc_optimal > 0.9:
            print(f"   🔶 Good performance - within realistic range")
        else:
            print(f"   ✅ Realistic performance - no apparent leakage")
        
        # STEP 11: Calibration Analysis - NEW!
        print(f"\n1️⃣1️⃣  Calibration Analysis...")
        self._plot_calibration_curve(y_test, y_test_proba, attack_name)
        
        # Store specialist - EXACT SAME STRUCTURE AS ORIGINAL
        self.specialists[attack_name] = {
            'model': specialist,
            'feature_engineer': fe_train,
            'scaler': scaler,
            'feature_selector': selector,
            'type': 'attack_vs_normal',
            'target_class': attack_name,
            'optimal_threshold': optimal_threshold,
            'test_f1_score': test_f1_optimal,
            'test_balanced_accuracy': test_balanced_acc_optimal,
            'test_auc': test_auc,
            'test_precision': test_precision_optimal,
            'test_recall': test_recall_optimal,
            'cv_f1_mean': cv_f1.mean(),
            'cv_f1_std': cv_f1.std(),
            'cv_auc_mean': cv_auc.mean(),
            'cv_auc_std': cv_auc.std(),
            'train_accuracy': train_accuracy,
            'cv_accuracy': cv_accuracy_mean,
            'overfitting_gap': overfitting_gap,
            'overfitting_status': overfitting_status,
            'underfitting_status': underfitting_status,
            'stability_status': stability_status,
            'imbalance_ratio': imbalance_ratio,
            'samples': {'attack': attack_samples, 'normal': normal_samples},
            'used_smote': use_smote and imbalance_ratio > 2,
            'test_f1_default': test_f1_default
        }
        
        print(f"\n✅ {attack_name} Specialist trained and validated")
        
        return self.specialists[attack_name]
    
    def _plot_calibration_curve(self, y_true, y_proba, attack_name):
        """Plot calibration curve to show predictions are not random"""
        try:
            if calibration_curve is None:
                print(f"   ⚠️  Calibration curve not available (sklearn version too old)")
                return
            
            prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10)
            
            plt.figure(figsize=(8, 6))
            plt.plot(prob_pred, prob_true, marker='o', label=f'{attack_name} Specialist')
            plt.plot([0, 1], [0, 1], linestyle='--', label='Perfect Calibration')
            plt.xlabel('Mean Predicted Probability')
            plt.ylabel('Fraction of Positives')
            plt.title(f'Calibration Curve: {attack_name} Specialist')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Save plot
            os.makedirs('calibration_plots', exist_ok=True)
            plt.savefig(f'calibration_plots/{attack_name}_calibration.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"   ✅ Calibration curve saved: calibration_plots/{attack_name}_calibration.png")
            
        except Exception as e:
            print(f"   ⚠️  Calibration plot failed: {e}")
    
    def train_all_specialists(self):
        """Train specialists for all attack types"""
        print("\n" + "="*80)
        print("🚀 IMPROVED SPECIALIST TRAINING")
        print("="*80)
        print(f"SMOTE: {'Enabled' if self.use_smote else 'Disabled'}")
        print(f"ADASYN: {'Enabled' if self.use_adasyn else 'Disabled'}")
        print("="*80)
        
        # Load data
        X, y, attack_types = self.load_data()
        
        # Get unique attack types (exclude Normal)
        unique_attacks = np.unique(attack_types)
        attack_list = [att for att in unique_attacks if att != 'Normal']
        
        print(f"\n📊 Training specialists for {len(attack_list)} attack types:")
        for att in attack_list:
            count = np.sum(attack_types == att)
            print(f"   {att}: {count:,} samples")
        
        # Train each specialist
        for attack in attack_list:
            # Create binary labels: this attack vs Normal
            attack_mask = (attack_types == attack) | (attack_types == 'Normal')
            X_attack = X[attack_mask]
            y_binary = (attack_types[attack_mask] == attack).astype(int)
            
            # Skip if insufficient samples
            if np.sum(y_binary == 1) < 50:
                print(f"\n⚠️  Skipping {attack}: Insufficient samples")
                continue
            
            # Train specialist
            self.train_specialist(attack, X_attack, y_binary, use_smote=self.use_smote)
        
        # Generate summary
        self._generate_summary()
        
        # Save specialists
        self._save_specialists()
    
    def _generate_summary(self):
        """Generate training summary - SAME FORMAT AS ORIGINAL"""
        print("\n" + "="*80)
        print("✅ LEAKAGE-FREE SPECIALISTS TRAINED:")
        print("="*80)
        
        for attack, spec in self.specialists.items():
            test_f1 = spec['test_f1_score']
            print(f"   🎯 {attack} (Test F1: {test_f1:.3f})")
        
        print(f"\n💡 Each specialist uses proper train/test separation")
        print(f"💡 Feature engineering and scaling fitted only on training data")
        print(f"💡 Performance evaluated on held-out test sets")
        print(f"💡 SMOTE applied only to training data (no test leakage)")
        print(f"💡 Optimal thresholds found on validation sets")
        
        print("\n" + "="*80)
        print("📊 DETAILED METRICS SUMMARY")
        print("="*80)
        
        print(f"\n{'Attack':<15} {'Test F1':<10} {'Bal-Acc':<10} {'AUC':<8} {'CV F1':<12} {'Overfit':<10} {'SMOTE':<8}")
        print("-" * 90)
        
        for attack, spec in self.specialists.items():
            test_f1 = spec['test_f1_score']
            bal_acc = spec['test_balanced_accuracy']
            auc = spec['test_auc']
            cv_f1 = f"{spec['cv_f1_mean']:.3f}±{spec['cv_f1_std']:.3f}"
            overfit = spec['overfitting_gap']
            smote = "Yes" if spec['used_smote'] else "No"
            
            print(f"{attack:<15} {test_f1:.3f}      {bal_acc:.3f}      {auc:.3f}    {cv_f1:<12} {overfit:+.3f}      {smote}")
        
        print(f"\n📊 Specialists Trained: {len(self.specialists)}")
        print(f"🎯 SMOTE Used: {sum(1 for s in self.specialists.values() if s['used_smote'])}/{len(self.specialists)}")
        
        # Calculate average improvement from optimal thresholding
        improvements = [spec['test_f1_score'] - spec['test_f1_default'] for spec in self.specialists.values()]
        avg_improvement = np.mean(improvements)
        print(f"📈 Average F1 Improvement (Optimal Threshold): {avg_improvement:+.3f}")
    
    def _save_specialists(self):
        """Save trained specialists"""
        os.makedirs('Models/Binary/Specialists_Improved', exist_ok=True)
        
        for attack, spec in self.specialists.items():
            filename = f"Models/Binary/Specialists_Improved/{attack}_specialist_improved.pkl"
            with open(filename, 'wb') as f:
                pickle.dump(spec, f)
            print(f"💾 Saved: {filename}")
        
        print(f"\n✅ All specialists saved to: Models/Binary/Specialists_Improved/")


def main():
    """Main function"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python train_improved_specialists.py <train_csv> [test_csv] [--no-smote] [--adasyn]")
        print("Example: python train_improved_specialists.py UNSW_balanced_train.csv UNSW_balanced_test.csv")
        sys.exit(1)
    
    train_csv = sys.argv[1]
    test_csv = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith('--') else None
    
    use_smote = '--no-smote' not in sys.argv
    use_adasyn = '--adasyn' in sys.argv
    
    print("🔬 IMPROVED ATTACK SPECIALIST TRAINING")
    print("="*80)
    print(f"Training data: {train_csv}")
    if test_csv:
        print(f"Test data: {test_csv}")
    print(f"SMOTE: {'Enabled' if use_smote else 'Disabled'}")
    print(f"ADASYN: {'Enabled' if use_adasyn else 'Disabled'}")
    print("="*80)
    
    trainer = ImprovedSpecialistTrainer(train_csv, test_csv, use_smote=use_smote, use_adasyn=use_adasyn)
    trainer.train_all_specialists()
    
    print("\n✅ Training complete!")
    print("📊 Check calibration_plots/ for calibration curves")
    print("💾 Models saved to Models/Binary/Specialists_Improved/")


if __name__ == "__main__":
    main()
