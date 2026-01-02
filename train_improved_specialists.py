#!/usr/bin/env python3
"""
Attack Specialist Training - Specification Compliant Implementation

Architecture matches the main binary ensemble exactly:
1. 70/30 train-test split (stratified)
2. Full 23-feature engineering pipeline
3. 4-method consensus feature selection
4. SMOTE only for extreme imbalance (ratio > 20:1)
5. GradientBoostingClassifier with imbalance-aware hyperparameters
6. Comprehensive evaluation with overfitting analysis
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.metrics import (f1_score, balanced_accuracy_score, precision_score, 
                            recall_score, accuracy_score, roc_auc_score)
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from imblearn.over_sampling import SMOTE
from collections import defaultdict
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

from novel_ensemble_ml import DynamicFeatureEngineer


class ConsensusFeatureSelector:
    """
    4-method consensus feature selection matching main architecture.
    
    Methods:
    1. Univariate (ANOVA F-test)
    2. Mutual Information
    3. RFE with Random Forest
    4. RFE with SGD Classifier
    
    Features are scored by weighted consensus based on each method's validation accuracy.
    """
    
    def __init__(self):
        self.n_features = None
        self.selected_indices = None
        self.method_scores = {}
        self.feature_scores = None
        
    def _find_optimal_k(self, X, y):
        """Find optimal number of features using validation accuracy."""
        print("   🔍 Finding optimal feature count...")
        
        n_samples, n_total_features = X.shape
        
        # Test k values from 6 to 65 (step 4)
        max_k = min(65, n_total_features)
        k_values = range(6, max_k + 1, 4)
        
        best_score = 0
        optimal_k = min(20, n_total_features)  # Default fallback
        
        # 80/20 sub-train/validation split
        X_sub_train, X_val, y_sub_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        for k in k_values:
            try:
                selector = SelectKBest(f_classif, k=k)
                X_selected = selector.fit_transform(X_sub_train, y_sub_train)
                X_val_selected = selector.transform(X_val)
                
                rf = RandomForestClassifier(n_estimators=50, random_state=42)
                rf.fit(X_selected, y_sub_train)
                score = rf.score(X_val_selected, y_val)
                
                if score > best_score:
                    best_score = score
                    optimal_k = k
                    
            except Exception as e:
                continue
        
        print(f"      ✅ Optimal k={optimal_k} (validation accuracy: {best_score:.4f})")
        return optimal_k
    
    def fit_transform(self, X, y, feature_names):
        """
        Apply 4-method consensus feature selection.
        
        Returns:
            X_selected: Selected features
            selected_indices: Indices of selected features
        """
        print("   🎯 Applying 4-method consensus feature selection...")
        
        # Step 1: Find optimal k
        self.n_features = self._find_optimal_k(X, y)
        
        # Step 2: Initialize selection methods
        selection_methods = {
            'univariate': SelectKBest(f_classif, k=self.n_features),
            'mutual_info': SelectKBest(mutual_info_classif, k=self.n_features),
            'rfe_rf': RFE(
                RandomForestClassifier(n_estimators=50, random_state=42),
                n_features_to_select=self.n_features
            ),
            'rfe_sgd': RFE(
                SGDClassifier(loss='log_loss', random_state=42, max_iter=1000),
                n_features_to_select=self.n_features
            )
        }
        
        # Step 3: Apply each method and score features
        feature_scores = defaultdict(float)
        
        # 80/20 split for method evaluation
        X_train_fs, X_val_fs, y_train_fs, y_val_fs = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        for method_name, selector in selection_methods.items():
            try:
                # Handle NaN values
                X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
                X_train_clean = np.nan_to_num(X_train_fs, nan=0.0, posinf=0.0, neginf=0.0)
                X_val_clean = np.nan_to_num(X_val_fs, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Fit selector
                selector.fit(X_train_clean, y_train_fs)
                selected_mask = selector.get_support()
                
                # Evaluate method accuracy
                X_selected = X_train_clean[:, selected_mask]
                X_val_selected = X_val_clean[:, selected_mask]
                
                rf_eval = RandomForestClassifier(n_estimators=50, random_state=42)
                rf_eval.fit(X_selected, y_train_fs)
                method_accuracy = rf_eval.score(X_val_selected, y_val_fs)
                
                self.method_scores[method_name] = method_accuracy
                
                # Add method's accuracy as weight to selected features
                for i, selected in enumerate(selected_mask):
                    if selected:
                        feature_scores[i] += method_accuracy
                
                print(f"      {method_name}: accuracy={method_accuracy:.4f}")
                
            except Exception as e:
                print(f"      ⚠️ {method_name} failed: {e}")
                continue
        
        # Step 4: Select top k features by cumulative score
        sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
        self.selected_indices = [idx for idx, score in sorted_features[:self.n_features]]
        self.feature_scores = dict(sorted_features)
        
        print(f"      ✅ Selected {len(self.selected_indices)} features by consensus")
        
        return X[:, self.selected_indices], self.selected_indices
    
    def transform(self, X):
        """Apply stored feature selection to new data."""
        if self.selected_indices is None:
            raise ValueError("Must call fit_transform first!")
        return X[:, self.selected_indices]


class SpecificationCompliantSpecialistTrainer:
    """
    Train attack specialists following the exact specification.
    
    Key differences from previous implementation:
    - 70/30 train-test split (was 80/20)
    - 4-method consensus feature selection (was SelectKBest only)
    - SMOTE only for ratio > 20:1 (was ratio > 2)
    - GradientBoostingClassifier (was XGBoost)
    - No optimal threshold (use default 0.5)
    
    Comparison mode:
    - use_smote=True: Apply SMOTE for extreme imbalance (ratio > 20:1)
    - use_smote=False: Skip SMOTE entirely for comparison
    """
    
    def __init__(self, train_csv, test_csv=None, use_smote=True):
        self.train_csv = train_csv
        self.test_csv = test_csv
        self.use_smote = use_smote
        self.specialists = {}
        self.feature_names = None
        self.label_encoders = {}
        
    def load_data(self):
        """Load and preprocess data."""
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
            raise ValueError("Dataset must have 'label' and 'attack_cat' columns")
        
        # Encode categorical features
        categorical_cols = ['proto', 'service', 'state']
        
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
        
        return X, y, attack_types, df

    
    def _get_hyperparameters(self, imbalance_ratio):
        """Get GradientBoostingClassifier hyperparameters based on imbalance ratio."""
        if imbalance_ratio > 100:
            # Extreme imbalance
            return {
                'n_estimators': 100,
                'max_depth': 5,
                'learning_rate': 0.05,
                'min_samples_split': 20,
                'min_samples_leaf': 10,
                'subsample': 0.8,
                'random_state': 42
            }
        elif imbalance_ratio > 20:
            # High imbalance
            return {
                'n_estimators': 150,
                'max_depth': 6,
                'learning_rate': 0.08,
                'min_samples_split': 15,
                'min_samples_leaf': 8,
                'subsample': 0.85,
                'random_state': 42
            }
        else:
            # Moderate imbalance
            return {
                'n_estimators': 200,
                'max_depth': 7,
                'learning_rate': 0.1,
                'min_samples_split': 10,
                'min_samples_leaf': 5,
                'subsample': 1.0,
                'random_state': 42
            }
    
    def train_specialist(self, attack_name, X_raw, y_binary):
        """
        Train a single attack specialist following the specification.
        
        Pipeline:
        1. Create binary dataset (attack vs Normal)
        2. 70/30 stratified train-test split
        3. Full 23-feature engineering
        4. RobustScaler (fit on train only)
        5. 4-method consensus feature selection
        6. SMOTE if ratio > 20:1 (train only)
        7. GradientBoostingClassifier with imbalance-aware hyperparameters
        8. Evaluation on held-out test set
        """
        print(f"\n{'='*80}")
        print(f"🎯 Training {attack_name} Specialist")
        print(f"{'='*80}")
        
        # Count samples
        attack_samples = np.sum(y_binary == 1)
        normal_samples = np.sum(y_binary == 0)
        imbalance_ratio = normal_samples / attack_samples if attack_samples > 0 else float('inf')
        
        print(f"\n📊 Class Distribution:")
        print(f"   {attack_name}: {attack_samples:,} samples")
        print(f"   Normal: {normal_samples:,} samples")
        print(f"   Imbalance Ratio: {imbalance_ratio:.1f}:1")
        
        # Skip if insufficient samples
        if attack_samples < 50 or normal_samples < 50:
            print(f"   ⚠️ Skipping: Insufficient samples (need >= 50 of each class)")
            return None
        
        # STEP 1: 70/30 Stratified Train-Test Split
        print(f"\n1️⃣ Train/Test Split (70/30)...")
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X_raw, y_binary, test_size=0.3, random_state=42, stratify=y_binary
        )
        print(f"   Training: {len(X_train_raw):,} samples")
        print(f"   Testing:  {len(X_test_raw):,} samples")
        
        # STEP 2: Feature Engineering (fit on train only)
        print(f"\n2️⃣ Feature Engineering (23 features)...")
        fe = DynamicFeatureEngineer()
        X_train_df = pd.DataFrame(X_train_raw, columns=self.feature_names)
        X_test_df = pd.DataFrame(X_test_raw, columns=self.feature_names)
        
        X_train_eng = fe.fit_transform(X_train_df)
        X_test_eng = fe.transform(X_test_df)
        
        # Convert to numpy arrays
        if isinstance(X_train_eng, pd.DataFrame):
            eng_feature_names = list(X_train_eng.columns)
            X_train_eng = X_train_eng.values
            X_test_eng = X_test_eng.values
        else:
            eng_feature_names = [f"feature_{i}" for i in range(X_train_eng.shape[1])]
        
        print(f"   Engineered features: {X_train_eng.shape[1]}")
        
        # STEP 3: Feature Scaling (fit on train only)
        print(f"\n3️⃣ Feature Scaling (RobustScaler)...")
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train_eng)
        X_test_scaled = scaler.transform(X_test_eng)
        
        # STEP 4: 4-Method Consensus Feature Selection
        print(f"\n4️⃣ Feature Selection (4-method consensus)...")
        feature_selector = ConsensusFeatureSelector()
        X_train_selected, selected_indices = feature_selector.fit_transform(
            X_train_scaled, y_train, eng_feature_names
        )
        X_test_selected = feature_selector.transform(X_test_scaled)
        print(f"   Selected: {X_train_selected.shape[1]} features")
        
        # STEP 5: SMOTE (only if ratio > 20:1 AND use_smote=True)
        print(f"\n5️⃣ SMOTE (conditional)...")
        used_smote = False
        
        if not self.use_smote:
            print(f"   Skipping SMOTE (--no-smote flag set)")
            X_train_resampled = X_train_selected
            y_train_resampled = y_train
        elif imbalance_ratio > 20:
            try:
                k_neighbors = min(5, attack_samples - 1)
                smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
                X_train_resampled, y_train_resampled = smote.fit_resample(X_train_selected, y_train)
                
                print(f"   Before SMOTE: {len(y_train):,} samples")
                print(f"      {attack_name}: {np.sum(y_train==1):,}")
                print(f"      Normal: {np.sum(y_train==0):,}")
                print(f"   After SMOTE: {len(y_train_resampled):,} samples")
                print(f"      {attack_name}: {np.sum(y_train_resampled==1):,}")
                print(f"      Normal: {np.sum(y_train_resampled==0):,}")
                print(f"   ✅ SMOTE applied (ratio > 20:1)")
                used_smote = True
                
            except Exception as e:
                print(f"   ⚠️ SMOTE failed: {e}")
                X_train_resampled = X_train_selected
                y_train_resampled = y_train
        else:
            print(f"   Skipping SMOTE (ratio {imbalance_ratio:.1f}:1 <= 20:1)")
            X_train_resampled = X_train_selected
            y_train_resampled = y_train
        
        # STEP 6: Train GradientBoostingClassifier
        print(f"\n6️⃣ Training GradientBoostingClassifier...")
        hyperparams = self._get_hyperparameters(imbalance_ratio)
        print(f"   Hyperparameters (imbalance={imbalance_ratio:.1f}:1):")
        for k, v in hyperparams.items():
            print(f"      {k}: {v}")
        
        model = GradientBoostingClassifier(**hyperparams)
        model.fit(X_train_resampled, y_train_resampled)
        print(f"   ✅ Model trained")
        
        # STEP 7: Cross-Validation (on original training data, not SMOTE'd)
        print(f"\n7️⃣ Cross-Validation (5-fold on original training data)...")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        cv_f1 = cross_val_score(model, X_train_selected, y_train, cv=cv, scoring='f1')
        cv_balanced_acc = cross_val_score(model, X_train_selected, y_train, cv=cv, scoring='balanced_accuracy')
        cv_auc = cross_val_score(model, X_train_selected, y_train, cv=cv, scoring='roc_auc')
        
        print(f"   CV F1: {cv_f1.mean():.3f} ± {cv_f1.std():.3f}")
        print(f"   CV Balanced Accuracy: {cv_balanced_acc.mean():.3f} ± {cv_balanced_acc.std():.3f}")
        print(f"   CV AUC-ROC: {cv_auc.mean():.3f} ± {cv_auc.std():.3f}")
        
        # STEP 8: Test Set Evaluation
        print(f"\n8️⃣ Test Set Performance...")
        y_test_pred = model.predict(X_test_selected)
        y_test_proba = model.predict_proba(X_test_selected)[:, 1]
        
        test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
        test_balanced_acc = balanced_accuracy_score(y_test, y_test_pred)
        test_precision = precision_score(y_test, y_test_pred, zero_division=0)
        test_recall = recall_score(y_test, y_test_pred, zero_division=0)
        test_auc = roc_auc_score(y_test, y_test_proba)
        
        print(f"   F1-Score: {test_f1:.3f}")
        print(f"   Balanced Accuracy: {test_balanced_acc:.3f}")
        print(f"   Precision: {test_precision:.3f}")
        print(f"   Recall: {test_recall:.3f}")
        print(f"   AUC-ROC: {test_auc:.3f}")
        
        # STEP 9: Overfitting Analysis
        print(f"\n9️⃣ Overfitting Analysis...")
        y_train_pred = model.predict(X_train_selected)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        cv_accuracy_mean = cv_balanced_acc.mean()
        overfitting_gap = train_accuracy - cv_accuracy_mean
        
        print(f"   Train Accuracy: {train_accuracy:.3f}")
        print(f"   CV Accuracy: {cv_accuracy_mean:.3f}")
        print(f"   Overfitting Gap: {overfitting_gap:.3f}")
        
        if overfitting_gap > 0.1:
            overfitting_flag = "⚠️ HIGH OVERFITTING"
        elif overfitting_gap > 0.05:
            overfitting_flag = "🔶 MODERATE OVERFITTING"
        else:
            overfitting_flag = "✅ GOOD GENERALIZATION"
        print(f"   Status: {overfitting_flag}")
        
        # Store specialist
        specialist = {
            'model': model,
            'feature_engineer': fe,
            'scaler': scaler,
            'feature_indices': selected_indices,
            'metrics': {
                'test_f1': test_f1,
                'test_balanced_acc': test_balanced_acc,
                'test_precision': test_precision,
                'test_recall': test_recall,
                'test_auc': test_auc,
                'cv_f1_mean': cv_f1.mean(),
                'cv_f1_std': cv_f1.std(),
                'cv_balanced_acc_mean': cv_balanced_acc.mean(),
                'cv_balanced_acc_std': cv_balanced_acc.std(),
                'cv_auc_mean': cv_auc.mean(),
                'cv_auc_std': cv_auc.std(),
                'overfitting_gap': overfitting_gap
            },
            'imbalance_ratio': imbalance_ratio,
            'used_smote': used_smote,
            'target_class': attack_name,
            'hyperparameters': hyperparams
        }
        
        self.specialists[attack_name] = specialist
        print(f"\n✅ {attack_name} Specialist trained successfully")
        
        return specialist

    
    def train_all_specialists(self):
        """Train specialists for all attack types."""
        print("\n" + "="*80)
        print("🚀 ATTACK SPECIALIST TRAINING (Specification Compliant)")
        print("="*80)
        print("Architecture:")
        print("  • 70/30 train-test split")
        print("  • Full 23-feature engineering")
        print("  • 4-method consensus feature selection")
        print(f"  • SMOTE: {'ENABLED (ratio > 20:1)' if self.use_smote else 'DISABLED (--no-smote)'}")
        print("  • GradientBoostingClassifier")
        print("="*80)
        
        # Load data
        X, y, attack_types, df = self.load_data()
        
        # Get unique attack types (exclude Normal)
        unique_attacks = np.unique(attack_types)
        attack_list = [att for att in unique_attacks if att != 'Normal']
        
        print(f"\n📊 Training specialists for {len(attack_list)} attack types:")
        for att in attack_list:
            count = np.sum(attack_types == att)
            print(f"   {att}: {count:,} samples")
        
        # Train each specialist
        for attack in attack_list:
            # Create binary dataset: attack vs Normal
            attack_mask = (attack_types == attack) | (attack_types == 'Normal')
            X_attack = X[attack_mask]
            y_binary = (attack_types[attack_mask] == attack).astype(int)
            
            # Train specialist
            self.train_specialist(attack, X_attack, y_binary)
        
        # Generate summary
        self._generate_summary()
        
        # Save specialists
        self._save_specialists()
    
    def _generate_summary(self):
        """Generate training summary."""
        print("\n" + "="*80)
        print("📊 TRAINING SUMMARY")
        print("="*80)
        
        if not self.specialists:
            print("   No specialists trained successfully")
            return
        
        print(f"\n{'Attack':<15} {'Test F1':<10} {'Bal-Acc':<10} {'AUC':<8} {'CV F1':<15} {'Overfit':<10} {'SMOTE':<8}")
        print("-" * 90)
        
        for attack, spec in self.specialists.items():
            metrics = spec['metrics']
            test_f1 = metrics['test_f1']
            bal_acc = metrics['test_balanced_acc']
            auc = metrics['test_auc']
            cv_f1 = f"{metrics['cv_f1_mean']:.3f}±{metrics['cv_f1_std']:.3f}"
            overfit = metrics['overfitting_gap']
            smote = "Yes" if spec['used_smote'] else "No"
            
            print(f"{attack:<15} {test_f1:.3f}      {bal_acc:.3f}      {auc:.3f}    {cv_f1:<15} {overfit:+.3f}      {smote}")
        
        # Summary statistics
        avg_f1 = np.mean([s['metrics']['test_f1'] for s in self.specialists.values()])
        avg_auc = np.mean([s['metrics']['test_auc'] for s in self.specialists.values()])
        smote_count = sum(1 for s in self.specialists.values() if s['used_smote'])
        
        print(f"\n📊 Summary:")
        print(f"   Specialists Trained: {len(self.specialists)}")
        print(f"   Average Test F1: {avg_f1:.3f}")
        print(f"   Average Test AUC: {avg_auc:.3f}")
        print(f"   SMOTE Applied: {smote_count}/{len(self.specialists)}")
    
    def _save_specialists(self):
        """Save trained specialists."""
        # Use different folder based on SMOTE setting
        folder_suffix = "_with_smote" if self.use_smote else "_no_smote"
        save_dir = f'Models/Binary/Specialists{folder_suffix}'
        os.makedirs(save_dir, exist_ok=True)
        
        for attack, spec in self.specialists.items():
            filename = f"{save_dir}/{attack}_specialist.pkl"
            with open(filename, 'wb') as f:
                pickle.dump(spec, f)
            print(f"💾 Saved: {filename}")
        
        # Save summary
        summary = {
            attack: {
                'metrics': spec['metrics'],
                'imbalance_ratio': spec['imbalance_ratio'],
                'used_smote': spec['used_smote'],
                'hyperparameters': spec['hyperparameters']
            }
            for attack, spec in self.specialists.items()
        }
        
        with open(f'{save_dir}/summary.pkl', 'wb') as f:
            pickle.dump(summary, f)
        
        print(f"\n✅ All specialists saved to: {save_dir}/")


def main():
    """Main function."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python train_improved_specialists.py <train_csv> [test_csv] [--no-smote]")
        print("")
        print("Options:")
        print("  --no-smote    Skip SMOTE entirely (for comparison)")
        print("")
        print("Examples:")
        print("  # With SMOTE (default, applies for ratio > 20:1):")
        print("  python train_improved_specialists.py UNSW_balanced_train.csv UNSW_balanced_test.csv")
        print("")
        print("  # Without SMOTE (for comparison):")
        print("  python train_improved_specialists.py UNSW_balanced_train.csv UNSW_balanced_test.csv --no-smote")
        sys.exit(1)
    
    train_csv = sys.argv[1]
    test_csv = None
    use_smote = True
    
    # Parse arguments
    for arg in sys.argv[2:]:
        if arg == '--no-smote':
            use_smote = False
        elif not arg.startswith('--'):
            test_csv = arg
    
    print("🔬 ATTACK SPECIALIST TRAINING")
    print("="*80)
    print(f"Training data: {train_csv}")
    if test_csv:
        print(f"Test data: {test_csv}")
    print(f"SMOTE: {'ENABLED (ratio > 20:1)' if use_smote else 'DISABLED'}")
    print("="*80)
    
    trainer = SpecificationCompliantSpecialistTrainer(train_csv, test_csv, use_smote=use_smote)
    trainer.train_all_specialists()
    
    print("\n✅ Training complete!")
    folder_suffix = "_with_smote" if use_smote else "_no_smote"
    print(f"💾 Models saved to Models/Binary/Specialists{folder_suffix}/")


if __name__ == "__main__":
    main()
