#!/usr/bin/env python3
"""
Ablation Study Implementation - Following ablation_study_corrected_spec.md

This implements the complete ablation study with:
- 14 configurations for Binary/Multiclass (B1a-B5b, E1-E4)
- 4 configurations per attack for Specialists (S1-S4)
- Cached models for ensemble configs (E1-E4, S1)
- Fresh training for GB baselines (B1-B5) and specialist variants (S2-S4)
- Consistent 5-fold CV with statistical testing

Feature Groups (23 total, engineered by DynamicFeatureEngineer in novel_ensemble_ml.py):
- Statistical (7): total_bytes, byte_ratio, byte_imbalance, log_total_bytes, 
                   total_packets, packet_ratio, avg_packet_size
- Duration (5): log_duration, is_short_connection, is_long_connection, throughput, log_throughput
- Network (9): total_loss, loss_ratio, has_loss, total_jitter, jitter_ratio, 
               high_jitter, window_ratio, min_window, max_window
- Interaction (2): proto_service_encoded, state_proto_encoded
"""

import pandas as pd
import numpy as np
import os
import pickle
import warnings
from collections import defaultdict
from scipy import stats
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score, precision_score, recall_score

warnings.filterwarnings('ignore')

# Import from existing codebase
from novel_ensemble_ml import DynamicFeatureEngineer, NovelEnsembleMLSystem


# =============================================================================
# FEATURE GROUP DEFINITIONS
# These features are created by DynamicFeatureEngineer in novel_ensemble_ml.py
# The ablation study tests each group's contribution to model performance
# =============================================================================

# Statistical Features (7) - Byte and packet statistics
# Created by: create_statistical_features() in DynamicFeatureEngineer
STATISTICAL_FEATURES = [
    'total_bytes',      # sbytes + dbytes
    'byte_ratio',       # sbytes / dbytes
    'byte_imbalance',   # |sbytes - dbytes|
    'log_total_bytes',  # log1p(total_bytes)
    'total_packets',    # spkts + dpkts
    'packet_ratio',     # spkts / dpkts
    'avg_packet_size'   # total_bytes / total_packets
]

# Duration Features (5) - Connection duration and throughput
# Created by: create_statistical_features() in DynamicFeatureEngineer
DURATION_FEATURES = [
    'log_duration',         # log1p(dur)
    'is_short_connection',  # dur < threshold (data-driven)
    'is_long_connection',   # dur > threshold (data-driven)
    'throughput',           # total_bytes / dur
    'log_throughput'        # log1p(throughput)
]

# Network Features (9) - Loss, jitter, and window patterns
# Created by: create_network_behavior_features() in DynamicFeatureEngineer
NETWORK_FEATURES = [
    'total_loss',    # sloss + dloss
    'loss_ratio',    # sloss / dloss
    'has_loss',      # (total_loss > 0)
    'total_jitter',  # sjit + djit
    'jitter_ratio',  # sjit / djit
    'high_jitter',   # total_jitter > threshold (data-driven)
    'window_ratio',  # swin / dwin
    'min_window',    # min(swin, dwin)
    'max_window'     # max(swin, dwin)
]

# Interaction Features (2) - Protocol/service/state combinations
# Created by: create_interaction_features() in DynamicFeatureEngineer
INTERACTION_FEATURES = [
    'proto_service_encoded',  # LabelEncoded(proto + '_' + service)
    'state_proto_encoded'     # LabelEncoded(state + '_' + proto)
]

# All 23 engineered features for ablation study
ALL_ENGINEERED_FEATURES = STATISTICAL_FEATURES + DURATION_FEATURES + NETWORK_FEATURES + INTERACTION_FEATURES

# Note: DynamicFeatureEngineer also creates temporal features (hour, day_of_week, etc.)
# and port features (src_is_wellknown, etc.) but these are not part of the 23-feature
# ablation study specification


# =============================================================================
# CONSENSUS FEATURE SELECTOR (4-method)
# =============================================================================

class ConsensusFeatureSelector:
    """4-method consensus feature selection matching main architecture."""
    
    def __init__(self, n_features=None):
        self.n_features = n_features
        self.selected_indices = None
        
    def fit_transform(self, X, y, feature_names=None):
        """Apply 4-method consensus feature selection."""
        n_samples, n_total_features = X.shape
        
        # Find optimal k if not specified
        if self.n_features is None:
            self.n_features = self._find_optimal_k(X, y)
        
        self.n_features = min(self.n_features, n_total_features)
        
        # 4-method selection
        selection_methods = {
            'univariate': SelectKBest(f_classif, k=self.n_features),
            'mutual_info': SelectKBest(mutual_info_classif, k=self.n_features),
            'rfe_rf': RFE(RandomForestClassifier(n_estimators=50, random_state=42), 
                         n_features_to_select=self.n_features),
            'rfe_sgd': RFE(SGDClassifier(loss='log_loss', random_state=42, max_iter=1000),
                          n_features_to_select=self.n_features)
        }
        
        feature_scores = defaultdict(float)
        X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        for method_name, selector in selection_methods.items():
            try:
                selector.fit(X_clean, y)
                selected_mask = selector.get_support()
                for i, selected in enumerate(selected_mask):
                    if selected:
                        feature_scores[i] += 1.0
            except Exception:
                continue
        
        sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
        self.selected_indices = [idx for idx, _ in sorted_features[:self.n_features]]
        
        return X[:, self.selected_indices], self.selected_indices
    
    def _find_optimal_k(self, X, y):
        """Find optimal number of features."""
        from sklearn.model_selection import train_test_split
        
        n_total = X.shape[1]
        max_k = min(65, n_total)
        k_values = range(6, max_k + 1, 4)
        
        best_score = 0
        optimal_k = min(20, n_total)
        
        X_sub, X_val, y_sub, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        for k in k_values:
            try:
                selector = SelectKBest(f_classif, k=k)
                X_sel = selector.fit_transform(X_sub, y_sub)
                X_val_sel = selector.transform(X_val)
                
                rf = RandomForestClassifier(n_estimators=50, random_state=42)
                rf.fit(X_sel, y_sub)
                score = rf.score(X_val_sel, y_val)
                
                if score > best_score:
                    best_score = score
                    optimal_k = k
            except Exception:
                continue
        
        return optimal_k
    
    def transform(self, X):
        """Apply stored selection."""
        if self.selected_indices is None:
            raise ValueError("Must call fit_transform first!")
        return X[:, self.selected_indices]


# =============================================================================
# PARTIAL FEATURE ENGINEER
# =============================================================================

class PartialFeatureEngineer:
    """
    Feature engineer that creates only specified feature groups.
    
    Uses DynamicFeatureEngineer from novel_ensemble_ml.py to create all features,
    then filters to keep only the requested groups.
    
    Feature Groups:
    - Statistical (7): Byte and packet statistics
    - Duration (5): Connection duration and throughput  
    - Network (9): Loss, jitter, and window patterns
    - Interaction (2): Protocol/service/state combinations
    """
    
    def __init__(self, include_statistical=False, include_duration=False, 
                 include_network=False, include_interaction=False):
        self.include_statistical = include_statistical
        self.include_duration = include_duration
        self.include_network = include_network
        self.include_interaction = include_interaction
        self.full_engineer = DynamicFeatureEngineer()
        self.is_fitted = False
        
    def fit_transform(self, df):
        """Apply feature engineering with only specified groups."""
        # First apply full engineering to get all features
        df_full = self.full_engineer.fit_transform(df)
        self.is_fitted = True
        
        # Determine which engineered features to keep
        features_to_keep = []
        if self.include_statistical:
            features_to_keep.extend([f for f in STATISTICAL_FEATURES if f in df_full.columns])
        if self.include_duration:
            features_to_keep.extend([f for f in DURATION_FEATURES if f in df_full.columns])
        if self.include_network:
            features_to_keep.extend([f for f in NETWORK_FEATURES if f in df_full.columns])
        if self.include_interaction:
            features_to_keep.extend([f for f in INTERACTION_FEATURES if f in df_full.columns])
        
        # Get original columns (non-engineered)
        original_cols = [col for col in df.columns if col not in ALL_ENGINEERED_FEATURES]
        
        # Build final column list
        final_cols = []
        for col in df_full.columns:
            if col in original_cols:
                final_cols.append(col)
            elif col in features_to_keep:
                final_cols.append(col)
        
        return df_full[final_cols]
    
    def transform(self, df):
        """Apply same transformation."""
        if not self.is_fitted:
            raise ValueError("Must call fit_transform first!")
        
        df_full = self.full_engineer.transform(df)
        
        features_to_keep = []
        if self.include_statistical:
            features_to_keep.extend([f for f in STATISTICAL_FEATURES if f in df_full.columns])
        if self.include_duration:
            features_to_keep.extend([f for f in DURATION_FEATURES if f in df_full.columns])
        if self.include_network:
            features_to_keep.extend([f for f in NETWORK_FEATURES if f in df_full.columns])
        if self.include_interaction:
            features_to_keep.extend([f for f in INTERACTION_FEATURES if f in df_full.columns])
        
        original_cols = [col for col in df.columns if col not in ALL_ENGINEERED_FEATURES]
        
        final_cols = []
        for col in df_full.columns:
            if col in original_cols:
                final_cols.append(col)
            elif col in features_to_keep:
                final_cols.append(col)
        
        return df_full[final_cols]



# =============================================================================
# ABLATION STUDY RUNNER
# =============================================================================

class AblationStudy:
    """
    Complete ablation study implementation following the specification.
    
    Configurations:
    - B1a/B1b: No engineered features (baseline)
    - B2a/B2b: Statistical features only
    - B3a/B3b: Interaction features only
    - B4a/B4b: Behavioral features only
    - B5a/B5b: All 23 engineered features
    - E1-E4: Full ensemble (cached)
    
    'a' suffix = no feature selection
    'b' suffix = with feature selection
    """
    
    def __init__(self, train_csv, test_csv=None, classification_type='binary', cache_dir='Models'):
        self.train_csv = train_csv
        self.test_csv = test_csv
        self.classification_type = classification_type
        self.cache_dir = cache_dir
        
        # CV setup (consistent across all configs)
        self.cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Results storage
        self.results = {}
        self.cv_scores = {}
        
        # Data storage
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.feature_names = None
        self.label_encoders = {}
        
    def load_data(self):
        """Load and preprocess data."""
        print("📁 Loading data...")
        
        df_train = pd.read_csv(self.train_csv)
        
        if self.test_csv:
            df_test = pd.read_csv(self.test_csv)
        else:
            # Split training data
            from sklearn.model_selection import train_test_split
            df_train, df_test = train_test_split(df_train, test_size=0.3, random_state=42)
        
        print(f"   Training samples: {len(df_train):,}")
        print(f"   Test samples: {len(df_test):,}")
        
        return df_train, df_test
    
    def preprocess_data(self, df):
        """Basic preprocessing without feature engineering."""
        df = df.copy()
        df = df.fillna(0)
        
        # Encode categorical columns
        categorical_cols = ['proto', 'service', 'state']
        for col in categorical_cols:
            if col in df.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    # Handle unseen categories
                    try:
                        df[col] = self.label_encoders[col].transform(df[col].astype(str))
                    except ValueError:
                        df[col] = 0
        
        return df
    
    def get_features_and_labels(self, df, engineered=False):
        """Extract features and labels from dataframe."""
        # Determine target column
        if self.classification_type == 'binary':
            target_col = 'label' if 'label' in df.columns else 'attack'
            y = df[target_col].values
        else:
            if 'attack_cat' in df.columns:
                if not hasattr(self, 'attack_encoder'):
                    self.attack_encoder = LabelEncoder()
                    y = self.attack_encoder.fit_transform(df['attack_cat'].astype(str))
                else:
                    y = self.attack_encoder.transform(df['attack_cat'].astype(str))
            else:
                raise ValueError("No attack_cat column for multiclass")
        
        # Get feature columns
        exclude_cols = ['label', 'attack', 'attack_cat', 'stime', 'srcip', 'dstip', 'id']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].values
        
        return X, y, feature_cols
    
    def get_cv_scores(self, model, X, y, scoring='accuracy'):
        """Get cross-validation scores."""
        return cross_val_score(model, X, y, cv=self.cv, scoring=scoring)
    
    def compare_configs(self, cv_a, cv_b, name_a, name_b):
        """
        Comprehensive statistical comparison between two configurations.
        
        Includes:
        1. Shapiro-Wilk normality test
        2. Adaptive test selection (t-test vs Wilcoxon)
        3. Cohen's d effect size with interpretation
        """
        from scipy.stats import shapiro, wilcoxon
        
        cv_a = np.array(cv_a)
        cv_b = np.array(cv_b)
        
        # 1. Normality check (Shapiro-Wilk)
        _, p_norm_a = shapiro(cv_a)
        _, p_norm_b = shapiro(cv_b)
        both_normal = (p_norm_a > 0.05) and (p_norm_b > 0.05)
        
        # 2. Adaptive test selection
        if both_normal and len(cv_a) >= 5:
            t_stat, p_value = stats.ttest_rel(cv_a, cv_b)
            test_used = "Paired t-test"
        else:
            t_stat, p_value = wilcoxon(cv_a, cv_b)
            test_used = "Wilcoxon signed-rank"
        
        # 3. Effect size (Cohen's d)
        diff = cv_a - cv_b
        cohens_d = np.mean(diff) / (np.std(diff, ddof=1) + 1e-10)
        
        # Effect size interpretation
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            effect_interp = "Negligible"
        elif abs_d < 0.5:
            effect_interp = "Small"
        elif abs_d < 0.8:
            effect_interp = "Medium"
        else:
            effect_interp = "Large"
        
        return {
            'comparison': f"{name_a} vs {name_b}",
            'test_used': test_used,
            'normality_a': p_norm_a,
            'normality_b': p_norm_b,
            'both_normal': both_normal,
            'mean_a': np.mean(cv_a),
            'mean_b': np.mean(cv_b),
            'std_a': np.std(cv_a, ddof=1),
            'std_b': np.std(cv_b, ddof=1),
            'diff': np.mean(diff),
            't_stat': t_stat,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'effect_interpretation': effect_interp,
            'significant_raw': p_value < 0.05
        }
    
    # =========================================================================
    # BASELINE CONFIGURATIONS (B1-B5) - Train Fresh GB
    # =========================================================================
    
    def run_baseline_config(self, config_name, df_train, df_test, 
                           include_statistical=False, include_duration=False,
                           include_network=False, include_interaction=False, 
                           use_selection=False):
        """Run a single baseline configuration with Gradient Boosting."""
        print(f"\n{'='*60}")
        print(f"🔬 Configuration: {config_name}")
        print(f"{'='*60}")
        
        # Feature engineering
        if include_statistical or include_duration or include_network or include_interaction:
            print(f"   Feature Engineering (via DynamicFeatureEngineer):")
            print(f"      Statistical (7): {include_statistical}")
            print(f"      Duration (5): {include_duration}")
            print(f"      Network (9): {include_network}")
            print(f"      Interaction (2): {include_interaction}")
            
            fe = PartialFeatureEngineer(
                include_statistical=include_statistical,
                include_duration=include_duration,
                include_network=include_network,
                include_interaction=include_interaction
            )
            df_train_eng = fe.fit_transform(df_train)
            df_test_eng = fe.transform(df_test)
        else:
            print(f"   Feature Engineering: None (baseline)")
            df_train_eng = df_train.copy()
            df_test_eng = df_test.copy()
        
        # Get features and labels
        X_train, y_train, feature_names = self.get_features_and_labels(df_train_eng)
        X_test, y_test, _ = self.get_features_and_labels(df_test_eng)
        
        print(f"   Features: {X_train.shape[1]}")
        
        # Scale features
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Feature selection (if enabled)
        if use_selection:
            print(f"   Feature Selection: Yes (4-method consensus)")
            selector = ConsensusFeatureSelector()
            X_train_selected, selected_indices = selector.fit_transform(X_train_scaled, y_train, feature_names)
            X_test_selected = selector.transform(X_test_scaled)
            print(f"   Selected Features: {X_train_selected.shape[1]}")
        else:
            print(f"   Feature Selection: No")
            X_train_selected = X_train_scaled
            X_test_selected = X_test_scaled
        
        # Train Gradient Boosting
        print(f"   Training: GradientBoostingClassifier (fresh)")
        
        n_classes = len(np.unique(y_train))
        if n_classes > 2:
            gb = GradientBoostingClassifier(
                n_estimators=50, max_depth=6, learning_rate=0.1, random_state=42
            )
        else:
            gb = GradientBoostingClassifier(
                n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42
            )
        
        # Cross-validation
        print(f"   Cross-Validation: 5-fold")
        cv_accuracy = self.get_cv_scores(gb, X_train_selected, y_train, 'accuracy')
        cv_f1 = self.get_cv_scores(gb, X_train_selected, y_train, 
                                   'f1' if n_classes == 2 else 'f1_macro')
        
        print(f"      CV Accuracy: {cv_accuracy.mean():.4f} ± {cv_accuracy.std():.4f}")
        print(f"      CV F1: {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")
        
        # Train final model and evaluate on test set
        gb.fit(X_train_selected, y_train)
        y_pred = gb.predict(X_test_selected)
        
        # Get probabilities for AUC
        y_proba = gb.predict_proba(X_test_selected)
        
        # Compute all metrics based on classification type
        test_accuracy = accuracy_score(y_test, y_pred)
        test_balanced_acc = balanced_accuracy_score(y_test, y_pred)
        
        if n_classes == 2:
            # Binary classification metrics
            test_precision = precision_score(y_test, y_pred, zero_division=0)
            test_recall = recall_score(y_test, y_pred, zero_division=0)
            test_f1 = f1_score(y_test, y_pred, zero_division=0)
            test_auc = roc_auc_score(y_test, y_proba[:, 1])
            
            print(f"   Test Metrics (Binary):")
            print(f"      Accuracy: {test_accuracy:.4f}")
            print(f"      Precision: {test_precision:.4f}")
            print(f"      Recall: {test_recall:.4f}")
            print(f"      F1-Score: {test_f1:.4f}")
            print(f"      AUC-ROC: {test_auc:.4f}")
            print(f"      Balanced Accuracy: {test_balanced_acc:.4f}")
        else:
            # Multiclass classification metrics
            test_precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
            test_recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
            test_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
            test_f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            try:
                test_auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')
            except ValueError:
                test_auc = 0.0
            
            print(f"   Test Metrics (Multiclass):")
            print(f"      Accuracy: {test_accuracy:.4f}")
            print(f"      Macro Precision: {test_precision:.4f}")
            print(f"      Macro Recall: {test_recall:.4f}")
            print(f"      Macro F1: {test_f1:.4f}")
            print(f"      Weighted F1: {test_f1_weighted:.4f}")
            print(f"      Macro AUC (OVR): {test_auc:.4f}")
            print(f"      Balanced Accuracy: {test_balanced_acc:.4f}")
        
        # Store results
        result = {
            'config': config_name,
            'cv_accuracy': cv_accuracy,
            'cv_accuracy_mean': cv_accuracy.mean(),
            'cv_accuracy_std': cv_accuracy.std(),
            'cv_f1': cv_f1,
            'cv_f1_mean': cv_f1.mean(),
            'cv_f1_std': cv_f1.std(),
            'test_accuracy': test_accuracy,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1': test_f1,
            'test_auc': test_auc,
            'test_balanced_acc': test_balanced_acc,
            'n_features': X_train_selected.shape[1],
            'model_type': 'GradientBoosting',
            'source': 'fresh'
        }
        
        # Add weighted F1 for multiclass
        if n_classes > 2:
            result['test_f1_weighted'] = test_f1_weighted
        
        self.results[config_name] = result
        self.cv_scores[config_name] = cv_accuracy
        
        return result

    
    # =========================================================================
    # ENSEMBLE CONFIGURATIONS (E1-E4) - Use Cached Models
    # =========================================================================
    
    def run_ensemble_config(self, config_name, df_train, df_test, 
                           use_selection=False, use_optimized_mixing=False):
        """Run ensemble configuration using cached models."""
        print(f"\n{'='*60}")
        print(f"🔬 Configuration: {config_name} (CACHED ENSEMBLE)")
        print(f"{'='*60}")
        
        # Load cached ensemble
        cache_subdir = "Binary" if self.classification_type == "binary" else "Multiclass"
        ensemble_cache_dir = os.path.join(self.cache_dir, cache_subdir)
        
        # Find cached model
        ensemble_path = None
        if os.path.exists(ensemble_cache_dir):
            import glob
            pattern = os.path.join(ensemble_cache_dir, f"ensemble_{self.classification_type}_*.pkl")
            cached_models = glob.glob(pattern)
            if cached_models:
                ensemble_path = max(cached_models, key=os.path.getmtime)
        
        # Fallback to main model file
        if ensemble_path is None and os.path.exists('trained_novel_ensemble_model.pkl'):
            ensemble_path = 'trained_novel_ensemble_model.pkl'
        
        if ensemble_path is None:
            print(f"   ❌ No cached ensemble found!")
            print(f"   💡 Run: python run_novel_ml.py --dataset {self.train_csv} first")
            return None
        
        print(f"   📦 Loading cached ensemble: {os.path.basename(ensemble_path)}")
        
        with open(ensemble_path, 'rb') as f:
            cached = pickle.load(f)
        
        # Extract components
        classifier = cached['classifier']
        feature_engineer = cached['feature_engineer']
        scaler = cached['scaler']
        feature_selector = cached.get('feature_selector', None)
        selected_indices = cached.get('selected_feature_indices', None)
        stored_feature_names = cached.get('feature_names', None)
        
        print(f"   ✅ Ensemble loaded successfully")
        
        # Get mixing ratio info
        if hasattr(classifier, 'optimal_mixing_ratio'):
            stored_ratio = classifier.optimal_mixing_ratio
            print(f"   📊 Stored mixing ratio: {stored_ratio:.2f}")
        else:
            stored_ratio = 0.6
            print(f"   📊 No stored mixing ratio, using default: {stored_ratio:.2f}")
        
        # Apply feature engineering
        print(f"   Feature Engineering: All 23 features (from cache)")
        df_train_eng = feature_engineer.fit_transform(self.preprocess_data(df_train))
        df_test_eng = feature_engineer.transform(self.preprocess_data(df_test))
        
        # Get features
        X_train, y_train, feature_names = self.get_features_and_labels(df_train_eng)
        X_test, y_test, _ = self.get_features_and_labels(df_test_eng)
        
        # Scale
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Feature selection
        if use_selection and selected_indices is not None:
            print(f"   Feature Selection: Yes (using cached indices)")
            X_train_selected = X_train_scaled[:, selected_indices]
            X_test_selected = X_test_scaled[:, selected_indices]
            print(f"   Selected Features: {X_train_selected.shape[1]}")
        else:
            print(f"   Feature Selection: No")
            X_train_selected = X_train_scaled
            X_test_selected = X_test_scaled
        
        # Set mixing ratio
        if use_optimized_mixing:
            print(f"   Mixing Ratio: Optimized ({stored_ratio:.2f})")
            # Keep the stored optimized ratio
        else:
            print(f"   Mixing Ratio: α=1.0 (pure meta-learner)")
            classifier.optimal_mixing_ratio = 1.0
        
        # Get CV scores from cached classifier if available
        if hasattr(classifier, 'ensemble_cv_scores') and classifier.ensemble_cv_scores is not None:
            cv_accuracy = np.array(classifier.ensemble_cv_scores)
            print(f"   CV Accuracy (cached): {cv_accuracy.mean():.4f} ± {cv_accuracy.std():.4f}")
        else:
            # Compute CV scores
            print(f"   Computing CV scores...")
            cv_accuracy = self.get_cv_scores(classifier, X_train_selected, y_train, 'accuracy')
            print(f"   CV Accuracy: {cv_accuracy.mean():.4f} ± {cv_accuracy.std():.4f}")
        
        # Test evaluation
        y_pred = classifier.predict(X_test_selected)
        
        # Get probabilities for AUC
        y_proba = None
        if hasattr(classifier, 'predict_proba'):
            try:
                y_proba = classifier.predict_proba(X_test_selected)
            except:
                pass
        
        n_classes = len(np.unique(y_train))
        test_accuracy = accuracy_score(y_test, y_pred)
        test_balanced_acc = balanced_accuracy_score(y_test, y_pred)
        
        if n_classes == 2:
            # Binary classification metrics
            test_precision = precision_score(y_test, y_pred, zero_division=0)
            test_recall = recall_score(y_test, y_pred, zero_division=0)
            test_f1 = f1_score(y_test, y_pred, zero_division=0)
            if y_proba is not None:
                test_auc = roc_auc_score(y_test, y_proba[:, 1])
            else:
                test_auc = 0.0
            
            print(f"   Test Metrics (Binary):")
            print(f"      Accuracy: {test_accuracy:.4f}")
            print(f"      Precision: {test_precision:.4f}")
            print(f"      Recall: {test_recall:.4f}")
            print(f"      F1-Score: {test_f1:.4f}")
            print(f"      AUC-ROC: {test_auc:.4f}")
            print(f"      Balanced Accuracy: {test_balanced_acc:.4f}")
        else:
            # Multiclass classification metrics
            test_precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
            test_recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
            test_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
            test_f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            if y_proba is not None:
                try:
                    test_auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')
                except ValueError:
                    test_auc = 0.0
            else:
                test_auc = 0.0
            
            print(f"   Test Metrics (Multiclass):")
            print(f"      Accuracy: {test_accuracy:.4f}")
            print(f"      Macro Precision: {test_precision:.4f}")
            print(f"      Macro Recall: {test_recall:.4f}")
            print(f"      Macro F1: {test_f1:.4f}")
            print(f"      Weighted F1: {test_f1_weighted:.4f}")
            print(f"      Macro AUC (OVR): {test_auc:.4f}")
            print(f"      Balanced Accuracy: {test_balanced_acc:.4f}")
        
        # Restore original mixing ratio if we changed it
        if not use_optimized_mixing:
            classifier.optimal_mixing_ratio = stored_ratio
        
        # Store results
        result = {
            'config': config_name,
            'cv_accuracy': cv_accuracy,
            'cv_accuracy_mean': cv_accuracy.mean(),
            'cv_accuracy_std': cv_accuracy.std(),
            'test_accuracy': test_accuracy,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1': test_f1,
            'test_auc': test_auc,
            'test_balanced_acc': test_balanced_acc,
            'n_features': X_train_selected.shape[1],
            'model_type': 'Ensemble',
            'mixing_ratio': classifier.optimal_mixing_ratio if use_optimized_mixing else 1.0,
            'source': 'cached'
        }
        
        # Add weighted F1 for multiclass
        if n_classes > 2:
            result['test_f1_weighted'] = test_f1_weighted
        
        self.results[config_name] = result
        self.cv_scores[config_name] = cv_accuracy
        
        return result
    
    # =========================================================================
    # RUN ALL BASELINE ABLATIONS
    # =========================================================================
    
    def run_all_baselines(self):
        """Run all baseline configurations (B1-B5, a and b variants)."""
        print("\n" + "="*80)
        print("🔬 RUNNING BASELINE ABLATIONS (GB, Train Fresh)")
        print("="*80)
        
        df_train, df_test = self.load_data()
        df_train = self.preprocess_data(df_train)
        df_test = self.preprocess_data(df_test)
        
        # B1: No engineered features (baseline)
        self.run_baseline_config('B1a', df_train, df_test, 
                                use_selection=False)
        self.run_baseline_config('B1b', df_train, df_test, 
                                use_selection=True)
        
        # B2: Statistical features only (7 features)
        self.run_baseline_config('B2a', df_train, df_test, 
                                include_statistical=True, use_selection=False)
        self.run_baseline_config('B2b', df_train, df_test, 
                                include_statistical=True, use_selection=True)
        
        # B3: Duration features only (5 features)
        self.run_baseline_config('B3a', df_train, df_test, 
                                include_duration=True, use_selection=False)
        self.run_baseline_config('B3b', df_train, df_test, 
                                include_duration=True, use_selection=True)
        
        # B4: Network features only (9 features)
        self.run_baseline_config('B4a', df_train, df_test, 
                                include_network=True, use_selection=False)
        self.run_baseline_config('B4b', df_train, df_test, 
                                include_network=True, use_selection=True)
        
        # B5: Interaction features only (2 features)
        self.run_baseline_config('B5a', df_train, df_test, 
                                include_interaction=True, use_selection=False)
        self.run_baseline_config('B5b', df_train, df_test, 
                                include_interaction=True, use_selection=True)
        
        # B6: All 23 engineered features
        self.run_baseline_config('B6a', df_train, df_test, 
                                include_statistical=True, include_duration=True, 
                                include_network=True, include_interaction=True,
                                use_selection=False)
        self.run_baseline_config('B6b', df_train, df_test, 
                                include_statistical=True, include_duration=True, 
                                include_network=True, include_interaction=True,
                                use_selection=True)
        
        return self.results
    
    # =========================================================================
    # RUN ALL ENSEMBLE ABLATIONS
    # =========================================================================
    
    def run_all_ensembles(self):
        """Run all ensemble configurations (E1-E4) using cached models."""
        print("\n" + "="*80)
        print("🔬 RUNNING ENSEMBLE ABLATIONS (Cached Models)")
        print("="*80)
        
        df_train, df_test = self.load_data()
        
        # E1: All features, no selection, α=1.0
        self.run_ensemble_config('E1', df_train, df_test, 
                                use_selection=False, use_optimized_mixing=False)
        
        # E2: All features, no selection, optimized mixing
        self.run_ensemble_config('E2', df_train, df_test, 
                                use_selection=False, use_optimized_mixing=True)
        
        # E3: All features, with selection, α=1.0
        self.run_ensemble_config('E3', df_train, df_test, 
                                use_selection=True, use_optimized_mixing=False)
        
        # E4: All features, with selection, optimized mixing
        self.run_ensemble_config('E4', df_train, df_test, 
                                use_selection=True, use_optimized_mixing=True)
        
        return self.results

    
    # =========================================================================
    # SPECIALIST ABLATIONS (S1-S4)
    # =========================================================================
    
    def run_specialist_ablations(self):
        """Run specialist ablations for each attack type."""
        print("\n" + "="*80)
        print("🔬 RUNNING SPECIALIST ABLATIONS")
        print("="*80)
        
        df_train, df_test = self.load_data()
        df_train = self.preprocess_data(df_train)
        df_test = self.preprocess_data(df_test)
        
        # Get attack types
        if 'attack_cat' not in df_train.columns:
            print("   ❌ No attack_cat column - skipping specialist ablations")
            return {}
        
        attack_types = df_train['attack_cat'].unique()
        attack_list = [att for att in attack_types if att != 'Normal']
        
        print(f"   Attack types: {len(attack_list)}")
        
        specialist_results = {}
        
        for attack in attack_list:
            print(f"\n{'='*60}")
            print(f"🎯 Attack: {attack}")
            print(f"{'='*60}")
            
            # Create binary dataset
            train_mask = (df_train['attack_cat'] == attack) | (df_train['attack_cat'] == 'Normal')
            test_mask = (df_test['attack_cat'] == attack) | (df_test['attack_cat'] == 'Normal')
            
            df_attack_train = df_train[train_mask].copy()
            df_attack_test = df_test[test_mask].copy()
            
            y_train = (df_attack_train['attack_cat'] == attack).astype(int).values
            y_test = (df_attack_test['attack_cat'] == attack).astype(int).values
            
            attack_samples = np.sum(y_train == 1)
            normal_samples = np.sum(y_train == 0)
            
            print(f"   Samples: {attack_samples} attack, {normal_samples} normal")
            
            if attack_samples < 50 or normal_samples < 50:
                print(f"   ⚠️ Skipping - insufficient samples")
                continue
            
            # Run S1-S4 configurations
            attack_results = self._run_specialist_configs(
                attack, df_attack_train, df_attack_test, y_train, y_test
            )
            
            specialist_results[attack] = attack_results
        
        self.results['specialists'] = specialist_results
        return specialist_results
    
    def _run_specialist_configs(self, attack_name, df_train, df_test, y_train, y_test):
        """Run S1-S4 configurations for a single attack type."""
        results = {}
        
        # Get features
        exclude_cols = ['label', 'attack', 'attack_cat', 'stime', 'srcip', 'dstip', 'id']
        feature_cols = [col for col in df_train.columns if col not in exclude_cols]
        
        X_train_raw = df_train[feature_cols].values
        X_test_raw = df_test[feature_cols].values
        
        # S1: Selection + SMOTE (CACHED)
        print(f"\n   📦 S1: Selection + SMOTE (checking cache)")
        s1_result = self._load_cached_specialist(attack_name)
        if s1_result:
            results['S1'] = s1_result
            print(f"      ✅ Loaded from cache: F1={s1_result['test_f1']:.3f}")
        else:
            print(f"      ⚠️ No cached specialist found")
            results['S1'] = self._train_specialist_config(
                'S1', X_train_raw, X_test_raw, y_train, y_test, feature_cols,
                use_selection=True, use_smote=True
            )
        
        # S2: Selection + No SMOTE (FRESH)
        print(f"\n   🔧 S2: Selection + No SMOTE (training fresh)")
        results['S2'] = self._train_specialist_config(
            'S2', X_train_raw, X_test_raw, y_train, y_test, feature_cols,
            use_selection=True, use_smote=False
        )
        
        # S3: No Selection + SMOTE (FRESH)
        print(f"\n   🔧 S3: No Selection + SMOTE (training fresh)")
        results['S3'] = self._train_specialist_config(
            'S3', X_train_raw, X_test_raw, y_train, y_test, feature_cols,
            use_selection=False, use_smote=True
        )
        
        # S4: No Selection + No SMOTE (FRESH)
        print(f"\n   🔧 S4: No Selection + No SMOTE (training fresh)")
        results['S4'] = self._train_specialist_config(
            'S4', X_train_raw, X_test_raw, y_train, y_test, feature_cols,
            use_selection=False, use_smote=False
        )
        
        return results
    
    def _load_cached_specialist(self, attack_name):
        """Load cached specialist model."""
        # Check both SMOTE folders
        for folder in ['Specialists_with_smote', 'Specialists']:
            path = os.path.join(self.cache_dir, 'Binary', folder, f'{attack_name}_specialist.pkl')
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    spec = pickle.load(f)
                
                metrics = spec.get('metrics', {})
                return {
                    'config': 'S1',
                    'test_accuracy': metrics.get('test_accuracy', metrics.get('test_balanced_acc', 0)),
                    'test_precision': metrics.get('test_precision', 0),
                    'test_recall': metrics.get('test_recall', 0),
                    'test_f1': metrics.get('test_f1', 0),
                    'test_auc': metrics.get('test_auc', 0),
                    'test_balanced_acc': metrics.get('test_balanced_acc', 0),
                    'cv_f1_mean': metrics.get('cv_f1_mean', 0),
                    'cv_f1_std': metrics.get('cv_f1_std', 0),
                    'source': 'cached'
                }
        
        return None
    
    def _train_specialist_config(self, config_name, X_train_raw, X_test_raw, 
                                 y_train, y_test, feature_names,
                                 use_selection=True, use_smote=True):
        """Train a specialist configuration."""
        from sklearn.model_selection import train_test_split
        
        # Feature engineering
        fe = DynamicFeatureEngineer()
        df_train = pd.DataFrame(X_train_raw, columns=feature_names)
        df_test = pd.DataFrame(X_test_raw, columns=feature_names)
        
        X_train_eng = fe.fit_transform(df_train)
        X_test_eng = fe.transform(df_test)
        
        if isinstance(X_train_eng, pd.DataFrame):
            X_train_eng = X_train_eng.values
            X_test_eng = X_test_eng.values
        
        # Scale
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train_eng)
        X_test_scaled = scaler.transform(X_test_eng)
        
        # Feature selection
        if use_selection:
            selector = ConsensusFeatureSelector()
            X_train_selected, _ = selector.fit_transform(X_train_scaled, y_train)
            X_test_selected = selector.transform(X_test_scaled)
        else:
            X_train_selected = X_train_scaled
            X_test_selected = X_test_scaled
        
        # SMOTE
        if use_smote:
            imbalance_ratio = np.sum(y_train == 0) / (np.sum(y_train == 1) + 1e-10)
            if imbalance_ratio > 20:
                try:
                    from imblearn.over_sampling import SMOTE
                    k_neighbors = min(5, np.sum(y_train == 1) - 1)
                    if k_neighbors >= 1:
                        smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
                        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_selected, y_train)
                    else:
                        X_train_resampled, y_train_resampled = X_train_selected, y_train
                except Exception:
                    X_train_resampled, y_train_resampled = X_train_selected, y_train
            else:
                X_train_resampled, y_train_resampled = X_train_selected, y_train
        else:
            X_train_resampled, y_train_resampled = X_train_selected, y_train
        
        # Train GB
        gb = GradientBoostingClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42
        )
        
        # CV on original (non-SMOTE) data
        cv_f1 = cross_val_score(gb, X_train_selected, y_train, cv=self.cv, scoring='f1')
        
        # Train and evaluate
        gb.fit(X_train_resampled, y_train_resampled)
        y_pred = gb.predict(X_test_selected)
        y_proba = gb.predict_proba(X_test_selected)
        
        # Binary classification metrics (specialists are always binary)
        test_accuracy = accuracy_score(y_test, y_pred)
        test_precision = precision_score(y_test, y_pred, zero_division=0)
        test_recall = recall_score(y_test, y_pred, zero_division=0)
        test_f1 = f1_score(y_test, y_pred, zero_division=0)
        test_auc = roc_auc_score(y_test, y_proba[:, 1])
        test_balanced_acc = balanced_accuracy_score(y_test, y_pred)
        
        print(f"      CV F1: {cv_f1.mean():.3f} ± {cv_f1.std():.3f}")
        print(f"      Test Metrics:")
        print(f"         Accuracy: {test_accuracy:.3f}")
        print(f"         Precision: {test_precision:.3f}")
        print(f"         Recall: {test_recall:.3f}")
        print(f"         F1-Score: {test_f1:.3f}")
        print(f"         AUC-ROC: {test_auc:.3f}")
        print(f"         Balanced Acc: {test_balanced_acc:.3f}")
        
        return {
            'config': config_name,
            'test_accuracy': test_accuracy,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1': test_f1,
            'test_auc': test_auc,
            'test_balanced_acc': test_balanced_acc,
            'cv_f1': cv_f1,
            'cv_f1_mean': cv_f1.mean(),
            'cv_f1_std': cv_f1.std(),
            'source': 'fresh'
        }

    
    # =========================================================================
    # STATISTICAL COMPARISONS
    # =========================================================================
    
    def run_statistical_comparisons(self):
        """
        Run all statistical comparisons with enhanced testing.
        
        Includes:
        1. Shapiro-Wilk normality test
        2. Adaptive test selection (t-test vs Wilcoxon)
        3. Cohen's d effect size
        4. Holm-Bonferroni multiple comparison correction
        """
        print("\n" + "="*80)
        print("📊 ENHANCED STATISTICAL COMPARISONS")
        print("="*80)
        
        comparisons = []
        
        # Feature engineering comparisons (each group vs baseline)
        comparison_pairs = [
            ('B2a', 'B1a', "Statistical features (no sel)"),
            ('B2b', 'B1b', "Statistical features (with sel)"),
            ('B3a', 'B1a', "Duration features (no sel)"),
            ('B3b', 'B1b', "Duration features (with sel)"),
            ('B4a', 'B1a', "Network features (no sel)"),
            ('B4b', 'B1b', "Network features (with sel)"),
            ('B5a', 'B1a', "Interaction features (no sel)"),
            ('B5b', 'B1b', "Interaction features (with sel)"),
            ('B6a', 'B1a', "All 23 features (no sel)"),
            ('B6b', 'B1b', "All 23 features (with sel)"),
            ('B6b', 'B6a', "Selection effect on full FE"),
        ]
        
        # Ensemble comparisons
        if 'E4' in self.cv_scores and 'B6b' in self.cv_scores:
            comparison_pairs.append(('E4', 'B6b', "Ensemble effect"))
        
        if 'E2' in self.cv_scores and 'E1' in self.cv_scores:
            comparison_pairs.append(('E2', 'E1', "Mixing ratio (no sel)"))
        
        if 'E4' in self.cv_scores and 'E3' in self.cv_scores:
            comparison_pairs.append(('E4', 'E3', "Mixing ratio (with sel)"))
        
        # Run individual comparisons
        p_values = []
        for config_a, config_b, description in comparison_pairs:
            if config_a in self.cv_scores and config_b in self.cv_scores:
                result = self.compare_configs(
                    self.cv_scores[config_a], 
                    self.cv_scores[config_b],
                    config_a, config_b
                )
                result['description'] = description
                comparisons.append(result)
                p_values.append(result['p_value'])
        
        # Apply Holm-Bonferroni correction for multiple comparisons
        if p_values:
            try:
                from statsmodels.stats.multitest import multipletests
                rejected, p_corrected, _, _ = multipletests(p_values, alpha=0.05, method='holm')
                
                for i, result in enumerate(comparisons):
                    result['p_value_corrected'] = p_corrected[i]
                    result['significant_corrected'] = rejected[i]
                
                print(f"\n📊 Multiple Comparison Correction: Holm-Bonferroni")
                print(f"   Number of comparisons: {len(p_values)}")
            except ImportError:
                print(f"\n⚠️  statsmodels not available - skipping multiple comparison correction")
                for result in comparisons:
                    result['p_value_corrected'] = result['p_value']
                    result['significant_corrected'] = result['significant_raw']
        
        # Print results table
        print(f"\n{'Comparison':<35} {'Test':<12} {'Δ Acc':>8} {'p-raw':>8} {'p-corr':>8} {'Cohen d':>12} {'Sig':>5}")
        print("-" * 100)
        
        for result in comparisons:
            sig_marker = "✅" if result.get('significant_corrected', result['significant_raw']) else "❌"
            effect = f"{result['cohens_d']:.2f} ({result['effect_interpretation'][:3]})"
            test_short = "t-test" if "t-test" in result['test_used'] else "Wilcox"
            p_corr = result.get('p_value_corrected', result['p_value'])
            
            print(f"{result['description']:<35} {test_short:<12} {result['diff']:>+8.4f} {result['p_value']:>8.4f} {p_corr:>8.4f} {effect:>12} {sig_marker:>5}")
        
        # Summary statistics
        n_significant_raw = sum(1 for r in comparisons if r['significant_raw'])
        n_significant_corr = sum(1 for r in comparisons if r.get('significant_corrected', r['significant_raw']))
        
        print(f"\n📊 Summary:")
        print(f"   Total comparisons: {len(comparisons)}")
        print(f"   Significant (raw p<0.05): {n_significant_raw}")
        print(f"   Significant (corrected): {n_significant_corr}")
        
        self.results['comparisons'] = comparisons
        return comparisons
    
    def compute_bootstrap_ci(self, y_true, y_pred, metric_func, n_bootstrap=1000, confidence=0.95):
        """
        Compute bootstrap confidence intervals for test set metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            metric_func: Metric function (e.g., accuracy_score)
            n_bootstrap: Number of bootstrap samples
            confidence: Confidence level (default 95%)
        
        Returns:
            dict with observed value, CI bounds, and std
        """
        n = len(y_true)
        scores = []
        
        np.random.seed(42)
        for _ in range(n_bootstrap):
            idx = np.random.choice(n, size=n, replace=True)
            scores.append(metric_func(y_true[idx], y_pred[idx]))
        
        scores = np.array(scores)
        alpha = 1 - confidence
        ci_lower = np.percentile(scores, alpha/2 * 100)
        ci_upper = np.percentile(scores, (1 - alpha/2) * 100)
        
        return {
            'observed': metric_func(y_true, y_pred),
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'std': np.std(scores)
        }
    
    # =========================================================================
    # SUMMARY AND REPORTING
    # =========================================================================
    
    def generate_summary(self):
        """Generate comprehensive summary of ablation study."""
        print("\n" + "="*80)
        print("📋 ABLATION STUDY SUMMARY")
        print("="*80)
        
        # Baseline results table
        print("\n📊 BASELINE CONFIGURATIONS (GB, Fresh Training)")
        print("-" * 80)
        print(f"{'Config':<8} {'Features':<15} {'Selection':<10} {'CV Acc':>14} {'Test Acc':>10} {'Test F1':>10}")
        print("-" * 80)
        
        baseline_configs = ['B1a', 'B1b', 'B2a', 'B2b', 'B3a', 'B3b', 'B4a', 'B4b', 'B5a', 'B5b', 'B6a', 'B6b']
        for config in baseline_configs:
            if config in self.results:
                r = self.results[config]
                selection = "Yes" if config.endswith('b') else "No"
                
                # Determine feature group
                if config.startswith('B1'):
                    features = "None"
                elif config.startswith('B2'):
                    features = "Statistical(7)"
                elif config.startswith('B3'):
                    features = "Duration(5)"
                elif config.startswith('B4'):
                    features = "Network(9)"
                elif config.startswith('B5'):
                    features = "Interaction(2)"
                elif config.startswith('B6'):
                    features = "All 23"
                else:
                    features = "Unknown"
                
                cv_acc = f"{r['cv_accuracy_mean']:.4f}±{r['cv_accuracy_std']:.4f}"
                print(f"{config:<8} {features:<15} {selection:<10} {cv_acc:>14} {r['test_accuracy']:>10.4f} {r['test_f1']:>10.4f}")
        
        # Ensemble results table
        print("\n📊 ENSEMBLE CONFIGURATIONS (Cached Models)")
        print("-" * 80)
        print(f"{'Config':<8} {'Selection':<10} {'Mixing':>12} {'CV Acc':>14} {'Test Acc':>10} {'Test F1':>10}")
        print("-" * 80)
        
        ensemble_configs = ['E1', 'E2', 'E3', 'E4']
        for config in ensemble_configs:
            if config in self.results:
                r = self.results[config]
                selection = "Yes" if config in ['E3', 'E4'] else "No"
                mixing = "Optimized" if config in ['E2', 'E4'] else "α=1.0"
                cv_acc = f"{r['cv_accuracy_mean']:.4f}±{r['cv_accuracy_std']:.4f}"
                print(f"{config:<8} {selection:<10} {mixing:>12} {cv_acc:>14} {r['test_accuracy']:>10.4f} {r['test_f1']:>10.4f}")
        
        # Key findings
        print("\n📊 KEY FINDINGS")
        print("-" * 80)
        
        if 'B6b' in self.results and 'B1b' in self.results:
            improvement = self.results['B6b']['test_accuracy'] - self.results['B1b']['test_accuracy']
            print(f"   Full FE pipeline improvement over baseline: {improvement:+.4f}")
        
        if 'E4' in self.results and 'B6b' in self.results:
            improvement = self.results['E4']['test_accuracy'] - self.results['B6b']['test_accuracy']
            print(f"   Ensemble improvement over single GB: {improvement:+.4f}")
        
        return self.results
    
    # =========================================================================
    # MAIN RUN METHOD
    # =========================================================================
    
    def run_full_study(self):
        """Run the complete ablation study."""
        print("\n" + "="*80)
        print("🔬 COMPLETE ABLATION STUDY")
        print(f"   Classification: {self.classification_type.upper()}")
        print(f"   Train data: {self.train_csv}")
        print(f"   Test data: {self.test_csv or 'Split from train'}")
        print("="*80)
        
        # Run all configurations
        self.run_all_baselines()
        self.run_all_ensembles()
        
        # Statistical comparisons
        self.run_statistical_comparisons()
        
        # Generate summary
        self.generate_summary()
        
        # Save results
        self.save_results()
        
        return self.results
    
    def save_results(self, filepath='ablation_results.pkl'):
        """Save ablation study results."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'results': self.results,
                'cv_scores': {k: v.tolist() if hasattr(v, 'tolist') else v 
                             for k, v in self.cv_scores.items()},
                'classification_type': self.classification_type
            }, f)
        print(f"\n💾 Results saved to: {filepath}")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main entry point for ablation study."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Ablation Study for Novel Ensemble ML')
    parser.add_argument('--train', type=str, default='UNSW_balanced_train.csv',
                       help='Training dataset CSV')
    parser.add_argument('--test', type=str, default=None,
                       help='Test dataset CSV (optional)')
    parser.add_argument('--mode', type=str, default='binary', choices=['binary', 'multiclass'],
                       help='Classification mode')
    parser.add_argument('--cache-dir', type=str, default='Models',
                       help='Cache directory for models')
    parser.add_argument('--baselines-only', action='store_true',
                       help='Run only baseline configurations')
    parser.add_argument('--ensembles-only', action='store_true',
                       help='Run only ensemble configurations')
    parser.add_argument('--specialists', action='store_true',
                       help='Include specialist ablations')
    
    args = parser.parse_args()
    
    # Check if training data exists
    if not os.path.exists(args.train):
        print(f"❌ Training data not found: {args.train}")
        print("   Run: python create_balanced_split.py first")
        return
    
    # Create ablation study
    study = AblationStudy(
        train_csv=args.train,
        test_csv=args.test,
        classification_type=args.mode,
        cache_dir=args.cache_dir
    )
    
    # Run appropriate configurations
    if args.baselines_only:
        study.run_all_baselines()
        study.run_statistical_comparisons()
        study.generate_summary()
    elif args.ensembles_only:
        study.run_all_ensembles()
        study.generate_summary()
    else:
        study.run_full_study()
    
    # Run specialist ablations if requested
    if args.specialists:
        study.run_specialist_ablations()
    
    print("\n✅ Ablation study complete!")


if __name__ == "__main__":
    main()
