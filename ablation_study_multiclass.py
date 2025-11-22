#!/usr/bin/env python3
"""
Ablation Study for Multiclass Novel Ensemble ML System
Evaluates the contribution of each component in the architecture

Methodology:
- Random Forest classifier as baseline
- 80-20 train-test split
- Accuracy as primary metric
- Sequential cumulative stages
- Multiclass classification (10 attack types)
"""

import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
import os
from novel_ensemble_ml import DynamicFeatureEngineer, IntelligentFeatureSelector

class MulticlassAblationStudy:
    """Ablation study for multiclass classification"""
    
    def __init__(self, train_csv, test_csv=None):
        self.train_csv = train_csv
        self.test_csv = test_csv
        self.results = []
        self.baseline_accuracy = None
        
    def load_and_preprocess_data(self):
        """Load and preprocess UNSW-NB15 data for multiclass"""
        print("📁 Loading UNSW-NB15 dataset...")
        df_train = pd.read_csv(self.train_csv)
        
        if self.test_csv:
            df_test = pd.read_csv(self.test_csv)
            df = pd.concat([df_train, df_test], ignore_index=True)
        else:
            df = df_train
        
        print(f"✅ Loaded {len(df):,} samples")
        
        # Encode target (attack_cat for multiclass)
        if 'attack_cat' not in df.columns:
            raise ValueError("attack_cat column not found - required for multiclass")
        
        self.label_encoder = LabelEncoder()
        df['target'] = self.label_encoder.fit_transform(df['attack_cat'])
        
        print(f"📊 Classes: {len(self.label_encoder.classes_)}")
        for i, cls in enumerate(self.label_encoder.classes_):
            count = np.sum(df['target'] == i)
            print(f"   {cls}: {count:,} samples")
        
        # Separate features and target
        exclude_cols = ['id', 'attack_cat', 'label', 'target']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].copy()
        y = df['target'].values
        
        # Encode categorical features
        categorical_cols = ['proto', 'service', 'state']
        self.label_encoders = {}
        
        for col in categorical_cols:
            if col in X.columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.label_encoders[col] = le
        
        # Handle missing values
        X = X.fillna(0)
        
        # Store original feature names
        self.original_features = X.columns.tolist()
        print(f"📊 Original features: {len(self.original_features)}")
        
        return X, y
    
    def run_ablation_study(self):
        """Run complete ablation study"""
        print("\n" + "="*80)
        print("🔬 MULTICLASS ABLATION STUDY")
        print("="*80)
        
        # Load data
        X, y = self.load_and_preprocess_data()
        
        # 80-20 split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\n📊 Data Split:")
        print(f"   Training: {len(X_train):,} samples")
        print(f"   Testing:  {len(X_test):,} samples")
        
        # Stage 1: Baseline (original features only)
        print("\n" + "="*80)
        print("STAGE 1: Baseline (Original Features)")
        print("="*80)
        accuracy_stage1 = self._evaluate_stage(
            X_train, X_test, y_train, y_test,
            stage_name="Stage 1: Baseline",
            description="Original 42 UNSW-NB15 features"
        )
        self.baseline_accuracy = accuracy_stage1
        
        # Stage 2: + Temporal Features (5 features)
        print("\n" + "="*80)
        print("STAGE 2: + Temporal Features")
        print("="*80)
        X_train_s2, X_test_s2 = self._add_temporal_features(X_train, X_test)
        accuracy_stage2 = self._evaluate_stage(
            X_train_s2, X_test_s2, y_train, y_test,
            stage_name="Stage 2: + Temporal",
            description="Original + 5 temporal features"
        )
        
        # Stage 3: + Statistical Features (16 features)
        print("\n" + "="*80)
        print("STAGE 3: + Statistical Features")
        print("="*80)
        X_train_s3, X_test_s3 = self._add_statistical_features(X_train, X_test, y_train)
        accuracy_stage3 = self._evaluate_stage(
            X_train_s3, X_test_s3, y_train, y_test,
            stage_name="Stage 3: + Statistical",
            description="Original + 16 statistical features"
        )
        
        # Stage 4: + Interaction Features (2 features)
        print("\n" + "="*80)
        print("STAGE 4: + Interaction Features")
        print("="*80)
        X_train_s4, X_test_s4 = self._add_interaction_features(X_train_s3, X_test_s3)
        accuracy_stage4 = self._evaluate_stage(
            X_train_s4, X_test_s4, y_train, y_test,
            stage_name="Stage 4: + Interactions",
            description="Previous + 2 interaction features"
        )
        
        # Stage 5: + Network Behavior Features (9 features)
        print("\n" + "="*80)
        print("STAGE 5: + Network Behavior Features")
        print("="*80)
        X_train_s5, X_test_s5 = self._add_network_behavior_features(X_train_s4, X_test_s4)
        accuracy_stage5 = self._evaluate_stage(
            X_train_s5, X_test_s5, y_train, y_test,
            stage_name="Stage 5: + Network Behavior",
            description="Previous + 9 network behavior features"
        )
        
        # Stage 6: Full Feature Engineering (all features combined)
        print("\n" + "="*80)
        print("STAGE 6: Full Feature Engineering")
        print("="*80)
        X_train_s6, X_test_s6 = self._apply_full_feature_engineering(X_train, X_test, y_train)
        accuracy_stage6 = self._evaluate_stage(
            X_train_s6, X_test_s6, y_train, y_test,
            stage_name="Stage 6: Full Engineering",
            description="All engineered features combined"
        )
        
        # Stage 7: + Feature Selection
        print("\n" + "="*80)
        print("STAGE 7: + Feature Selection")
        print("="*80)
        X_train_s7, X_test_s7 = self._apply_feature_selection(X_train_s6, X_test_s6, y_train)
        accuracy_stage7 = self._evaluate_stage(
            X_train_s7, X_test_s7, y_train, y_test,
            stage_name="Stage 7: + Feature Selection",
            description="Selected features from full set"
        )
        
        # Stage 8: Complete Architecture (Adaptive Ensemble)
        print("\n" + "="*80)
        print("STAGE 8: Complete Architecture")
        print("="*80)
        accuracy_stage8 = self._evaluate_complete_architecture(
            X_train_s7, X_test_s7, y_train, y_test
        )
        
        # Generate summary
        self._generate_summary()
        
        return self.results
    
    def _evaluate_stage(self, X_train, X_test, y_train, y_test, stage_name, description):
        """Evaluate a single stage using Random Forest"""
        print(f"\n📊 {stage_name}")
        print(f"   Description: {description}")
        print(f"   Features: {X_train.shape[1]}")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Random Forest
        rf = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        print("   🔄 Training Random Forest...")
        rf.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = rf.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        
        # Calculate improvement
        if self.baseline_accuracy is not None:
            improvement = accuracy - self.baseline_accuracy
            improvement_pct = (improvement / self.baseline_accuracy) * 100
        else:
            improvement = 0.0
            improvement_pct = 0.0
        
        print(f"   ✅ Accuracy: {accuracy:.4f}")
        print(f"   📊 F1-Macro: {f1_macro:.4f}")
        print(f"   📊 F1-Weighted: {f1_weighted:.4f}")
        
        if self.baseline_accuracy is not None:
            print(f"   📈 vs Baseline: {improvement:+.4f} ({improvement_pct:+.2f}%)")
        
        # Store results
        self.results.append({
            'stage': stage_name,
            'description': description,
            'n_features': X_train.shape[1],
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'improvement': improvement,
            'improvement_pct': improvement_pct
        })
        
        return accuracy
    
    def _add_temporal_features(self, X_train, X_test):
        """Add 5 temporal features"""
        print("   Adding temporal features...")
        
        X_train_new = X_train.copy()
        X_test_new = X_test.copy()
        
        # Temporal features from stime (if available)
        if 'stime' in X_train.columns:
            # Hour of day
            X_train_new['hour'] = (X_train['stime'] % 86400) // 3600
            X_test_new['hour'] = (X_test['stime'] % 86400) // 3600
            
            # Day of week
            X_train_new['day_of_week'] = (X_train['stime'] // 86400) % 7
            X_test_new['day_of_week'] = (X_test['stime'] // 86400) % 7
            
            # Is weekend
            X_train_new['is_weekend'] = (X_train_new['day_of_week'] >= 5).astype(int)
            X_test_new['is_weekend'] = (X_test_new['day_of_week'] >= 5).astype(int)
            
            # Is business hours (9-17)
            X_train_new['is_business_hours'] = ((X_train_new['hour'] >= 9) & (X_train_new['hour'] < 17)).astype(int)
            X_test_new['is_business_hours'] = ((X_test_new['hour'] >= 9) & (X_test_new['hour'] < 17)).astype(int)
            
            # Is night (22-6)
            X_train_new['is_night'] = ((X_train_new['hour'] >= 22) | (X_train_new['hour'] < 6)).astype(int)
            X_test_new['is_night'] = ((X_test_new['hour'] >= 22) | (X_test_new['hour'] < 6)).astype(int)
        
        print(f"   ✅ Added {X_train_new.shape[1] - X_train.shape[1]} temporal features")
        
        return X_train_new, X_test_new
    
    def _add_statistical_features(self, X_train, X_test, y_train):
        """Add 16 statistical features (matching binary ablation)"""
        print("   Adding statistical features...")
        
        X_train_new = X_train.copy()
        X_test_new = X_test.copy()
        
        # Total bytes and packets
        if 'sbytes' in X_train.columns and 'dbytes' in X_train.columns:
            X_train_new['total_bytes'] = X_train['sbytes'] + X_train['dbytes']
            X_test_new['total_bytes'] = X_test['sbytes'] + X_test['dbytes']
            
            # Log total bytes
            X_train_new['log_total_bytes'] = np.log1p(X_train_new['total_bytes'])
            X_test_new['log_total_bytes'] = np.log1p(X_test_new['total_bytes'])
            
            # Byte ratio and imbalance
            X_train_new['byte_ratio'] = X_train['sbytes'] / (X_train['dbytes'] + 1)
            X_test_new['byte_ratio'] = X_test['sbytes'] / (X_test['dbytes'] + 1)
            
            X_train_new['byte_imbalance'] = np.abs(X_train['sbytes'] - X_train['dbytes'])
            X_test_new['byte_imbalance'] = np.abs(X_test['sbytes'] - X_test['dbytes'])
        
        if 'spkts' in X_train.columns and 'dpkts' in X_train.columns:
            X_train_new['total_packets'] = X_train['spkts'] + X_train['dpkts']
            X_test_new['total_packets'] = X_test['spkts'] + X_test['dpkts']
            
            # Packet ratio
            X_train_new['packet_ratio'] = X_train['spkts'] / (X_train['dpkts'] + 1)
            X_test_new['packet_ratio'] = X_test['spkts'] / (X_test['dpkts'] + 1)
            
            # Average packet size
            if 'total_bytes' in X_train_new.columns:
                X_train_new['avg_packet_size'] = X_train_new['total_bytes'] / (X_train_new['total_packets'] + 1)
                X_test_new['avg_packet_size'] = X_test_new['total_bytes'] / (X_test_new['total_packets'] + 1)
        
        # Duration features
        if 'dur' in X_train.columns:
            X_train_new['log_duration'] = np.log1p(X_train['dur'])
            X_test_new['log_duration'] = np.log1p(X_test['dur'])
            
            # Short/long connection indicators
            dur_25 = np.percentile(X_train['dur'], 25)
            dur_75 = np.percentile(X_train['dur'], 75)
            X_train_new['is_short_connection'] = (X_train['dur'] < dur_25).astype(int)
            X_train_new['is_long_connection'] = (X_train['dur'] > dur_75).astype(int)
            X_test_new['is_short_connection'] = (X_test['dur'] < dur_25).astype(int)
            X_test_new['is_long_connection'] = (X_test['dur'] > dur_75).astype(int)
        
        # Throughput
        if 'sbytes' in X_train.columns and 'dur' in X_train.columns:
            throughput_train = X_train['sbytes'] / (X_train['dur'] + 1)
            throughput_test = X_test['sbytes'] / (X_test['dur'] + 1)
            X_train_new['throughput'] = throughput_train
            X_test_new['throughput'] = throughput_test
            X_train_new['log_throughput'] = np.log1p(throughput_train)
            X_test_new['log_throughput'] = np.log1p(throughput_test)
        
        # Port features
        if 'sport' in X_train.columns:
            X_train_new['src_is_wellknown'] = (X_train['sport'] < 1024).astype(int)
            X_test_new['src_is_wellknown'] = (X_test['sport'] < 1024).astype(int)
        
        if 'dport' in X_train.columns:
            X_train_new['dst_is_wellknown'] = (X_train['dport'] < 1024).astype(int)
            X_test_new['dst_is_wellknown'] = (X_test['dport'] < 1024).astype(int)
            
            # Common service ports
            common_ports = [80, 443, 22, 21, 25, 53, 3389]
            X_train_new['dst_is_common_service'] = X_train['dport'].isin(common_ports).astype(int)
            X_test_new['dst_is_common_service'] = X_test['dport'].isin(common_ports).astype(int)
        
        if 'sport' in X_train.columns and 'dport' in X_train.columns:
            X_train_new['port_difference'] = np.abs(X_train['sport'] - X_train['dport'])
            X_test_new['port_difference'] = np.abs(X_test['sport'] - X_test['dport'])
        
        print(f"   ✅ Added {X_train_new.shape[1] - X_train.shape[1]} statistical features")
        
        return X_train_new, X_test_new
    
    def _add_interaction_features(self, X_train, X_test):
        """Add 2 interaction features"""
        print("   Adding interaction features...")
        
        X_train_new = X_train.copy()
        X_test_new = X_test.copy()
        
        # Protocol-Service interaction
        if 'proto' in X_train.columns and 'service' in X_train.columns:
            X_train_new['proto_service'] = X_train['proto'].astype(str) + '_' + X_train['service'].astype(str)
            X_test_new['proto_service'] = X_test['proto'].astype(str) + '_' + X_test['service'].astype(str)
            
            # Encode with handling for unseen categories
            le = LabelEncoder()
            X_train_new['proto_service'] = le.fit_transform(X_train_new['proto_service'])
            
            # Handle unseen categories in test set
            test_categories = X_test_new['proto_service'].values
            encoded_test = []
            for cat in test_categories:
                if cat in le.classes_:
                    encoded_test.append(le.transform([cat])[0])
                else:
                    # Assign to a default "unknown" category (use -1)
                    encoded_test.append(-1)
            X_test_new['proto_service'] = encoded_test
        
        # State-Protocol interaction
        if 'state' in X_train.columns and 'proto' in X_train.columns:
            X_train_new['state_proto'] = X_train['state'].astype(str) + '_' + X_train['proto'].astype(str)
            X_test_new['state_proto'] = X_test['state'].astype(str) + '_' + X_test['proto'].astype(str)
            
            # Encode with handling for unseen categories
            le = LabelEncoder()
            X_train_new['state_proto'] = le.fit_transform(X_train_new['state_proto'])
            
            # Handle unseen categories in test set
            test_categories = X_test_new['state_proto'].values
            encoded_test = []
            for cat in test_categories:
                if cat in le.classes_:
                    encoded_test.append(le.transform([cat])[0])
                else:
                    # Assign to a default "unknown" category (use -1)
                    encoded_test.append(-1)
            X_test_new['state_proto'] = encoded_test
        
        print(f"   ✅ Added {X_train_new.shape[1] - X_train.shape[1]} interaction features")
        
        return X_train_new, X_test_new
    
    def _add_network_behavior_features(self, X_train, X_test):
        """Add 9 network behavior features"""
        print("   Adding network behavior features...")
        
        X_train_new = X_train.copy()
        X_test_new = X_test.copy()
        
        # Packet loss indicators
        if 'sloss' in X_train.columns:
            X_train_new['has_src_loss'] = (X_train['sloss'] > 0).astype(int)
            X_test_new['has_src_loss'] = (X_test['sloss'] > 0).astype(int)
        
        if 'dloss' in X_train.columns:
            X_train_new['has_dst_loss'] = (X_train['dloss'] > 0).astype(int)
            X_test_new['has_dst_loss'] = (X_test['dloss'] > 0).astype(int)
        
        # Jitter indicators
        if 'sjit' in X_train.columns:
            jit_threshold = np.percentile(X_train['sjit'], 75)
            X_train_new['high_jitter'] = (X_train['sjit'] > jit_threshold).astype(int)
            X_test_new['high_jitter'] = (X_test['sjit'] > jit_threshold).astype(int)
        
        # TCP window features
        if 'swin' in X_train.columns:
            X_train_new['log_swin'] = np.log1p(X_train['swin'])
            X_test_new['log_swin'] = np.log1p(X_test['swin'])
        
        if 'dwin' in X_train.columns:
            X_train_new['log_dwin'] = np.log1p(X_train['dwin'])
            X_test_new['log_dwin'] = np.log1p(X_test['dwin'])
        
        # Window ratio
        if 'swin' in X_train.columns and 'dwin' in X_train.columns:
            X_train_new['window_ratio'] = X_train['swin'] / (X_train['dwin'] + 1)
            X_test_new['window_ratio'] = X_test['swin'] / (X_test['dwin'] + 1)
        
        # TCP base sequence features
        if 'stcpb' in X_train.columns:
            X_train_new['log_stcpb'] = np.log1p(X_train['stcpb'])
            X_test_new['log_stcpb'] = np.log1p(X_test['stcpb'])
        
        if 'dtcpb' in X_train.columns:
            X_train_new['log_dtcpb'] = np.log1p(X_train['dtcpb'])
            X_test_new['log_dtcpb'] = np.log1p(X_test['dtcpb'])
        
        # Retransmission indicator
        if 'ct_state_ttl' in X_train.columns:
            X_train_new['has_retrans'] = (X_train['ct_state_ttl'] > 1).astype(int)
            X_test_new['has_retrans'] = (X_test['ct_state_ttl'] > 1).astype(int)
        
        print(f"   ✅ Added {X_train_new.shape[1] - X_train.shape[1]} network behavior features")
        
        return X_train_new, X_test_new
    
    def _apply_full_feature_engineering(self, X_train, X_test, y_train):
        """Apply full feature engineering pipeline"""
        print("   Applying full feature engineering...")
        
        # Use DynamicFeatureEngineer
        fe = DynamicFeatureEngineer()
        
        # Fit on training data (no y needed)
        X_train_eng = fe.fit_transform(X_train)
        X_test_eng = fe.transform(X_test)
        
        print(f"   ✅ Engineered features: {X_train_eng.shape[1]} total")
        
        return X_train_eng, X_test_eng
    
    def _apply_feature_selection(self, X_train, X_test, y_train):
        """Apply intelligent feature selection"""
        print("   Applying feature selection...")
        
        # Use IntelligentFeatureSelector
        selector = IntelligentFeatureSelector(
            n_features=None  # Auto-select optimal count
        )
        
        # Get feature names
        if hasattr(X_train, 'columns'):
            feature_names = X_train.columns.tolist()
            X_train_values = X_train.values
            X_test_values = X_test.values
        else:
            feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]
            X_train_values = X_train
            X_test_values = X_test
        
        # Fit on training data - returns (X_selected, selected_indices)
        X_train_selected, selected_indices = selector.fit_transform(X_train_values, y_train, feature_names)
        
        # Apply same selection to test data
        X_test_selected = X_test_values[:, selected_indices]
        
        print(f"   ✅ Selected features: {X_train_selected.shape[1]}")
        
        return X_train_selected, X_test_selected
    
    def _evaluate_complete_architecture(self, X_train, X_test, y_train, y_test):
        """Evaluate complete architecture with all 10 classifiers"""
        print("\n📊 Stage 7: Complete Architecture")
        print("   Description: Full pipeline with 10 classifiers")
        print(f"   Features: {X_train.shape[1]}")
        
        from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier
        from sklearn.linear_model import SGDClassifier, LogisticRegression
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.naive_bayes import GaussianNB
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.svm import SVC
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Define all 10 classifiers
        classifiers = {
            'RF': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced'),
            'GB': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'ET': ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced'),
            'SGD': SGDClassifier(loss='log_loss', random_state=42, max_iter=1000, class_weight='balanced'),
            'KNN': KNeighborsClassifier(n_neighbors=7),
            'NB': GaussianNB(),
            'LR': LogisticRegression(random_state=42, max_iter=2000, C=1.0, class_weight='balanced'),
            'DT': DecisionTreeClassifier(random_state=42, max_depth=10, class_weight='balanced'),
            'SVM': CalibratedClassifierCV(LinearSVC(C=1.0, random_state=42, max_iter=2000, dual=False), cv=5, method='sigmoid')
        }
        
        # Add XGBoost if available
        try:
            import xgboost as xgb
            classifiers['XGB'] = xgb.XGBClassifier(
                random_state=42,
                n_estimators=100,
                objective='multi:softprob',
                eval_metric='mlogloss'
            )
        except ImportError:
            print("   ⚠️  XGBoost not available")
        
        print(f"   🔄 Training {len(classifiers)} classifiers...")
        
        # Train all classifiers and collect predictions
        accuracies = {}
        for name, clf in classifiers.items():
            clf.fit(X_train_scaled, y_train)
            
            # Use predict_proba + argmax for consistent predictions
            if hasattr(clf, 'predict_proba'):
                y_proba = clf.predict_proba(X_test_scaled)
                y_pred = np.argmax(y_proba, axis=1)
            else:
                y_pred = clf.predict(X_test_scaled)
            
            acc = accuracy_score(y_test, y_pred)
            accuracies[name] = acc
            print(f"      {name}: {acc:.4f}")
        
        # Best individual
        best_name = max(accuracies, key=accuracies.get)
        best_accuracy = accuracies[best_name]
        
        print(f"\n   🏆 Best Classifier: {best_name} ({best_accuracy:.4f})")
        
        # Calculate improvement
        improvement = best_accuracy - self.baseline_accuracy
        improvement_pct = (improvement / self.baseline_accuracy) * 100
        
        print(f"   📈 vs Baseline: {improvement:+.4f} ({improvement_pct:+.2f}%)")
        
        # Store results
        self.results.append({
            'stage': "Stage 7: Complete Architecture",
            'description': f"Full pipeline with {len(classifiers)} classifiers",
            'n_features': X_train.shape[1],
            'accuracy': best_accuracy,
            'f1_macro': 0.0,  # Not calculated for ensemble
            'f1_weighted': 0.0,
            'improvement': improvement,
            'improvement_pct': improvement_pct
        })
        
        return best_accuracy
    
    def _generate_summary(self):
        """Generate ablation study summary"""
        print("\n" + "="*80)
        print("📊 ABLATION STUDY SUMMARY")
        print("="*80)
        
        print(f"\n{'Stage':<35} {'Features':<10} {'Accuracy':<10} {'Improvement':<15}")
        print("-" * 80)
        
        for result in self.results:
            stage = result['stage']
            n_feat = result['n_features']
            acc = result['accuracy']
            imp = result['improvement']
            imp_pct = result['improvement_pct']
            
            print(f"{stage:<35} {n_feat:<10} {acc:.4f}     {imp:+.4f} ({imp_pct:+.2f}%)")
        
        print("\n" + "="*80)
        print("📈 KEY FINDINGS:")
        print("="*80)
        
        # Find best stage
        best_stage = max(self.results, key=lambda x: x['accuracy'])
        print(f"🏆 Best Stage: {best_stage['stage']}")
        print(f"   Accuracy: {best_stage['accuracy']:.4f}")
        print(f"   Improvement: {best_stage['improvement']:+.4f} ({best_stage['improvement_pct']:+.2f}%)")
        
        # Component contributions
        print(f"\n📊 Component Contributions:")
        for i in range(1, len(self.results)):
            prev_acc = self.results[i-1]['accuracy']
            curr_acc = self.results[i]['accuracy']
            contribution = curr_acc - prev_acc
            contribution_pct = (contribution / prev_acc) * 100 if prev_acc > 0 else 0
            
            print(f"   {self.results[i]['stage']}: {contribution:+.4f} ({contribution_pct:+.2f}%)")
        
        # Save results
        self._save_results()
    
    def _save_results(self):
        """Save results to CSV and pickle"""
        import pandas as pd
        
        # Save to CSV
        df_results = pd.DataFrame(self.results)
        df_results.to_csv('ablation_study_multiclass_results.csv', index=False)
        print(f"\n💾 Results saved to: ablation_study_multiclass_results.csv")
        
        # Save to pickle
        with open('ablation_study_multiclass_results.pkl', 'wb') as f:
            pickle.dump(self.results, f)
        print(f"💾 Results saved to: ablation_study_multiclass_results.pkl")


def main():
    """Main function to run ablation study"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python ablation_study_multiclass.py <train_csv> [test_csv]")
        print("Example: python ablation_study_multiclass.py UNSW_balanced_train.csv UNSW_balanced_test.csv")
        sys.exit(1)
    
    train_csv = sys.argv[1]
    test_csv = sys.argv[2] if len(sys.argv) > 2 else None
    
    print("🔬 MULTICLASS ABLATION STUDY")
    print("="*80)
    print(f"Training data: {train_csv}")
    if test_csv:
        print(f"Test data: {test_csv}")
    print("="*80)
    
    # Run ablation study
    study = MulticlassAblationStudy(train_csv, test_csv)
    results = study.run_ablation_study()
    
    print("\n✅ Ablation study complete!")
    print(f"📊 {len(results)} stages evaluated")
    print(f"💾 Results saved to CSV and pickle files")


if __name__ == "__main__":
    main()
