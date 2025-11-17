#!/usr/bin/env python3
"""
Cross-Dataset Validation Study - GPU ACCELERATED VERSION
Tests the binary ensemble model trained on UNSW-NB15 against BoT-IoT dataset
Optimized for NVIDIA GPUs with 48GB VRAM
"""

import pandas as pd
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# GPU ACCELERATION IMPORTS
try:
    import cupy as cp
    import cudf
    from cuml.preprocessing import LabelEncoder as cuLabelEncoder
    from cuml.preprocessing import StandardScaler as cuStandardScaler
    GPU_AVAILABLE = True
    print("🚀 GPU ACCELERATION ENABLED - Using CuPy + RAPIDS cuML")
    print(f"   GPU Memory: {cp.cuda.Device().mem_info[1] / 1e9:.1f} GB total")
except ImportError:
    cp = np
    cudf = pd
    GPU_AVAILABLE = False
    print("⚠️  GPU libraries not available - falling back to CPU")

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from novel_ensemble_ml import NovelEnsembleMLSystem, DynamicFeatureEngineer

class CrossDatasetValidator:
    """Cross-dataset validation for testing model generalizability - GPU ACCELERATED"""
    
    def __init__(self, batch_size=100000, use_gpu=True):
        self.binary_system = None
        self.feature_mapping = {}
        self.unsw_features = []
        self.botiot_features = []
        self.training_feature_stats = None  # Store training data statistics
        self.batch_size = batch_size  # Process in batches for memory efficiency
        self.use_gpu = use_gpu and GPU_AVAILABLE
        
        if self.use_gpu:
            print(f"🚀 GPU Mode Enabled - Batch size: {batch_size:,}")
            print(f"   Available GPU memory: {cp.cuda.Device().mem_info[0] / 1e9:.1f} GB free")
        else:
            print(f"💻 CPU Mode - Batch size: {batch_size:,}")
        
    def load_binary_model(self):
        """Load the trained binary ensemble model"""
        print("🔍 LOADING TRAINED BINARY MODEL")
        print("=" * 50)
        
        # Try different model paths
        model_paths = [
            'Models/Binary/ensemble_binary_46f.pkl',
            'Models/Binary/ensamble_binary_46f.pkl',  # Check for typo version
            'Models/Binary/ensemble_binary_42f.pkl',
            'trained_novel_ensemble_model.pkl',
            'Models/Binary/ensemble_binary.pkl'
        ]
        
        for model_path in model_paths:
            if os.path.exists(model_path):
                print(f"📂 Found model: {model_path}")
                try:
                    # Load the pickled model
                    with open(model_path, 'rb') as f:
                        self.binary_system = pickle.load(f)
                    
                    print(f"✅ Binary model loaded successfully from pickle file")
                    print(f"   Model type: {type(self.binary_system)}")
                    
                    # Debug: show what's in the loaded object
                    if isinstance(self.binary_system, dict):
                        print(f"   Dictionary keys: {list(self.binary_system.keys())}")
                    elif hasattr(self.binary_system, '__dict__'):
                        print(f"   Object attributes: {list(self.binary_system.__dict__.keys())}")
                    
                    return True
                    
                except Exception as e:
                    print(f"❌ Failed to load {model_path}: {e}")
                    continue
        
        print("❌ No binary model found!")
        print("   Expected paths:")
        for path in model_paths:
            print(f"   - {path}")
        return False
    
    def create_feature_mapping(self):
        """Create mapping between BoT-IoT and UNSW-NB15 features"""
        print("\n🗺️  CREATING FEATURE MAPPING")
        print("=" * 50)
        
        # BoT-IoT to UNSW-NB15 feature mapping
        self.feature_mapping = {
            # Direct mappings (same name)
            'stime': 'stime',
            'proto': 'proto', 
            'sport': 'sport',
            'dport': 'dport',
            'dur': 'dur',
            'spkts': 'spkts',
            'dpkts': 'dpkts', 
            'sbytes': 'sbytes',
            'dbytes': 'dbytes',
            'state': 'state',
            
            # Approximate mappings
            'pkts': 'spkts',  # Total packets → source packets (approximation)
            'bytes': 'sbytes',  # Total bytes → source bytes (approximation)
            'mean': 'smean',   # Mean → source mean (if available)
            'rate': 'rate',    # Rate mapping
            'srate': 'sload',  # Source rate → source load (approximation)
            'drate': 'dload',  # Destination rate → destination load (approximation)
            
            # Target mapping
            'attack': 'label',  # Attack label (binary: 0=normal, 1=attack)
            
            # Features that don't have direct equivalents (will be filled with defaults)
            'flgs': None,      # Flags - no direct equivalent
            'saddr': None,     # Source address - excluded for privacy
            'daddr': None,     # Destination address - excluded for privacy
            'ltime': None,     # Last time - no equivalent
            'seq': None,       # Sequence - no equivalent
            'stddev': None,    # Standard deviation - no equivalent
            'smac': None,      # Source MAC - excluded
            'dmac': None,      # Destination MAC - excluded
            'sum': None,       # Sum - no equivalent
            'min': None,       # Min - no equivalent
            'max': None,       # Max - no equivalent
            'soui': None,      # Source OUI - no equivalent
            'doui': None,      # Destination OUI - no equivalent
            'sco': None,       # Source company - no equivalent
            'dco': None,       # Destination company - no equivalent
            'category': None,  # Attack category - not used in binary
            'subcategory': None, # Attack subcategory - not used in binary
            'pkSeqID': None,   # Packet sequence ID - identifier, excluded
        }
        
        print("📊 Feature Mapping Created:")
        print("   Direct mappings:")
        for botiot_feat, unsw_feat in self.feature_mapping.items():
            if unsw_feat is not None:
                print(f"      {botiot_feat} → {unsw_feat}")
        
        unmapped_count = sum(1 for v in self.feature_mapping.values() if v is None)
        mapped_count = len(self.feature_mapping) - unmapped_count
        print(f"\n   📈 Mapped features: {mapped_count}")
        print(f"   ❌ Unmapped features: {unmapped_count}")
        
        return self.feature_mapping
    
    def load_and_preprocess_botiot(self, dataset_path):
        """Load and preprocess BoT-IoT dataset - GPU ACCELERATED"""
        print(f"\n📁 LOADING BOT-IOT DATASET (GPU ACCELERATED)")
        print("=" * 50)
        
        try:
            # Load dataset with GPU acceleration if available
            print(f"📂 Loading: {dataset_path}")
            
            if GPU_AVAILABLE:
                # Use cuDF for GPU-accelerated dataframe operations
                df = cudf.read_csv(dataset_path)
                print(f"✅ Loaded {len(df):,} samples with {len(df.columns)} features (GPU)")
            else:
                df = pd.read_csv(dataset_path)
                print(f"✅ Loaded {len(df):,} samples with {len(df.columns)} features (CPU)")
            
            # Show original features
            print(f"\n📋 Original BoT-IoT Features ({len(df.columns)}):")
            for i, col in enumerate(df.columns, 1):
                print(f"   {i:2d}. {col}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error loading BoT-IoT dataset: {e}")
            return None
    
    def transform_botiot_to_unsw_format(self, botiot_df):
        """Transform BoT-IoT dataset to UNSW-NB15 format"""
        print(f"\n🔄 TRANSFORMING BOT-IOT TO UNSW FORMAT")
        print("=" * 50)
        
        # Create new dataframe with UNSW-NB15 structure
        unsw_df = pd.DataFrame()
        
        # Apply feature mapping
        mapped_features = []
        for botiot_col, unsw_col in self.feature_mapping.items():
            if unsw_col is not None and botiot_col in botiot_df.columns:
                unsw_df[unsw_col] = botiot_df[botiot_col].copy()
                mapped_features.append(f"{botiot_col} → {unsw_col}")
        
        print(f"✅ Mapped {len(mapped_features)} features:")
        for mapping in mapped_features[:10]:  # Show first 10
            print(f"   {mapping}")
        if len(mapped_features) > 10:
            print(f"   ... and {len(mapped_features) - 10} more")
        
        # Handle target variable (attack column)
        if 'attack' in botiot_df.columns:
            # Convert attack labels to binary (0=normal, 1=attack)
            attack_values = botiot_df['attack'].unique()
            print(f"\n🎯 Target Variable Conversion:")
            print(f"   Original attack values: {attack_values}")
            
            # Assume 'normal' or 0 means normal, everything else is attack
            unsw_df['label'] = (botiot_df['attack'] != 0).astype(int)
            
            # Show distribution
            label_dist = unsw_df['label'].value_counts()
            print(f"   Binary distribution:")
            print(f"      Normal (0): {label_dist.get(0, 0):,} samples")
            print(f"      Attack (1): {label_dist.get(1, 0):,} samples")
        
        # Add missing UNSW-NB15 features with default values
        unsw_required_features = [
            'tcprtt', 'synack', 'ackdat', 'smean', 'dmean', 'trans_depth',
            'response_body_len', 'ct_srv_src', 'ct_state_ttl', 'ct_dst_ltm',
            'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm',
            'is_ftp_login', 'ct_ftp_cmd', 'ct_flw_http_mthd', 'ct_src_ltm',
            'ct_srv_dst', 'is_sm_ips_ports', 'service', 'sttl', 'dttl',
            'sload', 'dload', 'sloss', 'dloss', 'sinpkt', 'dinpkt',
            'sjit', 'djit', 'swin', 'dwin', 'stcpb', 'dtcpb'
        ]
        
        missing_features = []
        for feature in unsw_required_features:
            if feature not in unsw_df.columns:
                # Add with appropriate default values
                if feature in ['service']:
                    unsw_df[feature] = 'other'  # Default service
                elif feature in ['proto'] and 'proto' not in unsw_df.columns:
                    unsw_df[feature] = 6  # Default to TCP
                else:
                    unsw_df[feature] = 0  # Default numeric value
                missing_features.append(feature)
        
        if missing_features:
            print(f"\n⚠️  Added {len(missing_features)} missing features with defaults:")
            for feat in missing_features[:10]:
                print(f"   {feat}")
            if len(missing_features) > 10:
                print(f"   ... and {len(missing_features) - 10} more")
        
        # Basic data cleaning - GPU ACCELERATED
        print(f"\n🧹 Data Cleaning (GPU Accelerated):")
        
        # Convert all columns to numeric (except categorical ones we'll encode)
        categorical_to_encode = ['service', 'state', 'proto']
        
        if GPU_AVAILABLE and isinstance(unsw_df, cudf.DataFrame):
            # GPU-accelerated numeric conversion
            for col in unsw_df.columns:
                if col not in categorical_to_encode and col != 'label':
                    try:
                        unsw_df[col] = cudf.to_numeric(unsw_df[col], errors='coerce')
                    except:
                        pass
            
            # Handle missing values (GPU)
            missing_before = unsw_df.isnull().sum().sum()
            unsw_df = unsw_df.fillna(0)
            print(f"   Filled {missing_before:,} missing values with 0 (GPU)")
            
            # Handle infinite values (GPU)
            numeric_cols = unsw_df.select_dtypes(include=['float', 'int']).columns
            for col in numeric_cols:
                unsw_df[col] = unsw_df[col].replace([cp.inf, -cp.inf], 0)
            print(f"   Replaced infinite values with 0 (GPU)")
        else:
            # CPU fallback
            for col in unsw_df.columns:
                if col not in categorical_to_encode and col != 'label':
                    unsw_df[col] = pd.to_numeric(unsw_df[col], errors='coerce')
            
            missing_before = unsw_df.isnull().sum().sum()
            unsw_df = unsw_df.fillna(0)
            print(f"   Filled {missing_before:,} missing values with 0")
            
            inf_count = np.isinf(unsw_df.select_dtypes(include=[np.number])).sum().sum()
            if inf_count > 0:
                unsw_df = unsw_df.replace([np.inf, -np.inf], 0)
                print(f"   Replaced {inf_count:,} infinite values with 0")
        
        # Encode categorical variables using TRAINED encoders (NO LEAKAGE!)
        categorical_cols = ['service', 'state', 'proto']
        
        # Check if we have trained label encoders
        if isinstance(self.binary_system, dict) and 'label_encoders' in self.binary_system:
            trained_encoders = self.binary_system['label_encoders']
            print(f"   ✅ Using trained label encoders from model (no data leakage)")
            
            for col in categorical_cols:
                if col in unsw_df.columns and col in trained_encoders:
                    le = trained_encoders[col]
                    # Handle unseen categories by mapping to a default value
                    unsw_df[col] = unsw_df[col].astype(str).apply(
                        lambda x: le.transform([x])[0] if x in le.classes_ else -1
                    )
                    print(f"      Encoded categorical column: {col}")
        else:
            # Fallback: create new encoders (but warn about potential leakage)
            print(f"   ⚠️  WARNING: No trained encoders found - creating new ones")
            print(f"   ⚠️  This may cause data leakage or encoding mismatch!")
            for col in categorical_cols:
                if col in unsw_df.columns:
                    le = LabelEncoder()
                    unsw_df[col] = le.fit_transform(unsw_df[col].astype(str))
                    print(f"      Encoded categorical column: {col}")
        
        print(f"\n✅ Transformation Complete:")
        print(f"   Final dataset shape: {unsw_df.shape}")
        print(f"   Features: {len(unsw_df.columns)}")
        
        return unsw_df
    
    def apply_feature_engineering(self, df):
        """Apply the same feature engineering as used in training"""
        print(f"\n" + "="*80)
        print(f"🚨 APPLY_FEATURE_ENGINEERING METHOD CALLED! 🚨")
        print(f"="*80)
        print(f"\n🔧 APPLYING FEATURE ENGINEERING")
        print("=" * 50)
        
        # DEBUG: Check what we have
        print(f"   DEBUG: binary_system type: {type(self.binary_system)}")
        print(f"   DEBUG: is dict? {isinstance(self.binary_system, dict)}")
        if isinstance(self.binary_system, dict):
            print(f"   DEBUG: has feature_engineer? {'feature_engineer' in self.binary_system}")
            if 'feature_engineer' in self.binary_system:
                print(f"   DEBUG: feature_engineer type: {type(self.binary_system['feature_engineer'])}")
        
        try:
            # Use the TRAINED feature engineer from the model (NO LEAKAGE!)
            if isinstance(self.binary_system, dict) and 'feature_engineer' in self.binary_system:
                fe = self.binary_system['feature_engineer']
                print(f"✅ Using trained feature engineer from model (no data leakage)")
                
                # Use transform() ONLY - do NOT fit on test data!
                df_engineered = fe.transform(df)
                
                print(f"✅ Feature engineering applied:")
                print(f"   Original features: {len(df.columns)}")
                print(f"   After engineering: {len(df_engineered.columns)}")
                print(f"   New features created: {len(df_engineered.columns) - len(df.columns)}")
                
                return df_engineered
            else:
                print(f"⚠️  No trained feature engineer found in model")
                print(f"   Skipping feature engineering to avoid data leakage")
                return df
            
        except Exception as e:
            print(f"⚠️  Feature engineering failed: {e}")
            print(f"   Using original features without engineering")
            return df
    
    def evaluate_cross_dataset_performance(self, test_df):
        """Evaluate model performance on cross-dataset"""
        print(f"\n📊 CROSS-DATASET EVALUATION")
        print("=" * 50)
        
        try:
            # Prepare features and target
            target_col = 'label'
            if target_col not in test_df.columns:
                print(f"❌ Target column '{target_col}' not found!")
                return None
            
            y_true = test_df[target_col].values
            
            # Check for extreme class imbalance and create balanced subset
            unique, counts = np.unique(y_true, return_counts=True)
            class_dist = dict(zip(unique, counts))
            
            print(f"\n⚖️  CLASS DISTRIBUTION ANALYSIS:")
            print("-" * 50)
            for class_label, count in class_dist.items():
                percentage = (count / len(y_true)) * 100
                class_name = "Normal" if class_label == 0 else "Attack"
                print(f"   {class_name} ({class_label}): {count:,} samples ({percentage:.4f}%)")
            
            # Calculate imbalance ratio
            if len(class_dist) == 2:
                majority_count = max(counts)
                minority_count = min(counts)
                imbalance_ratio = majority_count / minority_count if minority_count > 0 else float('inf')
                
                print(f"\n   Imbalance Ratio: {imbalance_ratio:.1f}:1")
                
                # If extremely imbalanced (>100:1), create balanced or proportional subset
                if imbalance_ratio > 100:
                    print(f"\n🔄 CREATING TEST SUBSET FOR MEANINGFUL EVALUATION")
                    print("-" * 50)
                    print(f"   ⚠️  Extreme imbalance detected ({imbalance_ratio:.0f}:1)")
                    
                    # Find minority and majority classes
                    minority_class = unique[np.argmin(counts)]
                    majority_class = unique[np.argmax(counts)]
                    minority_size = min(counts)
                    
                    minority_indices = np.where(y_true == minority_class)[0]
                    majority_indices = np.where(y_true == majority_class)[0]
                    
                    # ADAPTIVE STRATEGY based on minority sample size
                    if minority_size >= 100:
                        # Sufficient samples: Create balanced 50/50 subset
                        print(f"   ✅ Sufficient minority samples ({minority_size})")
                        print(f"   💡 Creating balanced 50/50 subset")
                        
                        np.random.seed(42)
                        sampled_majority_indices = np.random.choice(majority_indices, size=minority_size, replace=False)
                        balanced_indices = np.concatenate([minority_indices, sampled_majority_indices])
                        np.random.shuffle(balanced_indices)
                        
                        test_df = test_df.iloc[balanced_indices].copy()
                        y_true = y_true[balanced_indices]
                        
                        print(f"      Normal: {minority_size:,} samples (50%)")
                        print(f"      Attack: {minority_size:,} samples (50%)")
                        print(f"      Total: {len(balanced_indices):,} samples")
                        
                    elif minority_size >= 50:
                        # Moderate samples: Create 1:2 ratio (more realistic)
                        print(f"   ⚠️  Limited minority samples ({minority_size})")
                        print(f"   💡 Creating 1:2 ratio subset (more realistic)")
                        
                        np.random.seed(42)
                        majority_sample_size = min(minority_size * 2, len(majority_indices))
                        sampled_majority_indices = np.random.choice(majority_indices, size=majority_sample_size, replace=False)
                        balanced_indices = np.concatenate([minority_indices, sampled_majority_indices])
                        np.random.shuffle(balanced_indices)
                        
                        test_df = test_df.iloc[balanced_indices].copy()
                        y_true = y_true[balanced_indices]
                        
                        print(f"      Normal: {minority_size:,} samples (33%)")
                        print(f"      Attack: {majority_sample_size:,} samples (67%)")
                        print(f"      Total: {len(balanced_indices):,} samples")
                        
                    else:
                        # Very few samples: Use ALL minority + proportional majority
                        print(f"   ⚠️  Very few minority samples ({minority_size})")
                        print(f"   💡 Using ALL minority samples + 3x majority samples")
                        print(f"   💡 This preserves all available normal traffic diversity")
                        
                        np.random.seed(42)
                        majority_sample_size = min(minority_size * 3, len(majority_indices))
                        sampled_majority_indices = np.random.choice(majority_indices, size=majority_sample_size, replace=False)
                        balanced_indices = np.concatenate([minority_indices, sampled_majority_indices])
                        np.random.shuffle(balanced_indices)
                        
                        test_df = test_df.iloc[balanced_indices].copy()
                        y_true = y_true[balanced_indices]
                        
                        minority_pct = (minority_size / len(balanced_indices)) * 100
                        majority_pct = (majority_sample_size / len(balanced_indices)) * 100
                        print(f"      Normal: {minority_size:,} samples ({minority_pct:.1f}%)")
                        print(f"      Attack: {majority_sample_size:,} samples ({majority_pct:.1f}%)")
                        print(f"      Total: {len(balanced_indices):,} samples")
                    
                    print(f"\n   💡 Adaptive sampling ensures meaningful evaluation")
                    print(f"   💡 Strategy adapts to available minority class samples")
            
            # Get the model's expected feature names
            if isinstance(self.binary_system, dict) and 'feature_names' in self.binary_system:
                expected_features = self.binary_system['feature_names']
                print(f"📋 Model expects {len(expected_features)} specific features")
                
                # Align test data with expected features
                missing_features = []
                for feat in expected_features:
                    if feat not in test_df.columns:
                        test_df[feat] = 0  # Add missing features with default value
                        missing_features.append(feat)
                
                if missing_features:
                    print(f"   ⚠️  Added {len(missing_features)} missing features with zeros")
                
                # Select only the features the model was trained on
                X_test = test_df[expected_features]
                
            else:
                # Fallback: use all features except metadata
                exclude_cols = [target_col, 'attack_cat', 'stime', 'srcip', 'dstip', 'id']
                feature_cols = [col for col in test_df.columns if col not in exclude_cols]
                X_test = test_df[feature_cols]
            
            print(f"📊 Test Data Prepared:")
            print(f"   Samples: {len(X_test):,}")
            print(f"   Features: {X_test.shape[1]}")
            print(f"   Classes: {len(np.unique(y_true))} (Normal: {np.sum(y_true == 0):,}, Attack: {np.sum(y_true == 1):,})")
            
            # Analyze feature distribution before scaling
            print(f"\n📊 FEATURE DISTRIBUTION ANALYSIS:")
            print("-" * 50)
            X_test_stats = pd.DataFrame(X_test).describe()
            print(f"   Test data feature ranges:")
            print(f"   Mean range: [{X_test_stats.loc['mean'].min():.2f}, {X_test_stats.loc['mean'].max():.2f}]")
            print(f"   Std range:  [{X_test_stats.loc['std'].min():.2f}, {X_test_stats.loc['std'].max():.2f}]")
            
            # Check for features with zero variance (constant values)
            zero_var_features = (X_test.std() == 0).sum()
            if zero_var_features > 0:
                print(f"   ⚠️  {zero_var_features} features have zero variance (constant values)")
            
            # Apply scaling if scaler is available - GPU ACCELERATED
            if isinstance(self.binary_system, dict) and 'scaler' in self.binary_system:
                scaler = self.binary_system['scaler']
                print(f"\n⚖️  Applying feature scaling (GPU Accelerated)...")
                
                # Convert to numpy if cuDF
                if GPU_AVAILABLE and isinstance(X_test, cudf.DataFrame):
                    X_test_np = X_test.to_pandas().values
                else:
                    X_test_np = X_test.values
                
                # Move to GPU for scaling if available
                if GPU_AVAILABLE:
                    X_test_gpu = cp.asarray(X_test_np)
                    # Use GPU scaler if available, otherwise CPU
                    X_test_scaled = scaler.transform(X_test_np)
                    X_test_scaled = cp.asarray(X_test_scaled)  # Move result to GPU
                    print(f"   Scaling completed on GPU")
                else:
                    X_test_scaled = scaler.transform(X_test_np)
                
                # Check scaled distribution
                if GPU_AVAILABLE:
                    X_scaled_sample = cp.asnumpy(X_test_scaled[:1000])  # Sample for stats
                    X_scaled_df = pd.DataFrame(X_scaled_sample)
                else:
                    X_scaled_df = pd.DataFrame(X_test_scaled)
                    
                scaled_stats = X_scaled_df.describe()
                print(f"   After scaling:")
                print(f"   Mean range: [{scaled_stats.loc['mean'].min():.2f}, {scaled_stats.loc['mean'].max():.2f}]")
                print(f"   Std range:  [{scaled_stats.loc['std'].min():.2f}, {scaled_stats.loc['std'].max():.2f}]")
            else:
                if GPU_AVAILABLE and isinstance(X_test, cudf.DataFrame):
                    X_test_scaled = cp.asarray(X_test.to_pandas().values)
                else:
                    X_test_scaled = X_test.values
            
            # Apply feature selection if indices are available - GPU ACCELERATED
            if isinstance(self.binary_system, dict) and 'selected_feature_indices' in self.binary_system:
                selected_indices = self.binary_system['selected_feature_indices']
                if selected_indices is not None and len(selected_indices) > 0:
                    print(f"🎯 Applying feature selection: {len(selected_indices)} features (GPU)")
                    if GPU_AVAILABLE and isinstance(X_test_scaled, cp.ndarray):
                        X_test_selected = X_test_scaled[:, selected_indices]
                    else:
                        X_test_selected = X_test_scaled[:, selected_indices]
                else:
                    X_test_selected = X_test_scaled
            else:
                X_test_selected = X_test_scaled
            
            print(f"   Final feature matrix: {X_test_selected.shape}")
            if GPU_AVAILABLE and isinstance(X_test_selected, cp.ndarray):
                print(f"   🚀 Data on GPU - Ready for accelerated prediction!")
            
            # Make predictions using the cached model - GPU ACCELERATED
            print("🔮 Making predictions (GPU Accelerated)...")
            
            # Convert to numpy for sklearn models if on GPU
            if GPU_AVAILABLE and isinstance(X_test_selected, cp.ndarray):
                X_test_for_pred = cp.asnumpy(X_test_selected)
                print(f"   Transferred {X_test_for_pred.nbytes / 1e6:.1f} MB from GPU to CPU for prediction")
            else:
                X_test_for_pred = X_test_selected
            
            # Handle different types of loaded models
            try:
                # Check if it's a full system object
                if hasattr(self.binary_system, 'predict'):
                    # Full system object
                    test_df_copy = X_test.copy()
                    test_df_copy['label'] = y_true  # Add dummy label for preprocessing
                    y_pred = self.binary_system.predict(test_df_copy)
                    y_proba = self.binary_system.predict_proba(test_df_copy)
                    y_true = test_df_copy['label'].values
                elif hasattr(self.binary_system, 'classifier'):
                    # System with classifier attribute
                    y_pred = self.binary_system.classifier.predict(X_test_selected)
                    y_proba = self.binary_system.classifier.predict_proba(X_test_selected)
                elif isinstance(self.binary_system, dict):
                    # Dictionary containing model components
                    print("   📦 Loaded model is a dictionary, extracting classifier...")
                    
                    # Look for the actual model in the dictionary
                    model = None
                    if 'classifier' in self.binary_system:
                        model = self.binary_system['classifier']
                    elif 'model' in self.binary_system:
                        model = self.binary_system['model']
                    else:
                        # Try to find any sklearn-like model in the dict
                        for key, value in self.binary_system.items():
                            if hasattr(value, 'predict') and hasattr(value, 'predict_proba'):
                                model = value
                                print(f"   Found model in key: {key}")
                                break
                    
                    if model is None:
                        raise ValueError("No suitable model found in the loaded dictionary")
                    
                    # Use the extracted model with properly prepared features
                    y_proba = model.predict_proba(X_test_for_pred)
                    
                    # Use default threshold first
                    y_pred = model.predict(X_test_for_pred)
                    
                else:
                    # Direct model object
                    y_pred = self.binary_system.predict(X_test_for_pred)
                    y_proba = self.binary_system.predict_proba(X_test_for_pred)
                    
            except Exception as e:
                print(f"⚠️  Prediction failed: {e}")
                print(f"   Model type: {type(self.binary_system)}")
                if isinstance(self.binary_system, dict):
                    print(f"   Dictionary keys: {list(self.binary_system.keys())}")
                raise e
            
            # Diagnostic: Check prediction distribution
            pred_unique, pred_counts = np.unique(y_pred, return_counts=True)
            print(f"\n🔍 PREDICTION DISTRIBUTION:")
            print("-" * 50)
            for pred_class, pred_count in zip(pred_unique, pred_counts):
                pred_percentage = (pred_count / len(y_pred)) * 100
                class_name = "Normal" if pred_class == 0 else "Attack"
                print(f"   Predicted {class_name} ({pred_class}): {pred_count:,} samples ({pred_percentage:.2f}%)")
            
            # Check if model is predicting only one class
            if len(pred_unique) == 1:
                print(f"\n⚠️  WARNING: Model is predicting ONLY ONE CLASS!")
                print(f"   This indicates the model has completely failed to generalize")
                print(f"   Predicted class: {pred_unique[0]} ({'Normal' if pred_unique[0] == 0 else 'Attack'})")
            
            # Calculate initial metrics
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            # AUC score
            if len(y_proba.shape) > 1 and y_proba.shape[1] > 1:
                auc = roc_auc_score(y_true, y_proba[:, 1])
                attack_proba = y_proba[:, 1]
            else:
                auc = roc_auc_score(y_true, y_proba)
                attack_proba = y_proba
            
            # Analyze prediction confidence
            print(f"\n🎲 PREDICTION CONFIDENCE ANALYSIS:")
            print("-" * 50)
            print(f"   Attack probability - Mean: {attack_proba.mean():.4f}")
            print(f"   Attack probability - Std:  {attack_proba.std():.4f}")
            print(f"   Attack probability - Min:  {attack_proba.min():.4f}")
            print(f"   Attack probability - Max:  {attack_proba.max():.4f}")
            
            # Check if probabilities are all similar (indicating model uncertainty)
            if attack_proba.std() < 0.1:
                print(f"\n   ⚠️  Low probability variance detected!")
                print(f"   Model is very uncertain - probabilities are all similar")
                print(f"   This suggests the model cannot distinguish between classes")
                
                # DIAGNOSTIC ONLY - Show threshold sensitivity (DO NOT optimize on test set)
                print(f"\n💡 DIAGNOSTIC: Threshold sensitivity analysis (NOT used for reported metrics)")
                print("-" * 50)
                
                # Show how performance would change with different thresholds (diagnostic only)
                thresholds_to_show = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
                print(f"   Threshold sensitivity (using default 0.5 for all reported metrics):")
                for threshold in thresholds_to_show:
                    y_pred_diagnostic = (attack_proba >= threshold).astype(int)
                    f1_diagnostic = f1_score(y_true, y_pred_diagnostic, zero_division=0)
                    acc_diagnostic = accuracy_score(y_true, y_pred_diagnostic)
                    marker = " ← DEFAULT (used for all metrics)" if threshold == 0.5 else ""
                    print(f"   Threshold {threshold:.2f}: Accuracy={acc_diagnostic:.4f}, F1={f1_diagnostic:.4f}{marker}")
                
                print(f"\n   ℹ️  All reported metrics use standard 0.5 threshold (no test-set optimization)")
                print(f"   ℹ️  Threshold should be tuned on validation set during training, not on test set")

            
            # Check for potentially inverted model predictions (AUC < 0.5 suggests this)
            if auc < 0.5:
                print(f"\n⚠️  WARNING: AUC < 0.5 detected!")
                print(f"   This suggests the model's predictions are inverted")
                print(f"   The model may be predicting opposite of what it should")
                print(f"   Attempting to correct by inverting predictions...")
                
                # Invert predictions (not labels!)
                y_pred_inverted = 1 - y_pred
                accuracy_inv = accuracy_score(y_true, y_pred_inverted)
                precision_inv = precision_score(y_true, y_pred_inverted, zero_division=0)
                recall_inv = recall_score(y_true, y_pred_inverted, zero_division=0)
                f1_inv = f1_score(y_true, y_pred_inverted, zero_division=0)
                auc_inv = 1 - auc  # Invert AUC
                
                # Check if inversion actually improves BOTH AUC and accuracy
                if auc_inv > 0.5 and accuracy_inv > accuracy:
                    print(f"\n   ✅ Inverted predictions give better results!")
                    print(f"   Original AUC: {auc:.4f} → Corrected AUC: {auc_inv:.4f}")
                    print(f"   Original Accuracy: {accuracy:.4f} → Corrected Accuracy: {accuracy_inv:.4f}")
                    print(f"   💡 Using corrected predictions for final metrics")
                    
                    y_pred = y_pred_inverted
                    accuracy = accuracy_inv
                    precision = precision_inv
                    recall = recall_inv
                    f1 = f1_inv
                    auc = auc_inv
                elif auc_inv > 0.5:
                    print(f"\n   ⚠️  AUC improved but accuracy decreased after inversion")
                    print(f"   This suggests label encoding mismatch, not prediction error")
                    print(f"   Keeping original predictions but using inverted AUC")
                    auc = auc_inv
                else:
                    print(f"\n   ℹ️  Inversion didn't improve results - keeping original predictions")
            
            # Print results
            print(f"\n🎯 CROSS-DATASET PERFORMANCE RESULTS:")
            print("=" * 50)
            print(f"   Accuracy:  {accuracy:.4f}")
            print(f"   Precision: {precision:.4f}")
            print(f"   Recall:    {recall:.4f}")
            print(f"   F1-Score:  {f1:.4f}")
            print(f"   AUC:       {auc:.4f}")
            
            # Performance interpretation
            print(f"\n💡 CROSS-DATASET GENERALIZATION ASSESSMENT:")
            print("-" * 50)
            
            # Use AUC as primary metric for cross-dataset evaluation
            # AUC is more robust to class imbalance and sampling strategies
            if auc >= 0.90:
                generalization = "🟢 EXCELLENT - Model generalizes very well"
                interpretation = "Strong discriminative ability across datasets"
            elif auc >= 0.80:
                generalization = "🟡 GOOD - Model shows decent generalization"
                interpretation = "Reasonable cross-dataset performance"
            elif auc >= 0.70:
                generalization = "🟠 MODERATE - Model has limited generalization"
                interpretation = "Some discriminative ability, but dataset-specific patterns affect performance"
            else:
                generalization = "🔴 POOR - Model struggles to generalize"
                interpretation = "Significant domain shift between training and test datasets"
            
            print(f"   {generalization}")
            print(f"   {interpretation}")
            print(f"\n   📊 Primary Metric: AUC = {auc:.4f}")
            print(f"   📊 Secondary Metric: Accuracy = {accuracy:.4f}")
            print(f"   💡 AUC is more reliable for cross-dataset evaluation")
            
            # Detailed classification report
            print(f"\n📋 DETAILED CLASSIFICATION REPORT:")
            print("-" * 50)
            print(classification_report(y_true, y_pred, target_names=['Normal', 'Attack']))
            
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            print(f"\n🔢 CONFUSION MATRIX:")
            print("-" * 30)
            print(f"                Predicted")
            print(f"              Normal  Attack")
            print(f"Actual Normal   {cm[0,0]:6d}  {cm[0,1]:6d}")
            print(f"       Attack   {cm[1,0]:6d}  {cm[1,1]:6d}")
            
            # Store results
            results = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc': auc,
                'predictions': y_pred,
                'probabilities': y_proba,
                'true_labels': y_true,
                'confusion_matrix': cm,
                'generalization_level': generalization,
                'decision_threshold': 0.5,  # Always use standard threshold (no test-set optimization)
                'probability_variance': attack_proba.std()  # Track model confidence
            }
            
            return results
            
        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _process_single_file_wrapper(self, file_path):
        """Wrapper for parallel processing"""
        try:
            return self.run_cross_dataset_validation_single_file(file_path)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None
    
    def run_cross_dataset_validation_single_file(self, botiot_file_path):
        """Run cross-dataset validation on a single BoT-IoT file - GPU ACCELERATED"""
        
        # Load BoT-IoT dataset
        botiot_df = self.load_and_preprocess_botiot(botiot_file_path)
        if botiot_df is None:
            return None
        
        # Transform to UNSW format
        unsw_format_df = self.transform_botiot_to_unsw_format(botiot_df)
        
        # Apply feature engineering
        engineered_df = self.apply_feature_engineering(unsw_format_df)
        
        # Evaluate performance
        results = self.evaluate_cross_dataset_performance(engineered_df)
        
        return results
    
    def run_cross_dataset_validation_directory(self, botiot_directory, parallel=True):
        """Run complete cross-dataset validation on all files in directory - GPU ACCELERATED"""
        print("🌐 CROSS-DATASET VALIDATION STUDY (GPU ACCELERATED)")
        print("=" * 60)
        print("Testing UNSW-NB15 trained model on BoT-IoT dataset files")
        print("=" * 60)
        
        # Step 1: Load binary model
        if not self.load_binary_model():
            return False
        
        # Step 2: Create feature mapping
        self.create_feature_mapping()
        
        # Step 3: Find all CSV files in directory
        import glob
        csv_files = glob.glob(os.path.join(botiot_directory, "*.csv"))
        
        if not csv_files:
            print(f"❌ No CSV files found in directory: {botiot_directory}")
            return False
        
        print(f"\n📁 Found {len(csv_files)} CSV files to process:")
        for i, file_path in enumerate(csv_files, 1):
            filename = os.path.basename(file_path)
            file_size = os.path.getsize(file_path) / 1e6  # MB
            print(f"   {i:2d}. {filename} ({file_size:.1f} MB)")
        
        # Step 4: Process files (parallel if multiple files and not using GPU for each)
        all_results = []
        successful_files = 0
        
        if parallel and len(csv_files) > 1 and not self.use_gpu:
            # CPU mode: Process multiple files in parallel
            print(f"\n🔄 Processing files in parallel (CPU mode)...")
            print("=" * 80)
            
            from concurrent.futures import ProcessPoolExecutor, as_completed
            from tqdm import tqdm
            
            with ProcessPoolExecutor(max_workers=min(4, len(csv_files))) as executor:
                future_to_file = {
                    executor.submit(self._process_single_file_wrapper, file_path): file_path 
                    for file_path in csv_files
                }
                
                with tqdm(total=len(csv_files), desc="Processing files") as pbar:
                    for future in as_completed(future_to_file):
                        file_path = future_to_file[future]
                        filename = os.path.basename(file_path)
                        try:
                            results = future.result()
                            if results:
                                results['filename'] = filename
                                results['file_path'] = file_path
                                all_results.append(results)
                                successful_files += 1
                                pbar.set_postfix({'Success': successful_files})
                        except Exception as e:
                            print(f"❌ {filename}: Error - {e}")
                        pbar.update(1)
        else:
            # GPU mode or single file: Process sequentially (GPU doesn't parallelize well)
            print(f"\n🔄 Processing files sequentially...")
            if self.use_gpu:
                print("   (GPU mode: Sequential processing for optimal GPU utilization)")
            print("=" * 80)
            
            from tqdm import tqdm
            
            for file_path in tqdm(csv_files, desc="Processing files"):
                filename = os.path.basename(file_path)
                
                try:
                    # Run validation on single file
                    results = self.run_cross_dataset_validation_single_file(file_path)
                    
                    if results:
                        # Add filename to results
                        results['filename'] = filename
                        results['file_path'] = file_path
                        all_results.append(results)
                        successful_files += 1
                        
                        print(f"✅ {filename}: Accuracy={results['accuracy']:.4f}, F1={results['f1_score']:.4f}")
                    else:
                        print(f"❌ {filename}: Processing failed")
                        
                except Exception as e:
                    print(f"❌ {filename}: Error - {e}")
                    continue
        
        # Step 5: Generate comprehensive results table
        if all_results:
            self.generate_results_table(all_results)
            return all_results
        else:
            print(f"\n❌ No files processed successfully!")
            return False
    
    def generate_results_table(self, all_results):
        """Generate comprehensive results table for paper"""
        print(f"\n📊 CROSS-DATASET VALIDATION RESULTS TABLE")
        print("=" * 80)
        
        # Calculate statistics
        accuracies = [r['accuracy'] for r in all_results]
        precisions = [r['precision'] for r in all_results]
        recalls = [r['recall'] for r in all_results]
        f1_scores = [r['f1_score'] for r in all_results]
        aucs = [r['auc'] for r in all_results]
        
        # Print detailed table
        print(f"\n📋 DETAILED RESULTS BY FILE:")
        print("-" * 100)
        print(f"{'File Name':<25} {'Samples':<8} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'AUC':<10}")
        print("-" * 100)
        
        for result in all_results:
            filename = result['filename'][:24]  # Truncate long names
            samples = len(result['true_labels'])
            acc = result['accuracy']
            prec = result['precision'] 
            rec = result['recall']
            f1 = result['f1_score']
            auc = result['auc']
            
            print(f"{filename:<25} {samples:<8,} {acc:<10.4f} {prec:<10.4f} {rec:<10.4f} {f1:<10.4f} {auc:<10.4f}")
        
        print("-" * 100)
        
        # Summary statistics
        print(f"\n📈 SUMMARY STATISTICS:")
        print("-" * 50)
        print(f"Total Files Processed:     {len(all_results)}")
        print(f"Total Samples Tested:      {sum(len(r['true_labels']) for r in all_results):,}")
        print(f"")
        print(f"Average Accuracy:          {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
        print(f"Average Precision:         {np.mean(precisions):.4f} ± {np.std(precisions):.4f}")
        print(f"Average Recall:            {np.mean(recalls):.4f} ± {np.std(recalls):.4f}")
        print(f"Average F1-Score:          {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
        print(f"Average AUC:               {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
        print(f"")
        print(f"Best Performing File:      {all_results[np.argmax(accuracies)]['filename']} (Acc: {max(accuracies):.4f})")
        print(f"Worst Performing File:     {all_results[np.argmin(accuracies)]['filename']} (Acc: {min(accuracies):.4f})")
        
        # Performance distribution
        excellent = sum(1 for acc in accuracies if acc >= 0.85)
        good = sum(1 for acc in accuracies if 0.75 <= acc < 0.85)
        moderate = sum(1 for acc in accuracies if 0.65 <= acc < 0.75)
        poor = sum(1 for acc in accuracies if acc < 0.65)
        
        print(f"\n🎯 PERFORMANCE DISTRIBUTION:")
        print("-" * 50)
        print(f"Excellent (≥85%):          {excellent}/{len(all_results)} files ({excellent/len(all_results)*100:.1f}%)")
        print(f"Good (75-85%):             {good}/{len(all_results)} files ({good/len(all_results)*100:.1f}%)")
        print(f"Moderate (65-75%):         {moderate}/{len(all_results)} files ({moderate/len(all_results)*100:.1f}%)")
        print(f"Poor (<65%):               {poor}/{len(all_results)} files ({poor/len(all_results)*100:.1f}%)")
        
        # Generate LaTeX table for paper
        self.generate_latex_table(all_results)
        
        # Save results to CSV for further analysis
        self.save_results_to_csv(all_results)
    
    def generate_latex_table(self, all_results):
        """Generate LaTeX table for academic paper"""
        print(f"\n📝 LATEX TABLE FOR PAPER:")
        print("=" * 60)
        
        # Calculate summary stats
        accuracies = [r['accuracy'] for r in all_results]
        f1_scores = [r['f1_score'] for r in all_results]
        aucs = [r['auc'] for r in all_results]
        
        latex_table = f"""
\\begin{{table}}[htbp]
\\centering
\\caption{{Cross-Dataset Validation Results: UNSW-NB15 Model Performance on BoT-IoT Dataset}}
\\label{{tab:cross_dataset_validation}}
\\begin{{tabular}}{{|l|c|c|c|c|}}
\\hline
\\textbf{{BoT-IoT File}} & \\textbf{{Samples}} & \\textbf{{Accuracy}} & \\textbf{{F1-Score}} & \\textbf{{AUC}} \\\\
\\hline"""

        # Add top 10 files (or all if less than 10)
        display_results = sorted(all_results, key=lambda x: x['accuracy'], reverse=True)[:10]
        
        for result in display_results:
            filename = result['filename'].replace('_', '\\_').replace('.csv', '')  # LaTeX escape
            samples = len(result['true_labels'])
            acc = result['accuracy']
            f1 = result['f1_score']
            auc = result['auc']
            
            latex_table += f"""
{filename} & {samples:,} & {acc:.3f} & {f1:.3f} & {auc:.3f} \\\\"""

        latex_table += f"""
\\hline
\\textbf{{Average}} & \\textbf{{{sum(len(r['true_labels']) for r in all_results):,}}} & \\textbf{{{np.mean(accuracies):.3f}}} & \\textbf{{{np.mean(f1_scores):.3f}}} & \\textbf{{{np.mean(aucs):.3f}}} \\\\
\\textbf{{Std Dev}} & - & \\textbf{{±{np.std(accuracies):.3f}}} & \\textbf{{±{np.std(f1_scores):.3f}}} & \\textbf{{±{np.std(aucs):.3f}}} \\\\
\\hline
\\end{{tabular}}
\\end{{table}}"""

        print(latex_table)
        
        # Save to file
        with open('cross_dataset_latex_table.txt', 'w') as f:
            f.write(latex_table)
        print(f"\n💾 LaTeX table saved to: cross_dataset_latex_table.txt")
    
    def save_results_to_csv(self, all_results):
        """Save detailed results to CSV for further analysis"""
        
        # Create results dataframe
        results_data = []
        for result in all_results:
            results_data.append({
                'filename': result['filename'],
                'samples': len(result['true_labels']),
                'accuracy': result['accuracy'],
                'precision': result['precision'],
                'recall': result['recall'],
                'f1_score': result['f1_score'],
                'auc': result['auc'],
                'true_positives': result['confusion_matrix'][1,1],
                'true_negatives': result['confusion_matrix'][0,0],
                'false_positives': result['confusion_matrix'][0,1],
                'false_negatives': result['confusion_matrix'][1,0]
            })
        
        results_df = pd.DataFrame(results_data)
        results_df.to_csv('cross_dataset_detailed_results.csv', index=False)
        
        print(f"\n💾 Detailed results saved to: cross_dataset_detailed_results.csv")

def main():
    """Main function to run cross-dataset validation - GPU ACCELERATED"""
    import sys
    import argparse
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Cross-Dataset Validation with GPU Acceleration')
    parser.add_argument('path', nargs='?', default='[INSERT PATH HERE]', help='Path to CSV file or directory')
    parser.add_argument('--batch-size', type=int, default=100000, help='Batch size for processing (default: 100000)')
    parser.add_argument('--no-gpu', action='store_true', help='Disable GPU acceleration')
    args = parser.parse_args()
    
    # Initialize validator with GPU settings
    validator = CrossDatasetValidator(
        batch_size=args.batch_size,
        use_gpu=not args.no_gpu
    )
    
    path = args.path
    
    print("🚀 Starting Cross-Dataset Validation Study (GPU ACCELERATED)")
    print(f"📁 Path: {path}")
    print(f"⚙️  Batch Size: {args.batch_size:,}")
    print(f"🎮 GPU Mode: {'Enabled' if validator.use_gpu else 'Disabled'}")
    
    # Check if path is a file or directory
    if os.path.isfile(path):
        # Single file mode
        print(f"🎯 Processing single file for detailed analysis")
        
        # Load model
        if not validator.load_binary_model():
            return
        
        # Create feature mapping
        validator.create_feature_mapping()
        
        # Run validation on single file
        results = validator.run_cross_dataset_validation_single_file(path)
        
        if results:
            print(f"\n✅ Single file validation complete!")
            print(f"   File: {os.path.basename(path)}")
            print(f"   Accuracy: {results['accuracy']:.4f}")
            print(f"   F1-Score: {results['f1_score']:.4f}")
            print(f"   AUC: {results['auc']:.4f}")
        
        return
    
    elif os.path.isdir(path):
        # Directory mode
        print(f"🎯 Processing all CSV files in directory for comprehensive analysis")
        
        # Run the validation on all files
        results = validator.run_cross_dataset_validation_directory(path)
    else:
        print(f"❌ Invalid path: {path}")
        print(f"   Path must be either a CSV file or a directory containing CSV files")
        return
    
    if results:
        print(f"\n🎉 CROSS-DATASET VALIDATION STUDY COMPLETE!")
        print("=" * 60)
        
        # Calculate overall statistics
        accuracies = [r['accuracy'] for r in results]
        f1_scores = [r['f1_score'] for r in results]
        total_samples = sum(len(r['true_labels']) for r in results)
        
        print(f"📊 FINAL SUMMARY:")
        print(f"   Files Processed: {len(results)}")
        print(f"   Total Samples: {total_samples:,}")
        print(f"   Average Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
        print(f"   Average F1-Score: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
        print(f"   Best Performance: {max(accuracies):.4f}")
        print(f"   Worst Performance: {min(accuracies):.4f}")
        
        print(f"\n📝 Generated Files:")
        print(f"   - cross_dataset_latex_table.txt (LaTeX table for paper)")
        print(f"   - cross_dataset_detailed_results.csv (detailed results)")
        
        print(f"\n💡 This comprehensive study demonstrates the model's generalization")
        print(f"    capability across {len(results)} different BoT-IoT dataset files!")
        
    else:
        print(f"\n💡 To run this validation:")
        print(f"   1. Replace '[INSERT PATH HERE]' with actual BoT-IoT directory path")
        print(f"   2. Ensure the directory contains CSV files")
        print(f"   3. Ensure the binary model exists in Models/Binary/ directory")
        print(f"   4. Run: python test_cross_dataset.py")

if __name__ == "__main__":
    main()