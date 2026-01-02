#!/usr/bin/env python3
"""
Complete execution script for Novel Ensemble ML System
A sophisticated ML approach without deep learning
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from novel_ensemble_ml import NovelEnsembleMLSystem

def check_ml_requirements():
    """Check if required ML packages are installed"""
    required_packages = [
        'pandas', 'numpy', 'scikit-learn', 'matplotlib', 'seaborn'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("Installing missing packages...")
        
        import subprocess
        for package in missing_packages:
            subprocess.run([sys.executable, '-m', 'pip', 'install', package])
        
        print("✅ Packages installed successfully")
    else:
        print("✅ All required packages are available")
    
    return True

def run_complete_experiment(csv_path, test_csv_path=None, classification_type='binary', force_retrain=False, force_retrain_models=None, cache_dir="Models"):
    """Run the complete novel ML experiment with smart model caching"""
    
    print("🚀 NOVEL ENSEMBLE ML NETWORK INTRUSION DETECTION")
    print("=" * 60)
    print("🎯 NOVELTY: Dynamic Feature Engineering + Adaptive Ensemble Learning")
    print(f"🎯 CLASSIFICATION: {classification_type.upper()}")
    print("=" * 60)
    
    # Initialize the system
    system = NovelEnsembleMLSystem(classification_type=classification_type)
    
    # Train the system (with smart caching)
    print("\n🔧 TRAINING PHASE")
    print("-" * 40)
    system.fit(csv_path, force_retrain=force_retrain, force_retrain_models=force_retrain_models, cache_dir=cache_dir)
    
    # QUICK FIX: Override mixing ratio to use weighted average instead of broken meta-learner
    # This bypasses the meta-learner (52% CV accuracy) and uses weighted average (~75-80% accuracy)
    if classification_type == 'multiclass':
        print("\n🔧 APPLYING ENSEMBLE FIX...")
        print("   Setting mixing_ratio = 0.0 (weighted average instead of meta-learner)")
        system.classifier.optimal_mixing_ratio = 0.0
        print("   ✅ Ensemble will now use weighted average of base classifiers")
    
    # Ensemble is automatically saved during training
    
    # Evaluate the system
    print("\n📊 EVALUATION PHASE")
    print("-" * 40)
    
    # Use test set if provided, otherwise use training set (for demo)
    eval_path = test_csv_path if test_csv_path else csv_path
    if not test_csv_path:
        print("⚠️  Using training data for evaluation (demo mode)")
        print("   For proper evaluation, provide separate test dataset")
    
    results = system.evaluate(eval_path)
    
    # Generate visualizations
    print("\n📈 GENERATING VISUALIZATIONS")
    print("-" * 40)
    system.plot_results(results)
    
    # Print summary
    print("\n🎉 EXPERIMENT COMPLETE!")
    print("=" * 40)
    print(f"🎯 Final Accuracy: {results['accuracy']:.4f}")
    print(f"📈 AUC Score: {results['auc_score']:.4f}")
    print("📁 Generated: novel_ensemble_results.png")
    
    return results

def compare_with_baseline(csv_path):
    """Compare novel approach with baseline methods"""
    print("\n🔬 BASELINE COMPARISON")
    print("=" * 40)
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, roc_auc_score
    
    # Load and preprocess data (simple preprocessing)
    df = pd.read_csv(csv_path)
    df = df.fillna(0)
    
    # Simple encoding
    categorical_cols = ['proto', 'service', 'state']
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    
    # Basic features (no engineering)
    target_col = 'label' if 'label' in df.columns else 'attack'
    feature_cols = [col for col in df.columns if col not in [target_col, 'attack_cat', 'stime', 'srcip', 'dstip', 'id']]
    
    X = df[feature_cols].values
    y = df[target_col].values
    
    # Split data - Main split: 70/30 (consistent across codebase)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Baseline models
    from sklearn.linear_model import SGDClassifier
    baselines = {
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
        'SGD': SGDClassifier(loss='log_loss', random_state=42, max_iter=1000, alpha=0.01, penalty='l2', class_weight='balanced')
    }
    
    baseline_results = {}
    
    for name, model in baselines.items():
        print(f"   Training {name}...")
        model.fit(X_train_scaled, y_train)
        
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        
        baseline_results[name] = {'accuracy': accuracy, 'auc': auc}
        print(f"      Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")
    
    return baseline_results

def analyze_novel_components(csv_path):
    """Proper ablation study analyzing the contribution of each feature engineering component"""
    print("\n🔍 FEATURE ENGINEERING ABLATION STUDY")
    print("=" * 50)
    
    from novel_ensemble_ml import DynamicFeatureEngineer, IntelligentFeatureSelector, AdaptiveEnsembleClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    
    # Load data
    df = pd.read_csv(csv_path)
    df = df.fillna(0)
    
    # Basic preprocessing
    categorical_cols = ['proto', 'service', 'state']
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    
    target_col = 'label' if 'label' in df.columns else 'attack'
    y = df[target_col].values
    
    # Get original features
    original_cols = [col for col in df.columns if col not in [target_col, 'attack_cat', 'stime', 'srcip', 'dstip', 'id']]
    
    # Create feature engineer instance
    fe = DynamicFeatureEngineer()
    
    # Helper function to evaluate a feature set
    def evaluate_features(X, description):
        # Main split: 70/30 (consistent across codebase)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        rf.fit(X_train_scaled, y_train)
        acc = accuracy_score(y_test, rf.predict(X_test_scaled))
        print(f"   {description}: {acc:.4f}")
        return acc
    
    results = {}
    
    # Test 1: Baseline - Original features only
    print("1️⃣  BASELINE: Original features only")
    X_original = df[original_cols].values
    acc_original = evaluate_features(X_original, "Original features")
    results['baseline'] = acc_original
    
    # Test 2: + Temporal features
    print("\n2️⃣  + TEMPORAL FEATURES")
    df_temporal = df.copy()
    df_temporal = fe.create_temporal_features(df_temporal)
    
    # Get temporal feature names
    temporal_features = ['hour', 'day_of_week', 'is_weekend', 'is_business_hours', 'is_night']
    available_temporal = [f for f in temporal_features if f in df_temporal.columns]
    
    temporal_cols = original_cols + available_temporal
    X_temporal = df_temporal[temporal_cols].fillna(0).values
    acc_temporal = evaluate_features(X_temporal, f"+ Temporal features ({len(available_temporal)} added)")
    results['temporal'] = acc_temporal
    print(f"   Improvement over baseline: +{acc_temporal - acc_original:.4f}")
    
    # Test 3: + Statistical features
    print("\n3️⃣  + STATISTICAL FEATURES")
    df_statistical = df.copy()
    df_statistical = fe.create_statistical_features(df_statistical)
    
    # Get statistical feature names
    statistical_features = ['total_bytes', 'byte_ratio', 'byte_imbalance', 'log_total_bytes', 
                          'total_packets', 'packet_ratio', 'avg_packet_size', 
                          'log_duration', 'is_short_connection', 'is_long_connection',
                          'throughput', 'log_throughput', 'src_is_wellknown', 'dst_is_wellknown',
                          'port_difference', 'dst_is_common_service']
    available_statistical = [f for f in statistical_features if f in df_statistical.columns]
    
    statistical_cols = original_cols + available_statistical
    X_statistical = df_statistical[statistical_cols].fillna(0).values
    acc_statistical = evaluate_features(X_statistical, f"+ Statistical features ({len(available_statistical)} added)")
    results['statistical'] = acc_statistical
    print(f"   Improvement over baseline: +{acc_statistical - acc_original:.4f}")
    
    # Test 4: + Interaction features
    print("\n4️⃣  + INTERACTION FEATURES")
    df_interaction = df.copy()
    df_interaction = fe.create_interaction_features(df_interaction)
    
    # Get interaction feature names
    interaction_features = ['proto_service_encoded', 'state_proto_encoded']
    available_interaction = [f for f in interaction_features if f in df_interaction.columns]
    
    interaction_cols = original_cols + available_interaction
    X_interaction = df_interaction[interaction_cols].fillna(0).values
    acc_interaction = evaluate_features(X_interaction, f"+ Interaction features ({len(available_interaction)} added)")
    results['interaction'] = acc_interaction
    print(f"   Improvement over baseline: +{acc_interaction - acc_original:.4f}")
    
    # Test 5: + Network Behavior features
    print("\n5️⃣  + NETWORK BEHAVIOR FEATURES")
    df_behavior = df.copy()
    df_behavior = fe.create_network_behavior_features(df_behavior)
    
    # Get behavior feature names
    behavior_features = ['total_loss', 'loss_ratio', 'has_loss', 
                        'total_jitter', 'jitter_ratio', 'high_jitter',
                        'window_ratio', 'min_window', 'max_window']
    available_behavior = [f for f in behavior_features if f in df_behavior.columns]
    
    behavior_cols = original_cols + available_behavior
    X_behavior = df_behavior[behavior_cols].fillna(0).values
    acc_behavior = evaluate_features(X_behavior, f"+ Behavior features ({len(available_behavior)} added)")
    results['behavior'] = acc_behavior
    print(f"   Improvement over baseline: +{acc_behavior - acc_original:.4f}")
    
    # Test 6: ALL feature engineering combined
    print("\n6️⃣  ALL FEATURE ENGINEERING COMBINED")
    fe_full = DynamicFeatureEngineer()
    df_all = fe_full.fit_transform(df)
    
    all_cols = [col for col in df_all.columns if col not in [target_col, 'attack_cat', 'stime', 'srcip', 'dstip', 'id']]
    X_all = df_all[all_cols].fillna(0).values
    acc_all = evaluate_features(X_all, f"All engineered features ({len(all_cols)} total)")
    results['all_features'] = acc_all
    print(f"   Improvement over baseline: +{acc_all - acc_original:.4f}")
    
    # Test 7: + Feature Selection
    print("\n7️⃣  + INTELLIGENT FEATURE SELECTION")
    from sklearn.feature_selection import SelectKBest, f_classif
    
    # Main split: 70/30 (consistent across codebase)
    X_train, X_test, y_train, y_test = train_test_split(X_all, y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Use same feature count as main pipeline
    standardized_feature_count = 42
    selector = SelectKBest(f_classif, k=min(standardized_feature_count, X_train_scaled.shape[1]))
    X_selected = selector.fit_transform(X_train_scaled, y_train)
    X_test_selected = selector.transform(X_test_scaled)
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf.fit(X_selected, y_train)
    acc_selected = accuracy_score(y_test, rf.predict(X_test_selected))
    print(f"   + Feature selection ({X_selected.shape[1]} features): {acc_selected:.4f}")
    results['feature_selection'] = acc_selected
    print(f"   Improvement over baseline: +{acc_selected - acc_original:.4f}")
    
    # Test 8: + Adaptive Ensemble
    print("\n8️⃣  + ADAPTIVE ENSEMBLE")
    ensemble = AdaptiveEnsembleClassifier()
    ensemble.fit(X_selected, y_train)
    acc_ensemble = accuracy_score(y_test, ensemble.predict(X_test_selected))
    print(f"   + Adaptive ensemble: {acc_ensemble:.4f}")
    results['adaptive_ensemble'] = acc_ensemble
    print(f"   Total improvement: +{acc_ensemble - acc_original:.4f}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Novel Ensemble ML for Network Intrusion Detection')
    parser.add_argument('--dataset', type=str, default=None,
                       help='Path to training dataset CSV')
    parser.add_argument('--test-dataset', type=str, default=None,
                       help='Path to test dataset CSV')

    parser.add_argument('--compare-baseline', action='store_true',
                       help='Compare with baseline methods')
    parser.add_argument('--analyze-components', action='store_true',
                       help='Analyze contribution of novel components')
    parser.add_argument('--multiclass', action='store_true',
                       help='Perform multi-class attack classification instead of binary')
    parser.add_argument('--force-retrain', action='store_true',
                       help='Force retraining of entire ensemble')
    parser.add_argument('--force-retrain-models', nargs='*', 
                       help='Force retrain specific models (e.g., rf gb et)')
    parser.add_argument('--cache-dir', default='Models',
                       help='Directory for model caching (default: Models)')
    parser.add_argument('--no-cache', action='store_true',
                       help='Disable caching entirely (always train fresh)')
    
    args = parser.parse_args()
    
    # Check requirements
    if not check_ml_requirements():
        sys.exit(1)
    
    # Determine dataset
    if args.dataset is None:
        # Use balanced dataset if available, otherwise require user to specify
        if os.path.exists('UNSW_balanced_train.csv'):
            csv_path = 'UNSW_balanced_train.csv'
            print("📊 Using balanced UNSW dataset: UNSW_balanced_train.csv")
        else:
            print("❌ No dataset specified and UNSW_balanced_train.csv not found")
            print("   Run: python create_balanced_split.py first")
            sys.exit(1)
    else:
        csv_path = args.dataset
        if not os.path.exists(csv_path):
            print(f"❌ Dataset not found: {csv_path}")
            sys.exit(1)
    
    # Determine classification type
    classification_type = 'multiclass' if args.multiclass else 'binary'
    
    # Configure caching
    force_retrain = args.force_retrain or args.no_cache
    force_retrain_models = args.force_retrain_models
    cache_dir = args.cache_dir
    
    if args.no_cache:
        print("🔧 Caching disabled - training fresh models")
    elif force_retrain:
        print("🔧 Force retrain enabled - clearing cache")
    elif force_retrain_models:
        print(f"🔧 Force retraining models: {force_retrain_models}")
    else:
        print("🔧 Smart caching enabled")
        print(f"📁 Cache directory: {cache_dir}")
        
        # Show cache status
        cache_subdir = "Binary" if classification_type == "binary" else "Multiclass"
        model_cache_dir = os.path.join(cache_dir, cache_subdir)
        if os.path.exists(model_cache_dir):
            cached_models = [f for f in os.listdir(model_cache_dir) if f.endswith('.pkl')]
            if cached_models:
                print(f"📦 Found {len(cached_models)} cached models: {', '.join(cached_models)}")
            else:
                print("📦 No cached models found - will train fresh")
        else:
            print("📦 Cache directory doesn't exist - will train fresh")
        
        print("💡 Use --force-retrain-models rf gb to retrain specific models")
        print("💡 Use --force-retrain to clear all cache and retrain everything")
    
    results = run_complete_experiment(csv_path, args.test_dataset, classification_type, 
                                    force_retrain, force_retrain_models, cache_dir)
    
    # Optional analyses
    if args.compare_baseline:
        try:
            print("\n🔄 Running baseline comparison...")
            baseline_results = compare_with_baseline(csv_path)
        except Exception as e:
            print(f"\n⚠️  Baseline comparison failed: {e}")
            baseline_results = {}
        
        if baseline_results:
            print("\n📊 COMPARISON SUMMARY")
            print("-" * 30)
            print("Baseline Methods:")
            for method, metrics in baseline_results.items():
                print(f"   {method}: Acc={metrics['accuracy']:.4f}, AUC={metrics['auc']:.4f}")
            
            print(f"\nNovel Ensemble ML:")
            print(f"   Accuracy: {results['accuracy']:.4f}, AUC: {results['auc_score']:.4f}")
            
            # Calculate improvements
            try:
                best_baseline_acc = max([m['accuracy'] for m in baseline_results.values()])
                best_baseline_auc = max([m['auc'] for m in baseline_results.values()])
                
                print(f"\n🎯 IMPROVEMENTS:")
                print(f"   Accuracy: +{results['accuracy'] - best_baseline_acc:.4f}")
                print(f"   AUC: +{results['auc_score'] - best_baseline_auc:.4f}")
            except Exception as e:
                print(f"   ⚠️  Could not calculate improvements: {e}")
        else:
            print("\n⚠️  No baseline results to compare")
    
    if args.analyze_components:
        component_results = analyze_novel_components(csv_path)
        
        print("\n🔍 ABLATION STUDY SUMMARY:")
        print("=" * 50)
        baseline = component_results['baseline']
        print(f"   Baseline (Original):     {baseline:.4f}")
        print(f"   + Temporal Features:     {component_results['temporal']:.4f} (+{component_results['temporal'] - baseline:.4f})")
        print(f"   + Statistical Features:  {component_results['statistical']:.4f} (+{component_results['statistical'] - baseline:.4f})")
        print(f"   + Interaction Features:  {component_results['interaction']:.4f} (+{component_results['interaction'] - baseline:.4f})")
        print(f"   + Behavior Features:     {component_results['behavior']:.4f} (+{component_results['behavior'] - baseline:.4f})")
        print(f"   All Features Combined:   {component_results['all_features']:.4f} (+{component_results['all_features'] - baseline:.4f})")
        print(f"   + Feature Selection:     {component_results['feature_selection']:.4f} (+{component_results['feature_selection'] - baseline:.4f})")
        print(f"   + Adaptive Ensemble:     {component_results['adaptive_ensemble']:.4f} (+{component_results['adaptive_ensemble'] - baseline:.4f})")
        
        print(f"\n📊 FEATURE CATEGORY RANKING:")
        print("-" * 30)
        categories = [
            ('Temporal', component_results['temporal'] - baseline),
            ('Statistical', component_results['statistical'] - baseline),
            ('Interaction', component_results['interaction'] - baseline),
            ('Behavior', component_results['behavior'] - baseline)
        ]
        categories.sort(key=lambda x: x[1], reverse=True)
        
        for i, (name, improvement) in enumerate(categories, 1):
            print(f"   {i}. {name:<12}: +{improvement:.4f}")
        
        print(f"\n💡 INSIGHTS:")
        print("-" * 30)
        best_category, best_improvement = categories[0]
        print(f"   🏆 Most valuable: {best_category} features (+{best_improvement:.4f})")
        
        total_individual = sum(cat[1] for cat in categories)
        combined_improvement = component_results['all_features'] - baseline
        synergy = combined_improvement - total_individual
        
        if synergy > 0.001:
            print(f"   🤝 Positive synergy: +{synergy:.4f} (features work better together)")
        elif synergy < -0.001:
            print(f"   ⚠️  Negative synergy: {synergy:.4f} (feature interference detected)")
        else:
            print(f"   ➡️  Additive effect: Features contribute independently")
    
    print("\n" + "=" * 60)
    print("🎉 NOVEL ML EXPERIMENT COMPLETE!")
    print("=" * 60)
    print("📁 Generated files:")
    print("   - novel_ensemble_results.png (comprehensive results)")
    print("   - trained_novel_ensemble_model.pkl (trained model)")
    
    print("\n💡 Key Novelties Demonstrated:")
    print("   ✅ Dynamic Feature Engineering (temporal, statistical, interaction)")
    print("   ✅ Adaptive Ensemble Learning (data-characteristic-based weighting)")
    print("   ✅ Intelligent Feature Selection (multi-method ensemble)")
    print("   ✅ Attack-Type Specialist Models")
    
    print("\n🚀 To run with your dataset:")
    print("   python run_novel_ml.py --dataset your_data.csv --compare-baseline --analyze-components")

if __name__ == "__main__":
    main()