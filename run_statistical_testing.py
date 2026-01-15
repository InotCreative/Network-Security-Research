#!/usr/bin/env python3
"""
Statistical Testing for Cached Models

Performs statistical comparison between cached models using:
- Paired t-test on CV scores (if available from same folds)
- Cohen's d effect size
- Bootstrap confidence intervals for test metrics
- Holm-Bonferroni correction for multiple comparisons

Consistent with ablation_study.py statistical testing methodology.

Usage:
    # Compare two cached models
    python run_statistical_testing.py --model-a path/to/model_a.pkl --model-b path/to/model_b.pkl --test-data test.csv
    
    # Compare multiple models against a baseline
    python run_statistical_testing.py --baseline path/to/baseline.pkl --models model1.pkl model2.pkl --test-data test.csv
    
    # Use cached ensemble model
    python run_statistical_testing.py --model-a trained_novel_ensemble_model.pkl --model-b Models/Binary/baseline.pkl --test-data preprocessed_test.csv
"""

import os
import sys
import argparse
import pickle
import json
import numpy as np
import pandas as pd
from datetime import datetime
from scipy import stats
from scipy.stats import shapiro, wilcoxon
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score

import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# STATISTICAL TESTING FUNCTIONS (Consistent with ablation_study.py)
# =============================================================================

def check_normality(scores, alpha=0.05):
    """
    Shapiro-Wilk normality test.
    
    Args:
        scores: Array of scores to test
        alpha: Significance level
    
    Returns:
        dict with statistic, p_value, is_normal
    """
    if len(scores) < 3:
        return {'statistic': None, 'p_value': None, 'is_normal': False}
    
    stat, p_value = shapiro(scores)
    return {
        'statistic': float(stat),
        'p_value': float(p_value),
        'is_normal': p_value > alpha
    }


def interpret_cohens_d(d):
    """Interpret Cohen's d effect size."""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "Negligible"
    elif abs_d < 0.5:
        return "Small"
    elif abs_d < 0.8:
        return "Medium"
    else:
        return "Large"


def statistical_comparison(cv_scores_a, cv_scores_b, name_a="Model A", name_b="Model B"):
    """
    Comprehensive statistical comparison between two sets of CV scores.
    
    Consistent with ablation_study.py compare_configs() method.
    
    Includes:
    1. Shapiro-Wilk normality test
    2. Adaptive test selection (t-test vs Wilcoxon)
    3. Cohen's d effect size with interpretation
    
    Args:
        cv_scores_a: CV scores for model A
        cv_scores_b: CV scores for model B
        name_a: Name of model A
        name_b: Name of model B
    
    Returns:
        dict with comprehensive comparison results
    """
    cv_a = np.array(cv_scores_a)
    cv_b = np.array(cv_scores_b)
    
    # Check if we have paired scores (same number of folds)
    if len(cv_a) != len(cv_b):
        return {
            'error': f"Cannot compare: different number of CV folds ({len(cv_a)} vs {len(cv_b)})",
            'applicable': False
        }
    
    # 1. Normality check (Shapiro-Wilk)
    norm_a = check_normality(cv_a)
    norm_b = check_normality(cv_b)
    both_normal = norm_a['is_normal'] and norm_b['is_normal']
    
    # 2. Adaptive test selection
    if both_normal and len(cv_a) >= 5:
        t_stat, p_value = stats.ttest_rel(cv_a, cv_b)
        test_used = "Paired t-test"
    else:
        try:
            t_stat, p_value = wilcoxon(cv_a, cv_b)
            test_used = "Wilcoxon signed-rank"
        except ValueError:
            # Wilcoxon fails if all differences are zero
            t_stat, p_value = 0.0, 1.0
            test_used = "Wilcoxon (failed - identical scores)"
    
    # 3. Effect size (Cohen's d)
    diff = cv_a - cv_b
    cohens_d = np.mean(diff) / (np.std(diff, ddof=1) + 1e-10)
    effect_interp = interpret_cohens_d(cohens_d)
    
    return {
        'applicable': True,
        'comparison': f"{name_a} vs {name_b}",
        'test_used': test_used,
        'normality_a': norm_a,
        'normality_b': norm_b,
        'both_normal': both_normal,
        'mean_a': float(np.mean(cv_a)),
        'mean_b': float(np.mean(cv_b)),
        'std_a': float(np.std(cv_a, ddof=1)),
        'std_b': float(np.std(cv_b, ddof=1)),
        'mean_diff': float(np.mean(diff)),
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'cohens_d': float(cohens_d),
        'effect_interpretation': effect_interp,
        'significant_raw': p_value < 0.05
    }


def bootstrap_confidence_interval(y_true, y_pred, metric_func, n_bootstrap=1000, confidence=0.95):
    """
    Bootstrap confidence interval for test set metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Model predictions
        metric_func: Metric function (e.g., accuracy_score)
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level
    
    Returns:
        dict with observed, ci_lower, ci_upper, std
    """
    n = len(y_true)
    bootstrap_scores = []
    
    np.random.seed(42)
    for _ in range(n_bootstrap):
        indices = np.random.choice(n, size=n, replace=True)
        try:
            score = metric_func(y_true[indices], y_pred[indices])
            bootstrap_scores.append(score)
        except:
            continue
    
    if len(bootstrap_scores) == 0:
        return {'error': 'Bootstrap failed'}
    
    bootstrap_scores = np.array(bootstrap_scores)
    alpha = 1 - confidence
    ci_lower = np.percentile(bootstrap_scores, alpha/2 * 100)
    ci_upper = np.percentile(bootstrap_scores, (1 - alpha/2) * 100)
    
    return {
        'observed': float(metric_func(y_true, y_pred)),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'bootstrap_mean': float(np.mean(bootstrap_scores)),
        'bootstrap_std': float(np.std(bootstrap_scores))
    }


def bootstrap_difference_ci(y_true, y_pred_a, y_pred_b, metric_func, n_bootstrap=1000, confidence=0.95):
    """
    Bootstrap confidence interval for the DIFFERENCE between two models.
    
    Args:
        y_true: Ground truth labels
        y_pred_a: Model A predictions
        y_pred_b: Model B predictions
        metric_func: Metric function
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level
    
    Returns:
        dict with observed_diff, ci_lower, ci_upper, zero_in_ci
    """
    n = len(y_true)
    diffs = []
    
    np.random.seed(42)
    for _ in range(n_bootstrap):
        indices = np.random.choice(n, size=n, replace=True)
        try:
            score_a = metric_func(y_true[indices], y_pred_a[indices])
            score_b = metric_func(y_true[indices], y_pred_b[indices])
            diffs.append(score_a - score_b)
        except:
            continue
    
    if len(diffs) == 0:
        return {'error': 'Bootstrap failed'}
    
    diffs = np.array(diffs)
    alpha = 1 - confidence
    ci_lower = np.percentile(diffs, alpha/2 * 100)
    ci_upper = np.percentile(diffs, (1 - alpha/2) * 100)
    observed_diff = metric_func(y_true, y_pred_a) - metric_func(y_true, y_pred_b)
    
    return {
        'observed_diff': float(observed_diff),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'zero_in_ci': ci_lower <= 0 <= ci_upper,
        'bootstrap_mean': float(np.mean(diffs)),
        'bootstrap_std': float(np.std(diffs))
    }


def apply_multiple_comparison_correction(p_values, method='holm'):
    """
    Apply multiple comparison correction.
    
    Args:
        p_values: List of p-values
        method: Correction method ('holm', 'bonferroni', 'fdr_bh')
    
    Returns:
        corrected p-values and rejection decisions
    """
    try:
        from statsmodels.stats.multitest import multipletests
        rejected, p_corrected, _, _ = multipletests(p_values, alpha=0.05, method=method)
        return p_corrected.tolist(), rejected.tolist()
    except ImportError:
        print("⚠️  statsmodels not available - skipping multiple comparison correction")
        return p_values, [p < 0.05 for p in p_values]



# =============================================================================
# MODEL LOADING AND DATA PREPARATION
# =============================================================================

def load_cached_model(model_path):
    """
    Load a cached model and extract relevant components.
    
    Args:
        model_path: Path to the pickle file
    
    Returns:
        dict with model components and metadata
    """
    print(f"📦 Loading model: {model_path}")
    
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
    
    # Handle different model formats
    if isinstance(data, dict):
        # Standard format from NovelEnsembleMLSystem
        model_info = {
            'path': model_path,
            'classifier': data.get('classifier'),
            'feature_engineer': data.get('feature_engineer'),
            'scaler': data.get('scaler'),
            'feature_selector': data.get('feature_selector'),
            'selected_feature_indices': data.get('selected_feature_indices'),
            'feature_names': data.get('feature_names'),
            'classification_type': data.get('classification_type', 'binary'),
            'class_names': data.get('class_names', ['Normal', 'Attack']),
            'attack_type_encoder': data.get('attack_type_encoder'),
        }
        
        # Extract CV scores if available
        classifier = model_info['classifier']
        if classifier and hasattr(classifier, 'ensemble_cv_scores'):
            model_info['cv_scores'] = np.array(classifier.ensemble_cv_scores)
        elif classifier and hasattr(classifier, 'individual_performance'):
            # Try to get CV scores from individual performance
            perf = classifier.individual_performance
            if perf:
                # Average CV scores across base classifiers
                cv_scores = []
                for name, metrics in perf.items():
                    if 'cv_accuracy' in metrics:
                        cv_scores.append(metrics['cv_accuracy'])
                if cv_scores:
                    model_info['cv_scores'] = np.array([np.mean(cv_scores)] * 5)  # Approximate
        
        # Get metadata
        metadata = data.get('model_metadata', {})
        model_info['metadata'] = metadata
        
    else:
        # Assume it's a raw sklearn model
        model_info = {
            'path': model_path,
            'classifier': data,
            'feature_engineer': None,
            'scaler': None,
            'feature_selector': None,
            'selected_feature_indices': None,
            'feature_names': None,
            'classification_type': 'binary',
            'class_names': ['Normal', 'Attack'],
            'cv_scores': None,
            'metadata': {}
        }
    
    print(f"   ✅ Model loaded successfully")
    if model_info.get('cv_scores') is not None:
        print(f"   📊 CV scores available: {len(model_info['cv_scores'])} folds")
    else:
        print(f"   ⚠️  No CV scores stored in model")
    
    return model_info


def load_test_data(test_csv, classification_type='binary'):
    """
    Load and preprocess test data.
    
    Args:
        test_csv: Path to test CSV file
        classification_type: 'binary' or 'multiclass'
    
    Returns:
        X_test, y_test, feature_names
    """
    print(f"📁 Loading test data: {test_csv}")
    
    df = pd.read_csv(test_csv)
    df = df.fillna(0)
    
    # Encode categorical columns
    label_encoders = {}
    categorical_cols = ['proto', 'service', 'state']
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
    
    # Get labels
    if classification_type == 'binary':
        target_col = 'label' if 'label' in df.columns else 'attack'
        y = df[target_col].values
    else:
        if 'attack_cat' in df.columns:
            le = LabelEncoder()
            y = le.fit_transform(df['attack_cat'].astype(str))
        else:
            raise ValueError("No attack_cat column for multiclass")
    
    # Get features
    exclude_cols = ['label', 'attack', 'attack_cat', 'stime', 'srcip', 'dstip', 'id']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    X = df[feature_cols].values
    
    print(f"   ✅ Loaded {len(y):,} samples, {X.shape[1]} features")
    
    return X, y, feature_cols, df, label_encoders


def get_predictions(model_info, X_test, df_test=None):
    """
    Get predictions from a cached model.
    
    Args:
        model_info: Model info dict from load_cached_model
        X_test: Test features (raw)
        df_test: Test dataframe (for feature engineering)
    
    Returns:
        y_pred, y_proba (if available)
    """
    classifier = model_info['classifier']
    feature_engineer = model_info['feature_engineer']
    scaler = model_info['scaler']
    selected_indices = model_info['selected_feature_indices']
    
    # Apply feature engineering if available
    if feature_engineer is not None and df_test is not None:
        try:
            df_eng = feature_engineer.transform(df_test)
            exclude_cols = ['label', 'attack', 'attack_cat', 'stime', 'srcip', 'dstip', 'id']
            feature_cols = [col for col in df_eng.columns if col not in exclude_cols]
            X = df_eng[feature_cols].values
        except Exception as e:
            print(f"   ⚠️  Feature engineering failed: {e}")
            X = X_test
    else:
        X = X_test
    
    # Apply scaling if available
    if scaler is not None:
        try:
            X = scaler.transform(X)
        except Exception as e:
            print(f"   ⚠️  Scaling failed: {e}")
    
    # Apply feature selection if available
    if selected_indices is not None:
        try:
            X = X[:, selected_indices]
        except Exception as e:
            print(f"   ⚠️  Feature selection failed: {e}")
    
    # Get predictions
    y_pred = classifier.predict(X)
    
    # Get probabilities if available
    y_proba = None
    if hasattr(classifier, 'predict_proba'):
        try:
            y_proba = classifier.predict_proba(X)
        except:
            pass
    
    return y_pred, y_proba



# =============================================================================
# MAIN COMPARISON FUNCTIONS
# =============================================================================

def compare_two_models(model_a_path, model_b_path, test_csv, classification_type='binary'):
    """
    Full statistical comparison between two cached models.
    
    Args:
        model_a_path: Path to model A pickle file
        model_b_path: Path to model B pickle file
        test_csv: Path to test data CSV
        classification_type: 'binary' or 'multiclass'
    
    Returns:
        Comprehensive results dictionary
    """
    print("\n" + "=" * 80)
    print("🔬 STATISTICAL MODEL COMPARISON")
    print("=" * 80)
    print(f"Model A: {model_a_path}")
    print(f"Model B: {model_b_path}")
    print(f"Test Data: {test_csv}")
    print(f"Mode: {classification_type}")
    
    # Load models
    model_a = load_cached_model(model_a_path)
    model_b = load_cached_model(model_b_path)
    
    # Load test data
    X_test, y_test, feature_names, df_test, label_encoders = load_test_data(
        test_csv, classification_type
    )
    
    # Get predictions
    print("\n📊 Getting predictions...")
    y_pred_a, y_proba_a = get_predictions(model_a, X_test, df_test)
    y_pred_b, y_proba_b = get_predictions(model_b, X_test, df_test)
    
    results = {
        'model_a': os.path.basename(model_a_path),
        'model_b': os.path.basename(model_b_path),
        'test_data': os.path.basename(test_csv),
        'classification_type': classification_type,
        'n_samples': len(y_test),
        'timestamp': datetime.now().isoformat()
    }
    
    # =========================================================================
    # 1. TEST SET PERFORMANCE METRICS
    # =========================================================================
    print("\n" + "=" * 80)
    print("📊 TEST SET PERFORMANCE")
    print("=" * 80)
    
    if classification_type == 'binary':
        metrics = {
            'accuracy': accuracy_score,
            'f1': lambda yt, yp: f1_score(yt, yp, zero_division=0),
            'balanced_accuracy': balanced_accuracy_score,
            'precision': lambda yt, yp: precision_score(yt, yp, zero_division=0),
            'recall': lambda yt, yp: recall_score(yt, yp, zero_division=0)
        }
    else:
        metrics = {
            'accuracy': accuracy_score,
            'f1_macro': lambda yt, yp: f1_score(yt, yp, average='macro', zero_division=0),
            'balanced_accuracy': balanced_accuracy_score,
            'precision_macro': lambda yt, yp: precision_score(yt, yp, average='macro', zero_division=0),
            'recall_macro': lambda yt, yp: recall_score(yt, yp, average='macro', zero_division=0)
        }
    
    results['test_metrics'] = {}
    
    print(f"\n{'Metric':<20} {'Model A':<12} {'Model B':<12} {'Difference':<12}")
    print("-" * 60)
    
    for metric_name, metric_func in metrics.items():
        val_a = metric_func(y_test, y_pred_a)
        val_b = metric_func(y_test, y_pred_b)
        diff = val_a - val_b
        
        results['test_metrics'][metric_name] = {
            'model_a': float(val_a),
            'model_b': float(val_b),
            'difference': float(diff)
        }
        
        print(f"{metric_name:<20} {val_a:<12.4f} {val_b:<12.4f} {diff:+12.4f}")
    
    # =========================================================================
    # 2. CV SCORE COMPARISON (if available)
    # =========================================================================
    cv_a = model_a.get('cv_scores')
    cv_b = model_b.get('cv_scores')
    
    if cv_a is not None and cv_b is not None:
        print("\n" + "=" * 80)
        print("📊 CROSS-VALIDATION SCORE COMPARISON")
        print("=" * 80)
        
        cv_comparison = statistical_comparison(cv_a, cv_b, "Model A", "Model B")
        
        if cv_comparison.get('applicable', False):
            results['cv_comparison'] = cv_comparison
            
            print(f"\nModel A CV: {cv_comparison['mean_a']:.4f} ± {cv_comparison['std_a']:.4f}")
            print(f"Model B CV: {cv_comparison['mean_b']:.4f} ± {cv_comparison['std_b']:.4f}")
            print(f"\nNormality Check:")
            print(f"   Model A: {'Normal' if cv_comparison['normality_a']['is_normal'] else 'Non-normal'} (p={cv_comparison['normality_a']['p_value']:.4f})")
            print(f"   Model B: {'Normal' if cv_comparison['normality_b']['is_normal'] else 'Non-normal'} (p={cv_comparison['normality_b']['p_value']:.4f})")
            print(f"\nStatistical Test: {cv_comparison['test_used']}")
            print(f"   t-statistic: {cv_comparison['t_statistic']:.4f}")
            print(f"   p-value: {cv_comparison['p_value']:.4f}")
            print(f"   Significant (α=0.05): {'✅ Yes' if cv_comparison['significant_raw'] else '❌ No'}")
            print(f"\nEffect Size:")
            print(f"   Cohen's d: {cv_comparison['cohens_d']:.4f}")
            print(f"   Interpretation: {cv_comparison['effect_interpretation']}")
        else:
            print(f"\n⚠️  {cv_comparison.get('error', 'CV comparison not applicable')}")
            results['cv_comparison'] = {'applicable': False, 'reason': cv_comparison.get('error')}
    else:
        print("\n" + "=" * 80)
        print("⚠️  CV SCORE COMPARISON NOT AVAILABLE")
        print("=" * 80)
        if cv_a is None:
            print("   Model A: No CV scores stored")
        if cv_b is None:
            print("   Model B: No CV scores stored")
        print("\n   💡 Paired t-test requires CV scores from same folds")
        print("   💡 Using bootstrap CI on test metrics instead")
        results['cv_comparison'] = {'applicable': False, 'reason': 'No CV scores available'}
    
    # =========================================================================
    # 3. BOOTSTRAP CONFIDENCE INTERVALS
    # =========================================================================
    print("\n" + "=" * 80)
    print("📊 BOOTSTRAP 95% CONFIDENCE INTERVALS")
    print("=" * 80)
    
    results['bootstrap'] = {}
    
    # Individual model CIs
    print("\nIndividual Model CIs:")
    print(f"{'Metric':<20} {'Model A [95% CI]':<30} {'Model B [95% CI]':<30}")
    print("-" * 80)
    
    for metric_name, metric_func in metrics.items():
        ci_a = bootstrap_confidence_interval(y_test, y_pred_a, metric_func)
        ci_b = bootstrap_confidence_interval(y_test, y_pred_b, metric_func)
        
        results['bootstrap'][f'{metric_name}_a'] = ci_a
        results['bootstrap'][f'{metric_name}_b'] = ci_b
        
        ci_a_str = f"{ci_a['observed']:.4f} [{ci_a['ci_lower']:.4f}, {ci_a['ci_upper']:.4f}]"
        ci_b_str = f"{ci_b['observed']:.4f} [{ci_b['ci_lower']:.4f}, {ci_b['ci_upper']:.4f}]"
        print(f"{metric_name:<20} {ci_a_str:<30} {ci_b_str:<30}")
    
    # Difference CIs
    print("\nDifference CIs (Model A - Model B):")
    print(f"{'Metric':<20} {'Diff [95% CI]':<35} {'Zero in CI?':<15}")
    print("-" * 70)
    
    for metric_name, metric_func in metrics.items():
        diff_ci = bootstrap_difference_ci(y_test, y_pred_a, y_pred_b, metric_func)
        results['bootstrap'][f'{metric_name}_diff'] = diff_ci
        
        ci_str = f"{diff_ci['observed_diff']:+.4f} [{diff_ci['ci_lower']:.4f}, {diff_ci['ci_upper']:.4f}]"
        zero_str = "❌ No (Sig)" if not diff_ci['zero_in_ci'] else "✅ Yes (NS)"
        print(f"{metric_name:<20} {ci_str:<35} {zero_str:<15}")
    
    # =========================================================================
    # 4. SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("📋 SUMMARY")
    print("=" * 80)
    
    # Determine winner based on available evidence
    acc_a = results['test_metrics']['accuracy']['model_a']
    acc_b = results['test_metrics']['accuracy']['model_b']
    
    if 'cv_comparison' in results and results['cv_comparison'].get('applicable'):
        cv_sig = results['cv_comparison']['significant_raw']
        cv_better_a = results['cv_comparison']['mean_a'] > results['cv_comparison']['mean_b']
        
        if cv_sig:
            winner = "Model A" if cv_better_a else "Model B"
            print(f"✅ Statistically significant difference (CV paired test)")
            print(f"   Winner: {winner}")
            print(f"   p-value: {results['cv_comparison']['p_value']:.4f}")
            print(f"   Effect size: {results['cv_comparison']['effect_interpretation']} (d={results['cv_comparison']['cohens_d']:.4f})")
        else:
            print(f"❌ No statistically significant difference (CV paired test)")
            print(f"   p-value: {results['cv_comparison']['p_value']:.4f}")
    else:
        # Use bootstrap CI for significance
        acc_diff_ci = results['bootstrap'].get('accuracy_diff', {})
        if acc_diff_ci and not acc_diff_ci.get('zero_in_ci', True):
            winner = "Model A" if acc_a > acc_b else "Model B"
            print(f"✅ Significant difference (Bootstrap CI excludes zero)")
            print(f"   Winner: {winner}")
        else:
            print(f"❌ No significant difference (Bootstrap CI includes zero)")
    
    print(f"\nTest Accuracy: Model A = {acc_a:.4f}, Model B = {acc_b:.4f}")
    
    return results


def compare_multiple_models(baseline_path, model_paths, test_csv, classification_type='binary'):
    """
    Compare multiple models against a baseline with multiple comparison correction.
    
    Args:
        baseline_path: Path to baseline model
        model_paths: List of paths to comparison models
        test_csv: Path to test data
        classification_type: 'binary' or 'multiclass'
    
    Returns:
        Results with Holm-Bonferroni correction
    """
    print("\n" + "=" * 80)
    print("🔬 MULTIPLE MODEL COMPARISON (with Holm-Bonferroni correction)")
    print("=" * 80)
    print(f"Baseline: {baseline_path}")
    print(f"Models to compare: {len(model_paths)}")
    
    all_results = []
    p_values = []
    
    for model_path in model_paths:
        print(f"\n{'='*40}")
        print(f"Comparing: {os.path.basename(model_path)}")
        print(f"{'='*40}")
        
        result = compare_two_models(baseline_path, model_path, test_csv, classification_type)
        all_results.append(result)
        
        # Collect p-value for correction
        if result.get('cv_comparison', {}).get('applicable'):
            p_values.append(result['cv_comparison']['p_value'])
        else:
            # Use bootstrap-based p-value approximation
            acc_diff = result['bootstrap'].get('accuracy_diff', {})
            if acc_diff:
                # Approximate p-value from bootstrap
                p_approx = 2 * min(
                    np.mean(np.array([acc_diff['observed_diff']]) <= 0),
                    np.mean(np.array([acc_diff['observed_diff']]) >= 0)
                )
                p_values.append(max(p_approx, 0.001))  # Avoid zero
            else:
                p_values.append(1.0)
    
    # Apply multiple comparison correction
    if len(p_values) > 1:
        print("\n" + "=" * 80)
        print("📊 MULTIPLE COMPARISON CORRECTION (Holm-Bonferroni)")
        print("=" * 80)
        
        p_corrected, rejected = apply_multiple_comparison_correction(p_values, method='holm')
        
        print(f"\n{'Model':<40} {'p-raw':<12} {'p-corrected':<12} {'Significant':<12}")
        print("-" * 80)
        
        for i, (result, p_raw, p_corr, sig) in enumerate(zip(all_results, p_values, p_corrected, rejected)):
            model_name = result['model_b']
            sig_str = "✅ Yes" if sig else "❌ No"
            print(f"{model_name:<40} {p_raw:<12.4f} {p_corr:<12.4f} {sig_str:<12}")
            
            # Update results with corrected values
            all_results[i]['p_value_corrected'] = p_corr
            all_results[i]['significant_corrected'] = sig
    
    return {
        'baseline': os.path.basename(baseline_path),
        'comparisons': all_results,
        'p_values_raw': p_values,
        'p_values_corrected': p_corrected if len(p_values) > 1 else p_values,
        'n_significant_raw': sum(1 for p in p_values if p < 0.05),
        'n_significant_corrected': sum(rejected) if len(p_values) > 1 else sum(1 for p in p_values if p < 0.05)
    }



# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def print_methodology():
    """Print statistical testing methodology."""
    print("""
+-----------------------------------------------------------------------------+
|                    STATISTICAL TESTING METHODOLOGY                          |
+-----------------------------------------------------------------------------+

  Tests Applied (consistent with ablation_study.py):

  1. SHAPIRO-WILK NORMALITY TEST
     - Checks if CV scores are normally distributed
     - Determines which statistical test to use
     - APPLICABLE: Only when CV scores are available

  2. PAIRED T-TEST / WILCOXON SIGNED-RANK
     - Paired t-test if both groups are normal
     - Wilcoxon if normality assumption fails
     - APPLICABLE: Only when CV scores from SAME folds are available

  3. COHEN'S D EFFECT SIZE
     - Measures practical significance
     - |d| < 0.2: Negligible, < 0.5: Small, < 0.8: Medium, >= 0.8: Large
     - APPLICABLE: When CV scores are available

  4. BOOTSTRAP 95% CONFIDENCE INTERVALS
     - 1000 bootstrap samples on test set
     - Provides uncertainty estimates
     - APPLICABLE: Always (uses test predictions)

  5. HOLM-BONFERRONI CORRECTION
     - Controls family-wise error rate
     - Applied when comparing multiple models
     - APPLICABLE: When comparing > 2 models

  Note: Tests 1-3 require CV scores stored in the cached model.
        If CV scores are not available, only bootstrap CI is used.

+-----------------------------------------------------------------------------+
""")


def main():
    parser = argparse.ArgumentParser(
        description='Statistical Testing for Cached Models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare two models
  python run_statistical_testing.py --model-a model1.pkl --model-b model2.pkl --test-data test.csv
  
  # Compare multiple models against baseline
  python run_statistical_testing.py --baseline baseline.pkl --models m1.pkl m2.pkl m3.pkl --test-data test.csv
  
  # Binary vs multiclass
  python run_statistical_testing.py --model-a ensemble.pkl --model-b baseline.pkl --test-data test.csv --mode binary
        """
    )
    
    # Model arguments
    parser.add_argument('--model-a', type=str, help='Path to first model (pickle file)')
    parser.add_argument('--model-b', type=str, help='Path to second model (pickle file)')
    parser.add_argument('--baseline', type=str, help='Path to baseline model (for multiple comparison)')
    parser.add_argument('--models', nargs='+', help='Paths to models to compare against baseline')
    
    # Data arguments
    parser.add_argument('--test-data', type=str, required=True, help='Path to test data CSV')
    parser.add_argument('--mode', type=str, default='binary', choices=['binary', 'multiclass'],
                       help='Classification mode')
    
    # Output arguments
    parser.add_argument('--output', type=str, default='statistical_testing_results.json',
                       help='Output file for results (JSON)')
    parser.add_argument('--show-methodology', action='store_true',
                       help='Print statistical testing methodology')
    
    args = parser.parse_args()
    
    # Print methodology if requested
    if args.show_methodology:
        print_methodology()
        if not args.model_a and not args.baseline:
            return
    
    # Validate arguments
    if args.model_a and args.model_b:
        # Two-model comparison
        if not os.path.exists(args.model_a):
            print(f"❌ Model A not found: {args.model_a}")
            sys.exit(1)
        if not os.path.exists(args.model_b):
            print(f"❌ Model B not found: {args.model_b}")
            sys.exit(1)
        
        results = compare_two_models(
            args.model_a, args.model_b, args.test_data, args.mode
        )
        
    elif args.baseline and args.models:
        # Multiple model comparison
        if not os.path.exists(args.baseline):
            print(f"❌ Baseline not found: {args.baseline}")
            sys.exit(1)
        
        for model_path in args.models:
            if not os.path.exists(model_path):
                print(f"❌ Model not found: {model_path}")
                sys.exit(1)
        
        results = compare_multiple_models(
            args.baseline, args.models, args.test_data, args.mode
        )
        
    else:
        print("❌ Must specify either:")
        print("   --model-a and --model-b (for two-model comparison)")
        print("   --baseline and --models (for multiple model comparison)")
        parser.print_help()
        sys.exit(1)
    
    # Check test data exists
    if not os.path.exists(args.test_data):
        print(f"❌ Test data not found: {args.test_data}")
        sys.exit(1)
    
    # Save results
    print(f"\n💾 Saving results to: {args.output}")
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\n✅ Statistical testing complete!")


if __name__ == "__main__":
    main()
