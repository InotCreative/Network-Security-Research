#!/usr/bin/env python3
"""
AIRTIGHT STATISTICAL VALIDATION - Comprehensive Statistical Hurdles Coverage
Addresses ALL statistical validation requirements for rigorous ML research
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import ttest_rel, wilcoxon, shapiro, levene, spearmanr
from statsmodels.stats.multitest import multipletests
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, accuracy_score
from sklearn.utils import resample
import warnings
warnings.filterwarnings('ignore')

class AirtightStatisticalValidator:
    """Comprehensive statistical validation covering all research hurdles"""
    
    def __init__(self):
        self.results = {}
        self.p_values = []  # For multiple comparisons correction
        
    def test_statistical_significance(self, results_a, results_b, method='auto', alpha=0.05):
        """Test statistical significance between two sets of results"""
        print("📊 TESTING STATISTICAL SIGNIFICANCE")
        print("-" * 40)
        
        results_a = np.array(results_a)
        results_b = np.array(results_b)
        
        # Test normality assumptions
        _, p_norm_a = shapiro(results_a)
        _, p_norm_b = shapiro(results_b)
        
        both_normal = (p_norm_a > 0.05) and (p_norm_b > 0.05)
        
        print(f"   📈 Group A normality p-value: {p_norm_a:.4f}")
        print(f"   📈 Group B normality p-value: {p_norm_b:.4f}")
        print(f"   📊 Both groups normal: {both_normal}")
        
        # Choose appropriate test
        if method == 'auto':
            if both_normal and len(results_a) > 5:
                test_method = 'ttest'
            else:
                test_method = 'wilcoxon'
        else:
            test_method = method
        
        # Perform statistical test
        if test_method == 'ttest':
            statistic, p_value = ttest_rel(results_a, results_b)
            test_name = "Paired t-test"
        else:
            statistic, p_value = wilcoxon(results_a, results_b, alternative='two-sided')
            test_name = "Wilcoxon signed-rank test"
        
        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt((np.var(results_a, ddof=1) + np.var(results_b, ddof=1)) / 2)
        if pooled_std > 0:
            cohens_d = (np.mean(results_a) - np.mean(results_b)) / pooled_std
        else:
            cohens_d = 0
        
        # Interpret effect size
        if abs(cohens_d) < 0.2:
            effect_interpretation = "negligible"
        elif abs(cohens_d) < 0.5:
            effect_interpretation = "small"
        elif abs(cohens_d) < 0.8:
            effect_interpretation = "medium"
        else:
            effect_interpretation = "large"
        
        # Store p-value for multiple comparisons correction
        self.p_values.append(p_value)
        
        print(f"   🧪 Test used: {test_name}")
        print(f"   📊 Test statistic: {statistic:.4f}")
        print(f"   📊 P-value: {p_value:.4f}")
        print(f"   📊 Effect size (Cohen's d): {cohens_d:.4f} ({effect_interpretation})")
        print(f"   ✅ Significant: {'Yes' if p_value < alpha else 'No'} (α = {alpha})")
        
        return {
            'test_method': test_name,
            'statistic': statistic,
            'p_value': p_value,
            'effect_size': cohens_d,
            'effect_interpretation': effect_interpretation,
            'significant': p_value < alpha,
            'alpha': alpha
        }
    
    def correct_multiple_comparisons(self, method='holm'):
        """Apply multiple comparisons correction to all collected p-values"""
        print(f"\n🔢 MULTIPLE COMPARISONS CORRECTION")
        print("-" * 40)
        
        if len(self.p_values) == 0:
            print("   ⚠️  No p-values collected for correction")
            return None
        
        print(f"   📊 Number of comparisons: {len(self.p_values)}")
        print(f"   📊 Correction method: {method}")
        print(f"   📊 Original p-values: {[f'{p:.4f}' for p in self.p_values]}")
        
        # Apply correction
        rejected, p_corrected, alpha_sidak, alpha_bonf = multipletests(
            self.p_values, alpha=0.05, method=method
        )
        
        print(f"   📊 Corrected p-values: {[f'{p:.4f}' for p in p_corrected]}")
        print(f"   📊 Significant after correction: {sum(rejected)}/{len(rejected)}")
        
        # Show results table
        print(f"\n   📋 CORRECTION RESULTS:")
        for i, (orig_p, corr_p, is_sig) in enumerate(zip(self.p_values, p_corrected, rejected)):
            print(f"      Test {i+1}: {orig_p:.4f} → {corr_p:.4f} {'✅' if is_sig else '❌'}")
        
        return {
            'original_p_values': self.p_values,
            'corrected_p_values': p_corrected.tolist(),
            'rejected': rejected.tolist(),
            'method': method,
            'significant_count': sum(rejected)
        }
    
    def assess_probability_calibration(self, y_true, y_prob, n_bins=10):
        """Assess probability calibration quality"""
        print(f"\n🎯 PROBABILITY CALIBRATION ASSESSMENT")
        print("-" * 40)
        
        # Reliability diagram data
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_prob, n_bins=n_bins
        )
        
        # Brier score (lower is better, 0 = perfect)
        brier_score = brier_score_loss(y_true, y_prob)
        
        # Expected Calibration Error (ECE)
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_prob[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        print(f"   📊 Brier Score: {brier_score:.4f} (lower is better)")
        print(f"   📊 Expected Calibration Error: {ece:.4f}")
        
        # Calibration quality assessment
        if ece < 0.05:
            calibration_quality = "Excellent"
        elif ece < 0.1:
            calibration_quality = "Good"
        elif ece < 0.15:
            calibration_quality = "Fair"
        else:
            calibration_quality = "Poor"
        
        print(f"   🎯 Calibration Quality: {calibration_quality}")
        
        return {
            'brier_score': brier_score,
            'expected_calibration_error': ece,
            'calibration_quality': calibration_quality,
            'reliability_curve': (fraction_of_positives, mean_predicted_value)
        }
    
    def bootstrap_confidence_intervals(self, data, statistic_func, n_bootstrap=1000, confidence=0.95):
        """Calculate bootstrap confidence intervals"""
        print(f"\n🔄 BOOTSTRAP CONFIDENCE INTERVALS")
        print("-" * 40)
        
        bootstrap_stats = []
        
        print(f"   🔄 Bootstrap samples: {n_bootstrap}")
        print(f"   📊 Confidence level: {confidence*100}%")
        
        for i in range(n_bootstrap):
            # Bootstrap sample
            bootstrap_sample = resample(data, random_state=i)
            stat = statistic_func(bootstrap_sample)
            bootstrap_stats.append(stat)
        
        bootstrap_stats = np.array(bootstrap_stats)
        
        # Calculate confidence interval
        alpha = 1 - confidence
        lower_percentile = (alpha/2) * 100
        upper_percentile = (1 - alpha/2) * 100
        
        ci_lower = np.percentile(bootstrap_stats, lower_percentile)
        ci_upper = np.percentile(bootstrap_stats, upper_percentile)
        
        original_stat = statistic_func(data)
        
        print(f"   📊 Original statistic: {original_stat:.4f}")
        print(f"   📊 Bootstrap mean: {np.mean(bootstrap_stats):.4f}")
        print(f"   📊 Bootstrap std: {np.std(bootstrap_stats):.4f}")
        print(f"   📊 {confidence*100}% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
        
        return {
            'original_statistic': original_stat,
            'bootstrap_mean': np.mean(bootstrap_stats),
            'bootstrap_std': np.std(bootstrap_stats),
            'confidence_interval': (ci_lower, ci_upper),
            'bootstrap_distribution': bootstrap_stats
        }

if __name__ == "__main__":
    print("🔬 AIRTIGHT STATISTICAL VALIDATION MODULE")
    print("This module provides comprehensive statistical validation")
    print("covering all research hurdles for rigorous ML publications.")