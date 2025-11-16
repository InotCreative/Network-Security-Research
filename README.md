# 🛡️ Advanced Network Intrusion Detection System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-research-yellow.svg)

**A novel ensemble machine learning framework for enhanced network intrusion detection**

[Features](#-features) • [Architecture](#-architecture) • [Installation](#-installation) • [Usage](#-usage) • [Results](#-results) • [Citation](#-citation)

</div>

---

## 📋 Overview

This project presents a **three-tier ensemble machine learning framework** designed for high-accuracy network intrusion detection. The system combines dynamic feature engineering, intelligent feature selection, and adaptive meta-learning to identify sophisticated cyberattacks while maintaining low false-positive rates.

### 🎯 Key Highlights

- **23 Domain-Specific Features** extracted from duration, statistical, and behavioral patterns
- **Hybrid Feature Selection** combining F-test, mutual information, and recursive feature elimination
- **10 Diverse Base Classifiers** fused through a Logistic Regression-based meta-learner
- **Data-Driven Optimization** with automatic mixing ratio selection
- **Dual-Mode Detection** supporting both binary (attack/normal) and multiclass (attack type) classification
- **Attack-Type Specialists** for fine-grained threat identification
- **Rigorous Validation** with statistical significance testing and cross-dataset evaluation

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Raw Network Traffic                      │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Tier 1: Feature Engineering                    │
│  • Temporal patterns (hour, day, business hours)            │
│  • Statistical features (byte ratios, packet sizes)         │
│  • Behavioral patterns (connection duration, throughput)    │
│  • Port-based features (well-known ports, services)         │
│  • Network quality metrics (jitter, loss, window sizes)     │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│           Tier 2: Intelligent Feature Selection             │
│  • F-test (univariate statistical testing)                  │
│  • Mutual Information (dependency detection)                │
│  • RFE with Random Forest (tree-based importance)           │
│  • RFE with SVM (margin-based importance)                   │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│            Tier 3: Adaptive Meta-Learning                   │
│                                                             │
│  Base Classifiers:                                          │
│  ├─ Random Forest        ├─ Gradient Boosting               │
│  ├─ Extra Trees          ├─ XGBoost                         │
│  ├─ SVM (RBF kernel)     ├─ K-Nearest Neighbors             │
│  ├─ Logistic Regression  ├─ Naive Bayes                     │
│  ├─ Decision Tree        └─ SGD Classifier                  │
│                                                             │
│  Meta-Learner: OneVsRest(LogisticRegression)                │
│  Mixing Strategy: Optimized weighted + meta combination     │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  Attack Classification                      │
│  • Binary: Normal vs Attack                                 │
│  • Multiclass: Specific attack type identification          │
│  • Specialists: Per-attack-type binary classifiers          │
└─────────────────────────────────────────────────────────────┘
```

---

## ✨ Features

### 🔬 Advanced Feature Engineering
- **Dynamic Feature Creation**: Automatically generates 23 domain-specific features
- **Temporal Analysis**: Business hours, weekend patterns, time-of-day features
- **Statistical Metrics**: Byte ratios, packet sizes, throughput calculations
- **Behavioral Patterns**: Connection duration, port usage, protocol interactions

### 🎯 Intelligent Feature Selection
- **Multi-Method Ensemble**: Combines 4 different selection strategies
- **Consensus-Based**: Features selected by multiple methods are prioritized
- **Adaptive**: Automatically determines optimal feature count
- **Validation-Driven**: Uses cross-validation to prevent overfitting

### 🤖 Adaptive Meta-Learning
- **10 Diverse Classifiers**: Covers different learning paradigms
- **Data-Aware Weighting**: Adjusts weights based on data characteristics
- **Optimized Mixing**: Tests 8 ratios (0.0-1.0) to find best combination
- **Smart Caching**: Saves trained models for fast retraining

### 🛡️ Attack Specialists
- **Per-Attack Models**: Specialized binary classifiers for each attack type
- **Leakage-Free Training**: Proper train/test splits with independent preprocessing
- **Imbalance Handling**: Adaptive parameters for rare attack types
- **Comprehensive Evaluation**: F1, precision, recall, AUC metrics

### 📊 Rigorous Validation
- **Statistical Testing**: Cohen's d effect size, 95% confidence intervals
- **Cross-Dataset Evaluation**: Tests generalization across different datasets
- **Overfitting Detection**: Monitors train-validation gaps
- **Baseline Comparison**: Compares against 8 standard algorithms

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/network-intrusion-detection.git
cd network-intrusion-detection

# Install dependencies
pip install -r requirements.txt

# Optional: Install XGBoost for enhanced performance
pip install xgboost
```

### Dependencies
```
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.2.0
matplotlib>=3.5.0
seaborn>=0.11.0
xgboost>=1.7.0 (optional)
scipy>=1.9.0
```

---

## 💻 Usage

### Quick Start

```python
from novel_ensemble_ml import NovelEnsembleSystem

# Initialize the system
system = NovelEnsembleSystem(classification_type='binary')

# Train on your dataset
system.fit(
    train_csv_path='data/UNSW_NB15_training-set.csv',
    cache_dir='Models'
)

# Evaluate on test data
results = system.evaluate(test_csv_path='data/UNSW_NB15_testing-set.csv')
```

### Binary Classification (Attack Detection)

```python
# Detect attacks vs normal traffic
system = NovelEnsembleSystem(classification_type='binary')
system.fit(train_csv_path='train.csv')
results = system.evaluate(test_csv_path='test.csv')
```

### Multiclass Classification (Attack Type Identification)

```python
# Classify specific attack types
system = NovelEnsembleSystem(classification_type='multiclass')
system.fit(train_csv_path='train.csv')
results = system.evaluate(test_csv_path='test.csv')
```

### Command-Line Interface

```bash
# Run binary classification experiment
python run_novel_ml.py

# Run multiclass classification experiment
python run_multiclass_experiment.py

# Perform cross-dataset validation
python test_cross_dataset.py

# Analyze feature distributions
python analyze_feature_distributions.py

# Run comprehensive robustness analysis
python robustness_analysis.py
```

### Advanced Options

```python
# Force retrain specific models
system.fit(
    train_csv_path='train.csv',
    force_retrain_models=['rf', 'xgb', 'svm']
)

# Custom cache directory
system.fit(
    train_csv_path='train.csv',
    cache_dir='custom_models'
)

# Save trained model
system.save_model(filepath='my_model.pkl')

# Load trained model
system = NovelEnsembleSystem.load_model(filepath='my_model.pkl')
```

---

## 📊 Results

### Performance Metrics (UNSW-NB15 Dataset)

#### Binary Classification
| Metric | Score |
|--------|-------|
| **Accuracy** | 93.48% |
| **Precision** | 92.73% |
| **Recall** | 97.44% |
| **F1-Score** | 95.03% |
| **AUC-ROC** | 98.82% |

#### Multiclass Classification
| Attack Type | F1-Score | Precision | Recall |
|-------------|----------|-----------|--------|
| Generic | 90.1% | 92.4% | 88.0% |
| Exploits | 53.0% | 38.1% | 87.0% |
| DoS | 8.9% | 93.7% | 4.6% |
| Reconnaissance | 0.7% | 76.2% | 0.4% |

### Mixing Ratio Optimization

The system automatically tests 8 different mixing ratios and selects the best:

```
Ratio    Acc      Prec     Rec      F1       AUC      Bal-Acc
------------------------------------------------------------
0.0      0.9431   0.9584   0.9522   0.9553   0.9902   0.9396
0.2      0.9498   0.9621   0.9593   0.9607   0.9920   0.9462
0.4      0.9531   0.9642   0.9623   0.9632   0.9928   0.9495
0.5      0.9539   0.9647   0.9630   0.9639   0.9931   0.9503
0.6      0.9544   0.9647   0.9639   0.9643   0.9933   0.9507
0.7      0.9551   0.9651   0.9645   0.9648   0.9934   0.9514
0.8      0.9553   0.9653   0.9648   0.9651   0.9935   0.9517
1.0 ✓    0.9558   0.9657   0.9650   0.9654   0.9937   0.9522
```

---

## 📁 Project Structure

```
network-intrusion-detection/
├── novel_ensemble_ml.py              # Core ensemble system
├── run_novel_ml.py                   # Binary classification runner
├── run_multiclass_experiment.py      # Multiclass classification runner
├── test_cross_dataset.py             # Cross-dataset validation
├── robustness_analysis.py            # Robustness testing
├── airtight_statistical_validation.py # Statistical tests
├── analyze_feature_distributions.py  # Feature analysis
├── create_balanced_split.py          # Dataset balancing
├── realistic_evaluation.py           # Evaluation utilities
├── requirements.txt                  # Python dependencies
├── README.md                         # This file
└── Models/                           # Cached trained models
    ├── Binary/                       # Binary classification models
    └── Multiclass/                   # Multiclass classification models
```

---

## 🔬 Methodology

### 1. Data Preprocessing
- Missing value imputation
- Categorical encoding (Label Encoding, One-Hot Encoding)
- Feature scaling (RobustScaler)
- Outlier handling

### 2. Feature Engineering
- **Temporal Features**: Hour, day of week, business hours, night time
- **Statistical Features**: Byte ratios, packet ratios, throughput
- **Behavioral Features**: Connection duration, port patterns
- **Network Quality**: Jitter, loss ratios, window sizes

### 3. Feature Selection
- **Univariate**: F-test statistical testing
- **Information Theory**: Mutual information
- **Wrapper Methods**: RFE with Random Forest and SVM
- **Consensus**: Features selected by multiple methods

### 4. Ensemble Learning
- **Base Classifiers**: 10 diverse algorithms
- **Meta-Learning**: Logistic Regression with OneVsRest
- **Adaptive Weighting**: Data characteristic-based weights
- **Mixing Optimization**: Validation-based ratio selection

### 5. Validation
- **Cross-Validation**: 5-fold stratified CV
- **Statistical Tests**: Cohen's d, confidence intervals
- **Baseline Comparison**: Against 8 standard algorithms
- **Cross-Dataset**: Generalization testing

---

## 🎓 Research Contributions

1. **Novel Three-Tier Architecture**: Integrates feature engineering, selection, and meta-learning
2. **Data-Driven Optimization**: Automatic mixing ratio selection based on validation performance
3. **Attack-Type Specialists**: Per-attack binary classifiers for improved detection
4. **Comprehensive Validation**: Statistical significance testing and cross-dataset evaluation
5. **Leakage-Free Implementation**: Proper train/test separation at all stages

---

## 📈 Performance Considerations

### Strengths
- ✅ High recall (97.44%) - catches most attacks
- ✅ Excellent AUC (98.82%) - strong discriminative power
- ✅ Robust feature engineering - 23 domain-specific features
- ✅ Adaptive to data characteristics
- ✅ Smart caching for fast retraining

### Limitations
- ⚠️ Ensemble may not always outperform best individual model
- ⚠️ Specialist models struggle with rare attack types
- ⚠️ High computational cost during training
- ⚠️ Requires careful hyperparameter tuning

---

## 🛠️ Development

### Running Tests

```bash
# Run statistical validation
python airtight_statistical_validation.py

# Analyze feature importance
python analyze_feature_distributions.py

# Test robustness
python robustness_analysis.py
```

### Model Caching

Models are automatically cached in the `Models/` directory:
- `Models/Binary/` - Binary classification models
- `Models/Multiclass/` - Multiclass classification models

To force retrain specific models:
```python
system.fit(train_csv_path='train.csv', force_retrain_models=['rf', 'xgb'])
```

---

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@article{yourname2024intrusion,
  title={Advanced Network Intrusion Detection Using Adaptive Ensemble Machine Learning},
  author={Your Name},
  journal={Journal Name},
  year={2024}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📧 Contact

For questions or collaboration opportunities, please open an issue or contact [your.email@example.com](mailto:your.email@example.com).

---

## 🙏 Acknowledgments

- **UNSW-NB15 Dataset**: University of New South Wales for providing the dataset
- **scikit-learn**: For the excellent machine learning library
- **XGBoost**: For the high-performance gradient boosting implementation

---

<div align="center">

**⭐ Star this repository if you find it helpful!**

Made with ❤️ for the cybersecurity community

</div>
