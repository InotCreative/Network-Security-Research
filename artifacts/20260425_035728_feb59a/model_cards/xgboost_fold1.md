# Model Card: xgboost (Fold 1)

## Configuration

```yaml
family: tree
fold: 1
calibration: isotonic
weight: 0.37413420240077266
global_seed: 42
```

## Metrics

- **macro_f1**: 0.616638
- **weighted_f1**: 0.885625
- **accuracy**: 0.888808
- **log_loss**: 0.287457
- **brier_score**: 0.014442
- **ece**: 0.00726
- **roc_auc_macro_ovr**: 0.982413
- **recall_normal**: 0.987432
- **precision_normal**: 0.967174
- **f1_normal**: 0.977198
- **recall_analysis**: 0.051471
- **precision_analysis**: 0.875
- **f1_analysis**: 0.097222
- **recall_backdoor**: 0.059829
- **precision_backdoor**: 0.7
- **f1_backdoor**: 0.110236
- **recall_dos**: 0.599022
- **precision_dos**: 0.473888
- **f1_dos**: 0.529158
- **recall_exploits**: 0.760557
- **precision_exploits**: 0.6772
- **f1_exploits**: 0.716462
- **recall_fuzzers**: 0.669967
- **precision_fuzzers**: 0.875944
- **f1_fuzzers**: 0.759233
- **recall_generic**: 0.978543
- **precision_generic**: 0.993278
- **f1_generic**: 0.985855
- **recall_reconnaissance**: 0.818312
- **precision_reconnaissance**: 0.919614
- **f1_reconnaissance**: 0.866011
- **recall_shellcode**: 0.666667
- **precision_shellcode**: 0.588235
- **f1_shellcode**: 0.625
- **recall_worms**: 0.444444
- **precision_worms**: 0.571429
- **f1_worms**: 0.5

