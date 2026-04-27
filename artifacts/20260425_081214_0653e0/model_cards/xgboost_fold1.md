# Model Card: xgboost (Fold 1)

## Configuration

```yaml
family: tree
fold: 1
calibration: isotonic
weight: 0.3535036527071698
global_seed: 42
```

## Metrics

- **macro_f1**: 0.61897
- **weighted_f1**: 0.88472
- **accuracy**: 0.888018
- **log_loss**: 0.289931
- **brier_score**: 0.014553
- **ece**: 0.007182
- **roc_auc_macro_ovr**: 0.972429
- **recall_normal**: 0.989324
- **precision_normal**: 0.966214
- **f1_normal**: 0.977632
- **recall_analysis**: 0.051471
- **precision_analysis**: 0.875
- **f1_analysis**: 0.097222
- **recall_backdoor**: 0.051282
- **precision_backdoor**: 0.75
- **f1_backdoor**: 0.096
- **recall_dos**: 0.581907
- **precision_dos**: 0.460794
- **f1_dos**: 0.514317
- **recall_exploits**: 0.762354
- **precision_exploits**: 0.67368
- **f1_exploits**: 0.715279
- **recall_fuzzers**: 0.655941
- **precision_fuzzers**: 0.888268
- **f1_fuzzers**: 0.754627
- **recall_generic**: 0.979868
- **precision_generic**: 0.994088
- **f1_generic**: 0.986926
- **recall_reconnaissance**: 0.812589
- **precision_reconnaissance**: 0.928105
- **f1_reconnaissance**: 0.866514
- **recall_shellcode**: 0.666667
- **precision_shellcode**: 0.561798
- **f1_shellcode**: 0.609756
- **recall_worms**: 0.444444
- **precision_worms**: 0.8
- **f1_worms**: 0.571429

