# Model Card: xgboost (Fold 2)

## Configuration

```yaml
family: tree
fold: 2
calibration: isotonic
weight: 0.38888120207526056
global_seed: 42
```

## Metrics

- **macro_f1**: 0.612986
- **weighted_f1**: 0.882107
- **accuracy**: 0.885522
- **log_loss**: 0.290358
- **brier_score**: 0.014616
- **ece**: 0.004227
- **roc_auc_macro_ovr**: 0.98178
- **recall_normal**: 0.986892
- **precision_normal**: 0.963584
- **f1_normal**: 0.975098
- **recall_analysis**: 0.059259
- **precision_analysis**: 0.888889
- **f1_analysis**: 0.111111
- **recall_backdoor**: 0.042735
- **precision_backdoor**: 0.714286
- **f1_backdoor**: 0.080645
- **recall_dos**: 0.586797
- **precision_dos**: 0.458453
- **f1_dos**: 0.514745
- **recall_exploits**: 0.752022
- **precision_exploits**: 0.6731
- **f1_exploits**: 0.710376
- **recall_fuzzers**: 0.652927
- **precision_fuzzers**: 0.882943
- **f1_fuzzers**: 0.750711
- **recall_generic**: 0.982777
- **precision_generic**: 0.990123
- **f1_generic**: 0.986436
- **recall_reconnaissance**: 0.802575
- **precision_reconnaissance**: 0.925743
- **f1_reconnaissance**: 0.85977
- **recall_shellcode**: 0.573333
- **precision_shellcode**: 0.565789
- **f1_shellcode**: 0.569536
- **recall_worms**: 0.666667
- **precision_worms**: 0.5
- **f1_worms**: 0.571429

