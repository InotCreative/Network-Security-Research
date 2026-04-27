# Model Card: xgboost (Fold 4)

## Configuration

```yaml
family: tree
fold: 4
calibration: isotonic
weight: 0.32725642185264964
global_seed: 42
```

## Metrics

- **macro_f1**: 0.611113
- **weighted_f1**: 0.875142
- **accuracy**: 0.879266
- **log_loss**: 0.309592
- **brier_score**: 0.015515
- **ece**: 0.005396
- **roc_auc_macro_ovr**: 0.981499
- **recall_normal**: 0.984324
- **precision_normal**: 0.956282
- **f1_normal**: 0.970101
- **recall_analysis**: 0.074074
- **precision_analysis**: 0.909091
- **f1_analysis**: 0.136986
- **recall_backdoor**: 0.017241
- **precision_backdoor**: 0.5
- **f1_backdoor**: 0.033333
- **recall_dos**: 0.531785
- **precision_dos**: 0.446154
- **f1_dos**: 0.48522
- **recall_exploits**: 0.736866
- **precision_exploits**: 0.678379
- **f1_exploits**: 0.706414
- **recall_fuzzers**: 0.666667
- **precision_fuzzers**: 0.780676
- **f1_fuzzers**: 0.719181
- **recall_generic**: 0.978272
- **precision_generic**: 0.993274
- **f1_generic**: 0.985716
- **recall_reconnaissance**: 0.788269
- **precision_reconnaissance**: 0.919866
- **f1_reconnaissance**: 0.848998
- **recall_shellcode**: 0.657895
- **precision_shellcode**: 0.617284
- **f1_shellcode**: 0.636943
- **recall_worms**: 0.555556
- **precision_worms**: 0.625
- **f1_worms**: 0.588235

