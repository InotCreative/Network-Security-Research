# Model Card: xgboost (Fold 3)

## Configuration

```yaml
family: tree
fold: 3
calibration: isotonic
weight: 0.3507180457926494
global_seed: 42
```

## Metrics

- **macro_f1**: 0.617944
- **weighted_f1**: 0.878416
- **accuracy**: 0.882303
- **log_loss**: 0.297356
- **brier_score**: 0.015034
- **ece**: 0.006359
- **roc_auc_macro_ovr**: 0.983524
- **recall_normal**: 0.989324
- **precision_normal**: 0.957995
- **f1_normal**: 0.973408
- **recall_analysis**: 0.088889
- **precision_analysis**: 1.0
- **f1_analysis**: 0.163265
- **recall_backdoor**: 0.008547
- **precision_backdoor**: 1.0
- **f1_backdoor**: 0.016949
- **recall_dos**: 0.565483
- **precision_dos**: 0.452498
- **f1_dos**: 0.50272
- **recall_exploits**: 0.758419
- **precision_exploits**: 0.668117
- **f1_exploits**: 0.71041
- **recall_fuzzers**: 0.628195
- **precision_fuzzers**: 0.858108
- **f1_fuzzers**: 0.725369
- **recall_generic**: 0.973503
- **precision_generic**: 0.993779
- **f1_generic**: 0.983536
- **recall_reconnaissance**: 0.804006
- **precision_reconnaissance**: 0.941374
- **f1_reconnaissance**: 0.867284
- **recall_shellcode**: 0.513158
- **precision_shellcode**: 0.549296
- **f1_shellcode**: 0.530612
- **recall_worms**: 0.75
- **precision_worms**: 0.666667
- **f1_worms**: 0.705882

