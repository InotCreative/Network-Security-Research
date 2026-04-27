# Model Card: xgboost (Fold 2)

## Configuration

```yaml
family: tree
fold: 2
calibration: isotonic
weight: 0.36546865082966473
global_seed: 42
```

## Metrics

- **macro_f1**: 0.616847
- **weighted_f1**: 0.879921
- **accuracy**: 0.883578
- **log_loss**: 0.299254
- **brier_score**: 0.015007
- **ece**: 0.00514
- **roc_auc_macro_ovr**: 0.971853
- **recall_normal**: 0.985676
- **precision_normal**: 0.959484
- **f1_normal**: 0.972404
- **recall_analysis**: 0.059259
- **precision_analysis**: 1.0
- **f1_analysis**: 0.111888
- **recall_backdoor**: 0.042735
- **precision_backdoor**: 0.833333
- **f1_backdoor**: 0.081301
- **recall_dos**: 0.581907
- **precision_dos**: 0.458574
- **f1_dos**: 0.512931
- **recall_exploits**: 0.754717
- **precision_exploits**: 0.671463
- **f1_exploits**: 0.71066
- **recall_fuzzers**: 0.637263
- **precision_fuzzers**: 0.866592
- **f1_fuzzers**: 0.734442
- **recall_generic**: 0.981717
- **precision_generic**: 0.993298
- **f1_generic**: 0.987473
- **recall_reconnaissance**: 0.799714
- **precision_reconnaissance**: 0.930116
- **f1_reconnaissance**: 0.86
- **recall_shellcode**: 0.573333
- **precision_shellcode**: 0.558442
- **f1_shellcode**: 0.565789
- **recall_worms**: 0.666667
- **precision_worms**: 0.6
- **f1_worms**: 0.631579

