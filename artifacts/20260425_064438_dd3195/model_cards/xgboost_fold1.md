# Model Card: xgboost (Fold 1)

## Configuration

```yaml
family: tree
fold: 1
calibration: isotonic
weight: 0.3386629674808923
global_seed: 42
```

## Metrics

- **macro_f1**: 0.626859
- **weighted_f1**: 0.883524
- **accuracy**: 0.886804
- **log_loss**: 0.291414
- **brier_score**: 0.014661
- **ece**: 0.006427
- **roc_auc_macro_ovr**: 0.98114
- **recall_normal**: 0.987432
- **precision_normal**: 0.965896
- **f1_normal**: 0.976545
- **recall_analysis**: 0.044118
- **precision_analysis**: 1.0
- **f1_analysis**: 0.084507
- **recall_backdoor**: 0.042735
- **precision_backdoor**: 0.625
- **f1_backdoor**: 0.08
- **recall_dos**: 0.601467
- **precision_dos**: 0.463277
- **f1_dos**: 0.523404
- **recall_exploits**: 0.754717
- **precision_exploits**: 0.676873
- **f1_exploits**: 0.713679
- **recall_fuzzers**: 0.662541
- **precision_fuzzers**: 0.871878
- **f1_fuzzers**: 0.75293
- **recall_generic**: 0.977748
- **precision_generic**: 0.991938
- **f1_generic**: 0.984792
- **recall_reconnaissance**: 0.806867
- **precision_reconnaissance**: 0.923077
- **f1_reconnaissance**: 0.861069
- **recall_shellcode**: 0.666667
- **precision_shellcode**: 0.588235
- **f1_shellcode**: 0.625
- **recall_worms**: 0.555556
- **precision_worms**: 0.833333
- **f1_worms**: 0.666667

