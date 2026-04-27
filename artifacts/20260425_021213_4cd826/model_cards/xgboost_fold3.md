# Model Card: xgboost (Fold 3)

## Configuration

```yaml
family: tree
fold: 3
calibration: isotonic
weight: 0.3543421730300502
global_seed: 42
```

## Metrics

- **macro_f1**: 0.623338
- **weighted_f1**: 0.877478
- **accuracy**: 0.881513
- **log_loss**: 0.296605
- **brier_score**: 0.015033
- **ece**: 0.007193
- **roc_auc_macro_ovr**: 0.983282
- **recall_normal**: 0.989054
- **precision_normal**: 0.958486
- **f1_normal**: 0.97353
- **recall_analysis**: 0.088889
- **precision_analysis**: 1.0
- **f1_analysis**: 0.163265
- **recall_backdoor**: 0.008547
- **precision_backdoor**: 1.0
- **f1_backdoor**: 0.016949
- **recall_dos**: 0.559364
- **precision_dos**: 0.448919
- **f1_dos**: 0.498093
- **recall_exploits**: 0.743152
- **precision_exploits**: 0.681631
- **f1_exploits**: 0.711063
- **recall_fuzzers**: 0.641385
- **precision_fuzzers**: 0.792261
- **f1_fuzzers**: 0.708884
- **recall_generic**: 0.975623
- **precision_generic**: 0.994329
- **f1_generic**: 0.984887
- **recall_reconnaissance**: 0.804006
- **precision_reconnaissance**: 0.935108
- **f1_reconnaissance**: 0.864615
- **recall_shellcode**: 0.565789
- **precision_shellcode**: 0.558442
- **f1_shellcode**: 0.562092
- **recall_worms**: 0.75
- **precision_worms**: 0.75
- **f1_worms**: 0.75

