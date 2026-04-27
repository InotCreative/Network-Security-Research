# Model Card: xgboost (Fold 1)

## Configuration

```yaml
family: tree
fold: 1
calibration: isotonic
weight: 0.343695560377234
global_seed: 42
```

## Metrics

- **macro_f1**: 0.592625
- **weighted_f1**: 0.884261
- **accuracy**: 0.887654
- **log_loss**: 0.289208
- **brier_score**: 0.014473
- **ece**: 0.005968
- **roc_auc_macro_ovr**: 0.982222
- **recall_normal**: 0.986892
- **precision_normal**: 0.968054
- **f1_normal**: 0.977382
- **recall_analysis**: 0.044118
- **precision_analysis**: 0.857143
- **f1_analysis**: 0.083916
- **recall_backdoor**: 0.042735
- **precision_backdoor**: 0.714286
- **f1_backdoor**: 0.080645
- **recall_dos**: 0.580685
- **precision_dos**: 0.461613
- **f1_dos**: 0.514348
- **recall_exploits**: 0.766846
- **precision_exploits**: 0.675772
- **f1_exploits**: 0.718434
- **recall_fuzzers**: 0.663366
- **precision_fuzzers**: 0.87013
- **f1_fuzzers**: 0.752809
- **recall_generic**: 0.979603
- **precision_generic**: 0.993285
- **f1_generic**: 0.986396
- **recall_reconnaissance**: 0.811159
- **precision_reconnaissance**: 0.918963
- **f1_reconnaissance**: 0.861702
- **recall_shellcode**: 0.666667
- **precision_shellcode**: 0.574713
- **f1_shellcode**: 0.617284
- **recall_worms**: 0.222222
- **precision_worms**: 0.666667
- **f1_worms**: 0.333333

