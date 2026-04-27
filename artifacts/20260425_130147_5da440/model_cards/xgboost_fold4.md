# Model Card: xgboost (Fold 4)

## Configuration

```yaml
family: tree
fold: 4
calibration: isotonic
weight: 0.35103798072113723
global_seed: 42
```

## Metrics

- **macro_f1**: 0.611245
- **weighted_f1**: 0.877589
- **accuracy**: 0.881574
- **log_loss**: 0.301197
- **brier_score**: 0.015162
- **ece**: 0.005643
- **roc_auc_macro_ovr**: 0.982687
- **recall_normal**: 0.987297
- **precision_normal**: 0.957787
- **f1_normal**: 0.972318
- **recall_analysis**: 0.074074
- **precision_analysis**: 0.833333
- **f1_analysis**: 0.136054
- **recall_backdoor**: 0.017241
- **precision_backdoor**: 0.4
- **f1_backdoor**: 0.033058
- **recall_dos**: 0.556235
- **precision_dos**: 0.449605
- **f1_dos**: 0.497268
- **recall_exploits**: 0.747194
- **precision_exploits**: 0.675599
- **f1_exploits**: 0.709595
- **recall_fuzzers**: 0.639439
- **precision_fuzzers**: 0.833333
- **f1_fuzzers**: 0.723623
- **recall_generic**: 0.978802
- **precision_generic**: 0.992744
- **f1_generic**: 0.985724
- **recall_reconnaissance**: 0.795422
- **precision_reconnaissance**: 0.922056
- **f1_reconnaissance**: 0.854071
- **recall_shellcode**: 0.644737
- **precision_shellcode**: 0.583333
- **f1_shellcode**: 0.6125
- **recall_worms**: 0.555556
- **precision_worms**: 0.625
- **f1_worms**: 0.588235

