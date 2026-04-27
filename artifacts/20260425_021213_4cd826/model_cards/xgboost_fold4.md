# Model Card: xgboost (Fold 4)

## Configuration

```yaml
family: tree
fold: 4
calibration: isotonic
weight: 0.3523810886110191
global_seed: 42
```

## Metrics

- **macro_f1**: 0.612471
- **weighted_f1**: 0.875621
- **accuracy**: 0.87957
- **log_loss**: 0.310194
- **brier_score**: 0.015536
- **ece**: 0.00502
- **roc_auc_macro_ovr**: 0.97901
- **recall_normal**: 0.985135
- **precision_normal**: 0.956567
- **f1_normal**: 0.970641
- **recall_analysis**: 0.074074
- **precision_analysis**: 1.0
- **f1_analysis**: 0.137931
- **recall_backdoor**: 0.017241
- **precision_backdoor**: 0.4
- **f1_backdoor**: 0.033058
- **recall_dos**: 0.52934
- **precision_dos**: 0.443193
- **f1_dos**: 0.482451
- **recall_exploits**: 0.754378
- **precision_exploits**: 0.662722
- **f1_exploits**: 0.705586
- **recall_fuzzers**: 0.636964
- **precision_fuzzers**: 0.832794
- **f1_fuzzers**: 0.721833
- **recall_generic**: 0.976948
- **precision_generic**: 0.993265
- **f1_generic**: 0.985039
- **recall_reconnaissance**: 0.792561
- **precision_reconnaissance**: 0.937394
- **f1_reconnaissance**: 0.858915
- **recall_shellcode**: 0.657895
- **precision_shellcode**: 0.625
- **f1_shellcode**: 0.641026
- **recall_worms**: 0.555556
- **precision_worms**: 0.625
- **f1_worms**: 0.588235

