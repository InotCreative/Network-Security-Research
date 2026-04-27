# Model Card: xgboost (Fold 3)

## Configuration

```yaml
family: tree
fold: 3
calibration: isotonic
weight: 0.35155324322913695
global_seed: 42
```

## Metrics

- **macro_f1**: 0.628104
- **weighted_f1**: 0.878837
- **accuracy**: 0.883092
- **log_loss**: 0.294686
- **brier_score**: 0.014923
- **ece**: 0.00602
- **roc_auc_macro_ovr**: 0.983017
- **recall_normal**: 0.98973
- **precision_normal**: 0.957386
- **f1_normal**: 0.973289
- **recall_analysis**: 0.088889
- **precision_analysis**: 1.0
- **f1_analysis**: 0.163265
- **recall_backdoor**: 0.017094
- **precision_backdoor**: 1.0
- **f1_backdoor**: 0.033613
- **recall_dos**: 0.567931
- **precision_dos**: 0.457594
- **f1_dos**: 0.506827
- **recall_exploits**: 0.749439
- **precision_exploits**: 0.686266
- **f1_exploits**: 0.716463
- **recall_fuzzers**: 0.635614
- **precision_fuzzers**: 0.797311
- **f1_fuzzers**: 0.707339
- **recall_generic**: 0.977477
- **precision_generic**: 0.993269
- **f1_generic**: 0.98531
- **recall_reconnaissance**: 0.802575
- **precision_reconnaissance**: 0.944444
- **f1_reconnaissance**: 0.867749
- **recall_shellcode**: 0.565789
- **precision_shellcode**: 0.589041
- **f1_shellcode**: 0.577181
- **recall_worms**: 0.75
- **precision_worms**: 0.75
- **f1_worms**: 0.75

