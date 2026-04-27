# Model Card: xgboost (Fold 1)

## Configuration

```yaml
family: tree
fold: 1
calibration: isotonic
weight: 0.35785077864660714
global_seed: 42
```

## Metrics

- **macro_f1**: 0.617006
- **weighted_f1**: 0.888168
- **accuracy**: 0.89069
- **log_loss**: 0.28049
- **brier_score**: 0.014069
- **ece**: 0.006422
- **roc_auc_macro_ovr**: 0.972884
- **recall_normal**: 0.988514
- **precision_normal**: 0.972481
- **f1_normal**: 0.980432
- **recall_analysis**: 0.051471
- **precision_analysis**: 0.875
- **f1_analysis**: 0.097222
- **recall_backdoor**: 0.051282
- **precision_backdoor**: 0.857143
- **f1_backdoor**: 0.096774
- **recall_dos**: 0.629584
- **precision_dos**: 0.46147
- **f1_dos**: 0.532575
- **recall_exploits**: 0.751572
- **precision_exploits**: 0.681744
- **f1_exploits**: 0.714957
- **recall_fuzzers**: 0.676568
- **precision_fuzzers**: 0.891304
- **f1_fuzzers**: 0.769231
- **recall_generic**: 0.980662
- **precision_generic**: 0.993826
- **f1_generic**: 0.9872
- **recall_reconnaissance**: 0.822604
- **precision_reconnaissance**: 0.924437
- **f1_reconnaissance**: 0.870553
- **recall_shellcode**: 0.666667
- **precision_shellcode**: 0.581395
- **f1_shellcode**: 0.621118
- **recall_worms**: 0.444444
- **precision_worms**: 0.571429
- **f1_worms**: 0.5

