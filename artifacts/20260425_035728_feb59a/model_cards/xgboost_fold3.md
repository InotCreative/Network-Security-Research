# Model Card: xgboost (Fold 3)

## Configuration

```yaml
family: tree
fold: 3
calibration: isotonic
weight: 0.3818423550521694
global_seed: 42
```

## Metrics

- **macro_f1**: 0.640866
- **weighted_f1**: 0.878979
- **accuracy**: 0.882971
- **log_loss**: 0.295512
- **brier_score**: 0.01496
- **ece**: 0.004832
- **roc_auc_macro_ovr**: 0.983521
- **recall_normal**: 0.98973
- **precision_normal**: 0.957511
- **f1_normal**: 0.973354
- **recall_analysis**: 0.088889
- **precision_analysis**: 1.0
- **f1_analysis**: 0.163265
- **recall_backdoor**: 0.025641
- **precision_backdoor**: 0.75
- **f1_backdoor**: 0.049587
- **recall_dos**: 0.560588
- **precision_dos**: 0.458
- **f1_dos**: 0.504128
- **recall_exploits**: 0.749888
- **precision_exploits**: 0.676936
- **f1_exploits**: 0.711547
- **recall_fuzzers**: 0.643034
- **precision_fuzzers**: 0.818468
- **f1_fuzzers**: 0.720222
- **recall_generic**: 0.975358
- **precision_generic**: 0.995942
- **f1_generic**: 0.985542
- **recall_reconnaissance**: 0.802575
- **precision_reconnaissance**: 0.928808
- **f1_reconnaissance**: 0.86109
- **recall_shellcode**: 0.578947
- **precision_shellcode**: 0.586667
- **f1_shellcode**: 0.582781
- **recall_worms**: 0.75
- **precision_worms**: 1.0
- **f1_worms**: 0.857143

