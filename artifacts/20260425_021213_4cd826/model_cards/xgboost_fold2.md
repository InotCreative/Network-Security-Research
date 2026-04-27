# Model Card: xgboost (Fold 2)

## Configuration

```yaml
family: tree
fold: 2
calibration: isotonic
weight: 0.3679036415210693
global_seed: 42
```

## Metrics

- **macro_f1**: 0.614189
- **weighted_f1**: 0.884096
- **accuracy**: 0.887404
- **log_loss**: 0.287793
- **brier_score**: 0.014478
- **ece**: 0.005853
- **roc_auc_macro_ovr**: 0.981877
- **recall_normal**: 0.986216
- **precision_normal**: 0.965599
- **f1_normal**: 0.975799
- **recall_analysis**: 0.059259
- **precision_analysis**: 0.888889
- **f1_analysis**: 0.111111
- **recall_backdoor**: 0.034188
- **precision_backdoor**: 0.8
- **f1_backdoor**: 0.065574
- **recall_dos**: 0.5978
- **precision_dos**: 0.463947
- **f1_dos**: 0.522436
- **recall_exploits**: 0.758311
- **precision_exploits**: 0.680097
- **f1_exploits**: 0.717077
- **recall_fuzzers**: 0.663644
- **precision_fuzzers**: 0.862808
- **f1_fuzzers**: 0.750233
- **recall_generic**: 0.983572
- **precision_generic**: 0.993842
- **f1_generic**: 0.98868
- **recall_reconnaissance**: 0.796853
- **precision_reconnaissance**: 0.931438
- **f1_reconnaissance**: 0.858905
- **recall_shellcode**: 0.6
- **precision_shellcode**: 0.5625
- **f1_shellcode**: 0.580645
- **recall_worms**: 0.666667
- **precision_worms**: 0.5
- **f1_worms**: 0.571429

