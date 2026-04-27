# Model Card: xgboost (Fold 2)

## Configuration

```yaml
family: tree
fold: 2
calibration: isotonic
weight: 0.3485316293756022
global_seed: 42
```

## Metrics

- **macro_f1**: 0.615651
- **weighted_f1**: 0.879644
- **accuracy**: 0.883518
- **log_loss**: 0.29938
- **brier_score**: 0.015011
- **ece**: 0.004792
- **roc_auc_macro_ovr**: 0.971647
- **recall_normal**: 0.985946
- **precision_normal**: 0.959621
- **f1_normal**: 0.972605
- **recall_analysis**: 0.059259
- **precision_analysis**: 1.0
- **f1_analysis**: 0.111888
- **recall_backdoor**: 0.034188
- **precision_backdoor**: 0.8
- **f1_backdoor**: 0.065574
- **recall_dos**: 0.574572
- **precision_dos**: 0.458984
- **f1_dos**: 0.510315
- **recall_exploits**: 0.755615
- **precision_exploits**: 0.671189
- **f1_exploits**: 0.710904
- **recall_fuzzers**: 0.637263
- **precision_fuzzers**: 0.859844
- **f1_fuzzers**: 0.732008
- **recall_generic**: 0.981717
- **precision_generic**: 0.992765
- **f1_generic**: 0.98721
- **recall_reconnaissance**: 0.802575
- **precision_reconnaissance**: 0.928808
- **f1_reconnaissance**: 0.86109
- **recall_shellcode**: 0.573333
- **precision_shellcode**: 0.573333
- **f1_shellcode**: 0.573333
- **recall_worms**: 0.666667
- **precision_worms**: 0.6
- **f1_worms**: 0.631579

