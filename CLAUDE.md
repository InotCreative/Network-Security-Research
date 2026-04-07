# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research codebase implementing a **calibrated, disagreement-aware ensemble IDS** on the UNSW-NB15 network intrusion detection dataset. The central claim: a calibrated, disagreement-aware ensemble improves multiclass decision quality under class imbalance more convincingly than plain stacking or single-model baselines. The repository must be submission-ready and scientifically defensible.

The design document (`Claude_Code_Design_Document_UNSW_NB15_Ensemble.docx`) is the authoritative specification — every implementation decision should trace back to it.

## Data

All data lives in `data/`. Key files:
- `UNSW_NB15_training-set.csv` / `UNSW_NB15_testing-set.csv` — official splits used for all experiments
- `UNSW-NB15_1.csv` through `UNSW-NB15_4.csv` — raw source files
- `NUSW-NB15_features.csv` — feature schema reference (49 features + `attack_cat` + `label`)

The training set target columns are `attack_cat` (multiclass, primary task) and `label` (binary, secondary task).

## Planned Repository Architecture

```
src/
  data/          # Load official splits, validate schema, dtype contracts
  features/      # Audited flow-feature registry (one source of truth for formulas)
  pipeline/      # Experiment orchestration driven by configs/
  models/        # Base learners (RF, ExtraTrees, XGBoost, KNN) with family-specific preprocessing
  ensemble/      # Calibration wrappers, weighted combiner, stacker, disagreement features, gate
  select/        # Stability-weighted consensus feature selection
  eval/          # Metrics, reliability diagrams, confidence intervals, statistical tests
  utils/         # Seeds, serialization, logging, artifact management
configs/         # Declarative YAML/JSON experiment configs
reports/         # Manuscript-ready tables and figures (auto-generated)
artifacts/       # Serialized models, selected features, calibration reports, run manifests
tests/           # Unit, schema, leakage, smoke, and regression tests
```

## Implementation Order

Follow the staged plan from the design document:
1. **Stage 0** — Scaffold: repo layout, configs, schema validation, deterministic logging, smoke-test harness
2. **Stage 1** — Data loading and model-family preprocessing branches
3. **Stage 2** — Audited engineered-feature registry
4. **Stage 3** — Stability-weighted consensus feature selection
5. **Stage 4** — Single-model baselines with reproducible hyperparameter search
6. **Stage 5** — OOF prediction collection and calibration wrappers
7. **Stage 6** — Weighted averaging, stacker, disagreement meta-features, gate
8. **Stage 7** — Evaluation exporters, ablation runner, manuscript figures
9. **Stage 8** — Leak checks, full test suite, one-command reproduction

## Core Architectural Constraints

**Fold discipline is non-negotiable.** All preprocessors, selectors, calibrators, and meta-learners must be fit exclusively on training-fold data. Any leakage invalidates the paper.

**Preprocessing is model-family-specific:**
- Tree models (RF, ExtraTrees, XGBoost): median impute, ordinal-safe categorical handling
- Linear models (stacker, selectors): median impute + robust/standard scaling + one-hot encoding
- KNN: median impute + robust scaling + one-hot encoding (never ordinal-coded categories)

**Ensemble design (two-path + gate):**
- Path A — calibrated weighted average (`p_weighted`): per-model OOF calibration → inner-fold weights from macro F1 / log loss / Brier / calibration quality
- Path B — stacker (`p_stack`): multinomial logistic regression (default) trained on full calibrated OOF probability vectors
- Gate: lightweight model (logistic regression or shallow tree) outputting β(x) ∈ [0,1] from disagreement/uncertainty meta-features only
- Final: `p_final = β(x) · p_stack + (1 − β(x)) · p_weighted`

**Feature registry:** Every engineered feature must be registered with: name, paper display name, formula, input columns, transformation, clipping policy, and category. No formula duplication across scripts.

**One source of truth:** Schema, feature formulas, label mappings, and metric names are centralized. Hard-coded dataset quirks belong in a single location (schema/config), not scattered across modules.

## Commands

```bash
# Run full multiclass experiment end-to-end
python -m src.pipeline.run --config configs/multiclass_main.yaml

# Run binary sanity check
python -m src.pipeline.run --config configs/binary_sanity.yaml

# Run an ablation experiment (e.g. no_calibration)
python -m src.pipeline.run --config configs/ablations/no_calibration.yaml

# Run smoke tests only (fast, synthetic data, no real data needed)
python -m pytest tests/smoke/ -v

# Run full test suite (36 tests)
python -m pytest tests/ -v

# Run leakage-specific tests
python -m pytest tests/leakage/ -v

# Run regression tests on real data (requires data/ files)
python -m pytest tests/regression/ -v

# Run a single test
python -m pytest tests/unit/test_features.py::test_ratio_features_in_0_1 -v
```

## Required Output Artifacts

Every experiment run must emit (into `artifacts/` and `reports/`):
- `results_main_multiclass.csv` + JSON summary
- `ablation_results.csv`
- `feature_stability.csv`
- `selected_features_multiclass.txt` + JSON
- `calibration_report.json` + reliability diagram plots
- `model_cards/` (Markdown per model)
- `run_manifest.yaml` — single ledger linking every artifact to its config and seed

**No result table should require manual copying from notebooks.**

## Key Scientific Rules

- **Multiclass path is primary.** Binary results are secondary/sanity-check only.
- If the proposed ensemble fails to beat plain stacking on multiclass macro F1 after a clean run, surface this as a warning — do not patch paper claims into code.
- Ablation set must cover: no calibration, simplified meta-features, fixed mixing (no gate), no stability weighting, no engineered features, no feature selection.
- Candidate feature-selection k values: `[8, 12, 16, 20, 24, 30, 36, 42, 50]` (configurable).
- Base models: Random Forest, Extra Trees, XGBoost, KNN. Keep KNN for manuscript continuity; ablations must honestly show if it adds noise.
