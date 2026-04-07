"""
Smoke test: run the full pipeline on a tiny stratified sample.

Verifies:
  - Data loading and schema validation
  - Feature engineering
  - Consensus feature selection
  - EnsembleCombiner (OOF, calibration, stacker, gate, refit)
  - All predict_proba_all outputs have correct shape
  - No NaN in any output
  - Metrics are computed without error
  - No spuriously perfect accuracy

This test does NOT use the real data files — it generates a synthetic
sample with the correct schema to keep CI fast.
"""
import warnings
import numpy as np
import pandas as pd
import pytest

from src.data.schema import FEATURE_COLS, NUMERIC_COLS, CATEGORICAL_COLS, MULTICLASS_LABELS
from src.eval.metrics import sanity_check_accuracy
from src.utils.seeds import set_global_seed


N_SAMPLES = 400   # small but covers all 10 classes
N_CLASSES = 10


@pytest.fixture(scope="module")
def synthetic_dataset():
    """Generate a mini UNSW-NB15-shaped dataset."""
    set_global_seed(42)
    rng = np.random.default_rng(42)
    n = N_SAMPLES

    num_cols = [c for c in NUMERIC_COLS if c not in CATEGORICAL_COLS]
    data = {col: rng.random(n) * 100 for col in num_cols}
    data["proto"] = rng.choice(["tcp", "udp", "icmp"], size=n)
    data["service"] = rng.choice(["-", "http", "ftp", "dns"], size=n)
    data["state"] = rng.choice(["FIN", "INT", "CON", "REQ"], size=n)
    data["id"] = np.arange(n)

    # Labels: roughly balanced (10 samples per class minimum)
    labels = np.tile(np.arange(N_CLASSES), n // N_CLASSES + 1)[:n]
    rng.shuffle(labels)
    data["attack_cat"] = [MULTICLASS_LABELS[i] for i in labels]
    data["label"] = (labels > 0).astype(int)

    return pd.DataFrame(data)


def test_schema_validation(synthetic_dataset):
    from src.data.schema import SCHEMA
    SCHEMA.validate(synthetic_dataset)


def test_get_X_y(synthetic_dataset):
    from src.data.loader import get_X_y
    X, y = get_X_y(synthetic_dataset, task="multiclass")
    assert "attack_cat" not in X.columns
    assert "label" not in X.columns
    assert "id" not in X.columns
    assert len(X) == N_SAMPLES
    assert len(y) == N_SAMPLES


def test_feature_engineering(synthetic_dataset):
    from src.data.loader import get_X_y
    from src.features.engineer import FeatureEngineer
    X, y = get_X_y(synthetic_dataset)
    eng = FeatureEngineer()
    X_eng = eng.fit_transform(X)
    assert X_eng.shape[0] == N_SAMPLES
    assert X_eng.shape[1] > X.shape[1]   # new features added
    assert not X_eng.select_dtypes(include="number").isna().any().any()


def test_consensus_selection(synthetic_dataset):
    from src.data.loader import get_X_y
    from src.features.engineer import FeatureEngineer
    from src.select.consensus import ConsensusSelector
    X, y = get_X_y(synthetic_dataset)
    eng = FeatureEngineer()
    X_eng = eng.fit_transform(X)
    sel = ConsensusSelector(k_grid=[5, 8, 10], n_inner_folds=2, surrogate_n_estimators=10)
    result = sel.fit(X_eng, y.values)
    assert len(result.selected_names) > 0
    assert result.chosen_k <= X_eng.shape[1]


def test_full_ensemble_smoke(synthetic_dataset):
    from src.data.loader import get_X_y
    from src.features.engineer import FeatureEngineer
    from src.select.consensus import ConsensusSelector
    from src.models.base_models import build_default_models
    from src.ensemble.combiner import EnsembleCombiner

    X, y = get_X_y(synthetic_dataset)
    y_arr = y.values

    eng = FeatureEngineer()
    X_eng = eng.fit_transform(X)

    sel = ConsensusSelector(k_grid=[5, 8], n_inner_folds=2, surrogate_n_estimators=5)
    result = sel.fit(X_eng, y_arr)
    sel_cols = result.selected_names
    X_sel = X_eng[sel_cols]

    # Use small fold counts for speed
    models = build_default_models()
    combiner = EnsembleCombiner(
        base_models=models,
        n_classes=N_CLASSES,
        n_oof_splits=2,
        gate_model="ridge",
    )
    with warnings.catch_warnings(record=True):
        combiner.fit(X_sel, y_arr)
        all_probas = combiner.predict_proba_all(X_sel)

    # Shape checks
    for method, proba in all_probas.items():
        assert proba.shape == (N_SAMPLES, N_CLASSES), \
            f"{method}: wrong shape {proba.shape}"
        assert not np.isnan(proba).any(), f"{method}: NaN in output"
        # Rows must sum to ~1
        row_sums = proba.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-4), \
            f"{method}: rows don't sum to 1"


def test_sanity_check_warns_on_perfect_accuracy():
    """The leakage detector must warn when accuracy is >= 0.99."""
    y_true = np.arange(100) % 10
    y_pred = y_true.copy()   # perfect prediction
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        sanity_check_accuracy("test", y_true, y_pred, warn_threshold=0.99)
    assert any("LEAKAGE ALERT" in str(warning.message) for warning in w), \
        "sanity_check_accuracy did not warn on perfect accuracy"


def test_sanity_check_silent_on_reasonable_accuracy():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 10, 200)
    y_pred = rng.integers(0, 10, 200)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        sanity_check_accuracy("test", y_true, y_pred, warn_threshold=0.99)
    assert not any("LEAKAGE ALERT" in str(warning.message) for warning in w)
