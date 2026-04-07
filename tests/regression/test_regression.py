"""
Regression tests: same seed must reproduce the same selected features
and key metrics within tolerance.

These tests require the real data files (data/UNSW_NB15_training-set.csv).
They are skipped if the data is not present.
"""
import os
import numpy as np
import pandas as pd
import pytest

DATA_AVAILABLE = (
    os.path.exists("data/UNSW_NB15_training-set.csv") and
    os.path.exists("data/UNSW_NB15_testing-set.csv")
)

pytestmark = pytest.mark.skipif(
    not DATA_AVAILABLE,
    reason="Real data files not available",
)


@pytest.fixture(scope="module")
def small_real_sample():
    """Return a stratified 1000-sample subset of the real training data."""
    from src.data.loader import load_split
    from src.data.loader import get_X_y
    from sklearn.model_selection import StratifiedShuffleSplit

    df = load_split("data/UNSW_NB15_training-set.csv")
    X, y = get_X_y(df, task="multiclass")

    sss = StratifiedShuffleSplit(n_splits=1, test_size=1000, random_state=42)
    _, idx = next(sss.split(X, y))
    return X.iloc[idx].reset_index(drop=True), y.values[idx]


def test_consensus_selection_reproducible(small_real_sample):
    """Same seed must produce the same selected feature list."""
    from src.features.engineer import FeatureEngineer
    from src.select.consensus import ConsensusSelector
    from src.utils.seeds import set_global_seed

    X, y = small_real_sample

    results = []
    for _ in range(2):
        set_global_seed(42)
        eng = FeatureEngineer()
        X_eng = eng.fit_transform(X)
        sel = ConsensusSelector(k_grid=[8, 12, 16], n_inner_folds=2, surrogate_n_estimators=20)
        result = sel.fit(X_eng, y)
        results.append(result.selected_names)

    assert results[0] == results[1], (
        f"Non-reproducible feature selection:\n  Run1: {results[0]}\n  Run2: {results[1]}"
    )


def test_no_100_percent_accuracy(small_real_sample):
    """No model should achieve >= 99% accuracy on the real data — that's leakage."""
    from src.features.engineer import FeatureEngineer
    from src.models.base_models import RandomForestModel
    from sklearn.metrics import accuracy_score
    from src.utils.seeds import set_global_seed

    set_global_seed(42)
    X, y = small_real_sample

    eng = FeatureEngineer()
    X_eng = eng.fit_transform(X)

    # Train on 70%, evaluate on 30%
    n = len(y)
    n_train = int(0.7 * n)
    X_tr, X_val = X_eng.iloc[:n_train], X_eng.iloc[n_train:]
    y_tr, y_val = y[:n_train], y[n_train:]

    rf = RandomForestModel(n_estimators=50)
    rf.fit(X_tr, y_tr)
    y_pred = rf.predict(X_val)
    acc = accuracy_score(y_val, y_pred)

    assert acc < 0.99, (
        f"Suspiciously high accuracy: {acc:.4f}. "
        "This likely indicates a data-leakage bug."
    )
    # Also check it's not trivially bad
    assert acc > 0.5, f"Accuracy too low: {acc:.4f}"
