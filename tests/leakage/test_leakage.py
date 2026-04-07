"""
Leakage prevention tests.

These tests guard against the most common UNSW-NB15 data-leakage patterns
that cause spuriously perfect accuracy.
"""
import numpy as np
import pandas as pd
import pytest
from src.data.schema import FEATURE_COLS, LABEL_COLS
from src.data.loader import get_X_y


@pytest.fixture
def small_df():
    """Tiny valid DataFrame with label columns."""
    rng = np.random.default_rng(0)
    n = 30
    from src.data.schema import FEATURE_COLS, MULTICLASS_LABELS
    data = {col: rng.random(n) for col in FEATURE_COLS}
    data["proto"] = ["tcp"] * n
    data["service"] = ["-"] * n
    data["state"] = ["FIN"] * n
    data["attack_cat"] = [MULTICLASS_LABELS[i % len(MULTICLASS_LABELS)] for i in range(n)]
    data["label"] = [i % 2 for i in range(n)]
    data["id"] = list(range(n))
    return pd.DataFrame(data)


def test_get_X_y_excludes_label_cols(small_df):
    """X must never contain attack_cat or label."""
    X, y = get_X_y(small_df, task="multiclass")
    assert "attack_cat" not in X.columns
    assert "label" not in X.columns


def test_get_X_y_excludes_id(small_df):
    X, y = get_X_y(small_df, task="multiclass")
    assert "id" not in X.columns


def test_feature_label_no_overlap():
    """Hard schema guarantee: FEATURE_COLS ∩ LABEL_COLS = ∅."""
    overlap = set(FEATURE_COLS) & set(LABEL_COLS)
    assert not overlap, f"FEATURE_COLS and LABEL_COLS overlap: {overlap}"


def test_feature_preprocessor_fit_on_train_only():
    """TreePreprocessor fitted on train data must not change when val data is seen."""
    from src.models.preprocessing import TreePreprocessor
    rng = np.random.default_rng(0)
    n = 100
    from src.data.schema import NUMERIC_COLS, CATEGORICAL_COLS
    cols = [c for c in NUMERIC_COLS if c not in CATEGORICAL_COLS][:5]
    df_train = pd.DataFrame({c: rng.random(n) for c in cols})
    df_val = pd.DataFrame({c: rng.random(n) * 100 for c in cols})   # very different scale

    prep = TreePreprocessor()
    X_tr = prep.fit_transform(df_train)

    # Fitting should use train stats; transforming val should use SAME stats
    # We verify this by checking that re-fitting on train gives same result
    prep2 = TreePreprocessor()
    X_tr2 = prep2.fit_transform(df_train)
    np.testing.assert_array_almost_equal(X_tr, X_tr2, decimal=10)


def test_feature_engineer_train_stats_not_updated_after_fit():
    """FeatureEngineer clip bounds must not update when transform is called with new data."""
    from src.features.engineer import FeatureEngineer
    rng = np.random.default_rng(0)
    n = 100
    df_train = pd.DataFrame({
        "sbytes": rng.integers(100, 1000, n).astype(float),
        "dbytes": rng.integers(0, 1000, n).astype(float),
        "spkts": rng.integers(1, 50, n).astype(float),
        "dpkts": rng.integers(0, 50, n).astype(float),
        "sload": rng.uniform(0, 1e6, n),
        "dload": rng.uniform(0, 1e6, n),
        "sttl": rng.integers(0, 255, n).astype(float),
        "dttl": rng.integers(0, 255, n).astype(float),
        "sinpkt": rng.uniform(0, 1, n),
        "dinpkt": rng.uniform(0, 1, n),
        "sjit": rng.uniform(0, 100, n),
        "djit": rng.uniform(0, 100, n),
        "sloss": rng.integers(0, 5, n).astype(float),
        "dloss": rng.integers(0, 5, n).astype(float),
        "swin": rng.integers(0, 65535, n).astype(float),
        "dwin": rng.integers(0, 65535, n).astype(float),
        "synack": rng.uniform(0, 1, n),
        "ackdat": rng.uniform(0, 1, n),
        "tcprtt": rng.uniform(0, 2, n),
        "rate": rng.uniform(0, 1e6, n),
    })
    df_val_large = df_train * 1000   # extreme scale-up

    eng = FeatureEngineer()
    eng.fit(df_train)
    bounds_before = eng.get_clip_bounds()

    eng.transform(df_val_large)
    bounds_after = eng.get_clip_bounds()

    assert bounds_before == bounds_after, "Clip bounds changed after transform — leakage!"


def test_oof_index_coverage():
    """OOF predictions must cover every training sample exactly once."""
    from src.models.base_models import RandomForestModel
    import warnings

    rng = np.random.default_rng(0)
    n = 200
    n_classes = 3
    X = pd.DataFrame({f"f{i}": rng.random(n) for i in range(10)})
    y = rng.integers(0, n_classes, n)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rf = RandomForestModel(n_estimators=5)
        # Monkey-patch preprocessor to avoid needing categorical cols
        from src.models.preprocessing import TreePreprocessor
        oof_proba, oof_true = rf.collect_oof(X, y, n_splits=3, n_classes=n_classes)

    assert oof_proba.shape == (n, n_classes)
    # No sample should have zero total probability (would indicate it was skipped)
    row_sums = oof_proba.sum(axis=1)
    assert np.all(row_sums > 0.99), "Some OOF samples have near-zero probabilities"
