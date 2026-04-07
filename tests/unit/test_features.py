"""Unit tests for the feature engineering registry and FeatureEngineer."""
import numpy as np
import pandas as pd
import pytest
from src.features.registry import FEATURE_REGISTRY, ENGINEERED_FEATURE_NAMES
from src.features.engineer import FeatureEngineer


@pytest.fixture
def sample_flow_df():
    rng = np.random.default_rng(42)
    n = 100
    return pd.DataFrame({
        "sbytes": rng.integers(100, 10000, n).astype(float),
        "dbytes": rng.integers(0, 10000, n).astype(float),
        "spkts": rng.integers(1, 100, n).astype(float),
        "dpkts": rng.integers(0, 100, n).astype(float),
        "sload": rng.uniform(0, 1e7, n),
        "dload": rng.uniform(0, 1e7, n),
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


def test_all_features_in_registry():
    """Every registered feature has a unique name."""
    names = [s.name for s in FEATURE_REGISTRY]
    assert len(names) == len(set(names)), "Duplicate feature names in registry"


def test_feature_engineer_fit_transform(sample_flow_df):
    eng = FeatureEngineer()
    out = eng.fit_transform(sample_flow_df)
    for name in ENGINEERED_FEATURE_NAMES:
        assert name in out.columns, f"Missing engineered feature: {name}"


def test_ratio_features_in_0_1(sample_flow_df):
    """Ratio features should be in [0, 1] after clipping."""
    eng = FeatureEngineer()
    out = eng.fit_transform(sample_flow_df)
    ratio_features = ["byte_ratio", "pkt_ratio", "load_ratio", "loss_rate",
                      "src_loss_rate", "dst_loss_rate", "tcp_setup_ratio", "win_ratio"]
    for feat in ratio_features:
        if feat in out.columns:
            vals = out[feat].dropna()
            assert vals.min() >= -1e-9, f"{feat} min below 0: {vals.min()}"
            assert vals.max() <= 1 + 1e-9, f"{feat} max above 1: {vals.max()}"


def test_no_nan_after_transform(sample_flow_df):
    eng = FeatureEngineer()
    out = eng.fit_transform(sample_flow_df)
    eng_cols = [c for c in ENGINEERED_FEATURE_NAMES if c in out.columns]
    assert not out[eng_cols].isna().any().any(), "NaN found in engineered features"


def test_transform_without_fit_raises(sample_flow_df):
    eng = FeatureEngineer()
    with pytest.raises(RuntimeError, match="fit"):
        eng.transform(sample_flow_df)


def test_fit_uses_train_stats_only(sample_flow_df):
    """clip_bounds should be identical whether transform sees the same or different data."""
    eng = FeatureEngineer()
    eng.fit(sample_flow_df)
    bounds_before = eng.get_clip_bounds()

    # Transform a shifted version — bounds must NOT change
    shifted = sample_flow_df * 10
    eng.transform(shifted)
    bounds_after = eng.get_clip_bounds()
    assert bounds_before == bounds_after


def test_feature_manifest_structure(sample_flow_df):
    eng = FeatureEngineer()
    eng.fit(sample_flow_df)
    manifest = eng.feature_manifest()
    assert isinstance(manifest, list)
    assert len(manifest) == len(ENGINEERED_FEATURE_NAMES)
    for entry in manifest:
        for key in ["name", "display_name", "formula_desc", "input_cols", "category"]:
            assert key in entry, f"Manifest entry missing key: {key}"
