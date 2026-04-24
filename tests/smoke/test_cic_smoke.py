"""
Smoke test for the CIC-IDS2017 adapter end-to-end.

Verifies that the pipeline can consume a synthetic CIC-shaped dataset using
the adapter plug-in architecture:

  - Adapter-driven feature engineering
  - Adapter-driven preprocessing (numeric_cols / categorical_cols threading)
  - Consensus selection on the engineered frame
  - Full EnsembleCombiner fit/predict with shape and probability-sum checks

No real CIC-IDS2017 data is required: we fabricate the 70-column CIC schema
with balanced labels from the 8-class vocabulary.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from src.data.adapters import get_adapter
from src.ensemble.combiner import EnsembleCombiner
from src.features.engineer import FeatureEngineer
from src.models.base_models import build_default_models
from src.select.consensus import ConsensusSelector
from src.utils.seeds import set_global_seed


N_PER_CLASS = 60   # balanced per-class so stratified splits work cleanly
CIC_KEY = "cic_ids2017"


@pytest.fixture(scope="module")
def cic_adapter():
    return get_adapter(CIC_KEY)


@pytest.fixture(scope="module")
def synthetic_cic_dataset(cic_adapter):
    """Generate a synthetic CIC-shaped frame covering all 8 classes."""
    set_global_seed(42)
    rng = np.random.default_rng(42)
    classes = cic_adapter.class_names
    n = N_PER_CLASS * len(classes)

    data = {col: rng.random(n) * 100 for col in cic_adapter.schema.numeric_cols}
    labels_raw = np.repeat(classes, N_PER_CLASS)
    rng.shuffle(labels_raw)
    data["attack_cat"] = labels_raw
    data["label"] = (labels_raw != "BENIGN").astype(int)

    df = pd.DataFrame(data)
    cic_adapter.schema.validate(df)
    return df


def _get_X_y(df, cic_adapter):
    """Strip label columns using the adapter schema."""
    cols = [c for c in cic_adapter.schema.feature_cols if c in df.columns]
    X = df[cols].copy()
    label_map = {lbl: i for i, lbl in enumerate(cic_adapter.class_names)}
    y = df["attack_cat"].map(label_map).astype(int)
    return X, y


def test_cic_feature_engineering(cic_adapter, synthetic_cic_dataset):
    X, _ = _get_X_y(synthetic_cic_dataset, cic_adapter)
    eng = FeatureEngineer(registry=cic_adapter.feature_registry)
    X_eng = eng.fit_transform(X)
    assert X_eng.shape[0] == X.shape[0]
    assert X_eng.shape[1] > X.shape[1]
    # Every engineered feature's name should appear
    for name in cic_adapter.engineered_feature_names:
        assert name in X_eng.columns
    # No NaNs in numeric output
    assert not X_eng.select_dtypes(include="number").isna().any().any()


def test_cic_preprocessor_factory_accepts_adapter_cols(cic_adapter, synthetic_cic_dataset):
    """The tree preprocessor must consume adapter-supplied column lists."""
    from src.models.preprocessing import get_preprocessor

    X, _ = _get_X_y(synthetic_cic_dataset, cic_adapter)
    pre = get_preprocessor(
        "tree",
        numeric_cols=cic_adapter.schema.numeric_cols,
        categorical_cols=cic_adapter.schema.categorical_cols,
    )
    Xt = pre.fit_transform(X)
    assert Xt.shape[0] == X.shape[0]
    # All CIC numeric columns survive as output features
    out_names = set(pre.feature_names_out)
    assert set(cic_adapter.schema.numeric_cols) <= out_names


def test_cic_full_ensemble_smoke(cic_adapter, synthetic_cic_dataset):
    X, y = _get_X_y(synthetic_cic_dataset, cic_adapter)
    y_arr = y.values

    eng = FeatureEngineer(registry=cic_adapter.feature_registry)
    X_eng = eng.fit_transform(X)

    sel = ConsensusSelector(k_grid=[5, 8], n_inner_folds=2, surrogate_n_estimators=5)
    result = sel.fit(X_eng, y_arr)
    X_sel = X_eng[result.selected_names]

    models = build_default_models(
        numeric_cols=cic_adapter.schema.numeric_cols,
        categorical_cols=cic_adapter.schema.categorical_cols,
    )
    combiner = EnsembleCombiner(
        base_models=models,
        n_classes=cic_adapter.n_classes,
        n_oof_splits=2,
        gate_model="ridge",
    )
    with warnings.catch_warnings(record=True):
        combiner.fit(X_sel, y_arr)
        all_probas = combiner.predict_proba_all(X_sel)

    for method, proba in all_probas.items():
        assert proba.shape == (len(y_arr), cic_adapter.n_classes), (
            f"{method}: wrong shape {proba.shape}"
        )
        assert not np.isnan(proba).any(), f"{method}: NaN in output"
        row_sums = proba.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-4), (
            f"{method}: rows don't sum to 1"
        )


def test_cic_loader_stratified_split_on_single_path(tmp_path, cic_adapter):
    """When train_path == test_path, the loader must produce a 70/30 split,
    consolidate raw sub-labels (e.g. 'DoS Hulk' → 'DoS'), and honour the
    schema contract on both halves.
    """
    set_global_seed(42)
    rng = np.random.default_rng(0)
    # Raw labels include sub-variants — the loader should consolidate them.
    raw_label_pool = [
        "BENIGN", "BENIGN", "BENIGN",
        "Bot",
        "DDoS",
        "DoS GoldenEye", "DoS Hulk", "DoS Slowhttptest", "DoS slowloris",
        "Infiltration",
        "PortScan",
        "FTP-Patator", "SSH-Patator",
        "Web Attack – Brute Force", "Web Attack – XSS",
    ]
    n_per = 40
    raw_labels = np.repeat(raw_label_pool, n_per)
    n = len(raw_labels)
    rng.shuffle(raw_labels)

    # Build raw-shaped columns: " Flow Duration", " Total Fwd Packets", etc.,
    # plus " Label" — the loader rename map is keyed by these exact strings.
    from src.data.adapters.cic_ids2017 import _RAW_TO_CANONICAL
    raw_data = {}
    for raw_col in _RAW_TO_CANONICAL:
        if raw_col == " label":
            continue
        raw_data[raw_col] = rng.random(n) * 100
    raw_data[" Label"] = raw_labels  # loader lower-cases headers first
    raw_df = pd.DataFrame(raw_data)

    csv_path = tmp_path / "cic_synthetic.csv"
    raw_df.to_csv(csv_path, index=False)

    from src.data.adapters.cic_ids2017 import load_official_splits
    X_tr, y_tr, X_te, y_te = load_official_splits(csv_path, csv_path, task="multiclass")

    total = len(X_tr) + len(X_te)
    assert total == n
    assert abs(len(X_tr) / total - 0.70) < 0.02
    # DoS / Web Attack / Brute Force / Port Scan sub-variants must collapse
    # so that the 8 canonical classes all appear in both splits.
    assert y_tr.nunique() == cic_adapter.n_classes
    assert y_te.nunique() == cic_adapter.n_classes
