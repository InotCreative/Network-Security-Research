"""
Unit tests for the dataset adapter plug-in architecture.

Verifies:
  - The adapter registry lists both datasets and lazy-imports work.
  - Each adapter exposes a valid DatasetSchema, feature registry, and loader.
  - Schema validate() catches missing columns and unexpected label values.
  - The UNSW-NB15 adapter matches the legacy module-level globals so that
    backward compatibility is preserved.
  - The CIC-IDS2017 adapter's label consolidation and column normalisation
    behave as documented.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.data.adapters import available_datasets, get_adapter
from src.data.adapters.base import AdapterBundle, DatasetSchema


def test_registry_lists_both_datasets():
    names = available_datasets()
    assert "unsw_nb15" in names
    assert "cic_ids2017" in names


def test_unknown_dataset_raises():
    with pytest.raises(KeyError, match="Unknown dataset"):
        get_adapter("not_a_real_dataset")


def test_unsw_adapter_bundle_shape():
    adapter = get_adapter("unsw_nb15")
    assert isinstance(adapter, AdapterBundle)
    assert adapter.name == "unsw_nb15"
    assert adapter.schema.display_name == "UNSW-NB15"
    assert len(adapter.class_names) == 10
    assert adapter.n_classes == 10
    assert len(adapter.feature_registry) > 0
    # Every engineered feature has a non-empty name
    for name in adapter.engineered_feature_names:
        assert isinstance(name, str) and name


def test_unsw_adapter_matches_legacy_globals():
    """Adapter schema must mirror the legacy module-level UNSW globals."""
    from src.data.schema import (
        CATEGORICAL_COLS, MULTICLASS_LABELS, NUMERIC_COLS,
    )
    adapter = get_adapter("unsw_nb15")
    assert adapter.schema.numeric_cols == list(NUMERIC_COLS)
    assert adapter.schema.categorical_cols == list(CATEGORICAL_COLS)
    assert adapter.schema.multiclass_labels == list(MULTICLASS_LABELS)
    assert adapter.schema.multiclass_target_col == "attack_cat"
    assert adapter.schema.binary_target_col == "label"


def test_cic_adapter_bundle_shape():
    adapter = get_adapter("cic_ids2017")
    assert isinstance(adapter, AdapterBundle)
    assert adapter.name == "cic_ids2017"
    assert adapter.schema.display_name == "CIC-IDS2017"
    assert adapter.n_classes == 8
    assert "BENIGN" in adapter.class_names
    assert "DDoS" in adapter.class_names
    assert "Web Attack" in adapter.class_names
    # CIC has no low-cardinality categoricals
    assert adapter.schema.categorical_cols == []
    # Substantial numeric feature set
    assert len(adapter.schema.numeric_cols) >= 60
    assert len(adapter.feature_registry) >= 10


def test_cic_feature_registry_is_well_formed():
    adapter = get_adapter("cic_ids2017")
    expected_names = {
        "pkt_ratio", "byte_ratio", "pkt_size_asym", "iat_asym",
        "pkt_rate_asym", "bytes_per_pkt_total", "flag_density",
        "idle_active_ratio", "win_ratio", "down_up_ratio_norm",
    }
    registry_names = {f.name for f in adapter.feature_registry}
    assert expected_names <= registry_names
    # Every input column of every engineered feature is in the numeric schema.
    numeric_set = set(adapter.schema.numeric_cols)
    for spec in adapter.feature_registry:
        for col in spec.input_cols:
            assert col in numeric_set, (
                f"CIC feature {spec.name!r} references {col!r} which is not "
                f"in the numeric schema."
            )


def test_schema_validate_missing_columns_raises():
    schema = DatasetSchema(
        name="toy",
        display_name="Toy",
        numeric_cols=["a", "b"],
        categorical_cols=["c"],
        label_cols=["attack_cat", "label"],
        drop_cols=[],
        multiclass_labels=["X", "Y"],
        multiclass_target_col="attack_cat",
        binary_target_col="label",
    )
    df = pd.DataFrame({"a": [1.0], "attack_cat": ["X"], "label": [0]})
    with pytest.raises(ValueError, match="missing"):
        schema.validate(df)


def test_schema_validate_unexpected_label_raises():
    schema = DatasetSchema(
        name="toy", display_name="Toy",
        numeric_cols=["a"], categorical_cols=[],
        label_cols=["attack_cat", "label"], drop_cols=[],
        multiclass_labels=["X", "Y"],
        multiclass_target_col="attack_cat",
        binary_target_col="label",
    )
    df = pd.DataFrame({"a": [1.0], "attack_cat": ["Z"], "label": [0]})
    with pytest.raises(ValueError, match="unexpected"):
        schema.validate(df)


def test_cic_label_consolidation():
    """Raw sub-labels should collapse to the 8-class vocabulary."""
    from src.data.adapters.cic_ids2017 import _consolidate_labels, MULTICLASS_LABELS
    raw = pd.Series([
        "BENIGN",
        "DoS GoldenEye", "DoS Hulk", "DoS slowloris", "Heartbleed",
        "Web Attack – Brute Force", "Web Attack – XSS",
        "Web Attack - Brute Force",
        "FTP-Patator", "SSH-Patator",
        "PortScan",
        "DDoS", "Bot", "Infiltration",
    ])
    mapped = _consolidate_labels(raw)
    assert set(mapped.unique()) <= set(MULTICLASS_LABELS)
    assert (mapped[1:5] == "DoS").all()
    assert (mapped[5:8] == "Web Attack").all()
    assert (mapped[8:10] == "Brute Force").all()
    assert mapped.iloc[10] == "Port Scan"


def test_cic_label_consolidation_unknown_raises():
    from src.data.adapters.cic_ids2017 import _consolidate_labels
    raw = pd.Series(["BENIGN", "Nonexistent Attack"])
    with pytest.raises(ValueError, match="Unknown CIC-IDS2017 label"):
        _consolidate_labels(raw)


def test_cic_column_normalisation():
    """Raw CIC headers (with leading space / inconsistent case) map cleanly."""
    from src.data.adapters.cic_ids2017 import _normalise_columns, _RAW_TO_CANONICAL
    df = pd.DataFrame(
        {
            " Flow Duration": [1.0],
            " Total Fwd Packets": [2],
            " Label": ["BENIGN"],
        }
    )
    out = _normalise_columns(df)
    # The rename uses lower-cased versions of the raw keys
    assert "flow_duration" in out.columns
    assert "tot_fwd_pkts" in out.columns
    assert "label_raw" in out.columns
    # Sanity: the rename dict is keyed by pre-lowered names
    assert all(k == k.lower() for k in _RAW_TO_CANONICAL)
