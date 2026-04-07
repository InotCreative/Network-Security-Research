"""Schema validation tests."""
import pytest
import pandas as pd
import numpy as np
from src.data.schema import SCHEMA, FEATURE_COLS, LABEL_COLS, MULTICLASS_LABELS


def test_feature_label_no_overlap():
    """CRITICAL: feature set and label set must never overlap."""
    overlap = set(FEATURE_COLS) & set(LABEL_COLS) & {"id"}
    assert not overlap, f"Schema bug — FEATURE_COLS overlaps LABEL_COLS: {overlap}"


def test_label_cols_not_in_feature_cols():
    for lbl in LABEL_COLS:
        assert lbl not in FEATURE_COLS, f"'{lbl}' must not appear in FEATURE_COLS"


def test_id_not_in_feature_cols():
    assert "id" not in FEATURE_COLS


def test_schema_validate_accepts_correct_df(sample_df):
    SCHEMA.validate(sample_df)   # should not raise


def test_schema_validate_rejects_bad_attack_cat(sample_df):
    bad_df = sample_df.copy()
    bad_df.loc[0, "attack_cat"] = "HackerAttack"
    with pytest.raises(ValueError, match="unexpected attack_cat"):
        SCHEMA.validate(bad_df)


def test_schema_validate_rejects_bad_label(sample_df):
    bad_df = sample_df.copy()
    bad_df["label"] = bad_df["label"].astype(object)
    bad_df.loc[0, "label"] = 99
    with pytest.raises(ValueError, match="unexpected label"):
        SCHEMA.validate(bad_df)


@pytest.fixture
def sample_df():
    """Minimal valid DataFrame matching the UNSW-NB15 schema."""
    from src.data.schema import FEATURE_COLS
    rng = np.random.default_rng(0)
    n = 20
    data = {col: rng.random(n) for col in FEATURE_COLS}
    data["proto"] = ["tcp"] * n
    data["service"] = ["-"] * n
    data["state"] = ["FIN"] * n
    # Remove duplicates from previous loop for categoricals
    for col in ["proto", "service", "state"]:
        if col in data:
            pass  # already set above
    data["attack_cat"] = [MULTICLASS_LABELS[i % len(MULTICLASS_LABELS)] for i in range(n)]
    data["label"] = [i % 2 for i in range(n)]
    data["id"] = list(range(n))
    return pd.DataFrame(data)
