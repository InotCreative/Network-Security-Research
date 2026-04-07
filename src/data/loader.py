"""
Official UNSW-NB15 data loader.

Reads the official train/test splits, enforces the schema contract, and returns
clean DataFrames with:
  - BOM stripped from the 'id' column name
  - attack_cat values stripped of accidental whitespace
  - Explicit separation of X (features), y_mc (multiclass), y_bin (binary)
  - is_ftp_login, ct_ftp_cmd treated as Int64 (observed values: 0, 1, 2)

NOTE: The official splits deliberately exclude raw IP addresses, ports, and
timestamps (srcip, dstip, sport, dsport, Stime, Ltime). We never add them back.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from src.data.schema import DTYPE_MAP, FEATURE_COLS, LABEL_COLS, SCHEMA

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Paths (relative to repo root or configurable)
# ──────────────────────────────────────────────────────────────────────────────

_DEFAULT_TRAIN = Path("data/UNSW_NB15_training-set.csv")
_DEFAULT_TEST = Path("data/UNSW_NB15_testing-set.csv")


def load_split(
    path: Path | str,
    *,
    validate: bool = True,
) -> pd.DataFrame:
    """Load one official split CSV into a validated DataFrame.

    Parameters
    ----------
    path:
        Path to the CSV file.
    validate:
        If True, run full schema validation before returning.

    Returns
    -------
    df : pd.DataFrame
        Index is reset. All columns present; dtypes coerced.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path.resolve()}")

    logger.info("Loading %s", path)
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)

    # Normalise column names: strip whitespace, lower-case
    df.columns = [c.strip() for c in df.columns]

    # Strip leading/trailing whitespace from string columns
    for col in ["attack_cat", "proto", "service", "state"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # Coerce dtypes (best-effort; failures become NaN then float64)
    for col, dtype in DTYPE_MAP.items():
        if col not in df.columns:
            continue
        try:
            df[col] = df[col].astype(dtype)
        except (ValueError, TypeError):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.reset_index(drop=True)

    if validate:
        SCHEMA.validate(df)

    logger.info("Loaded %d rows × %d cols from %s", len(df), df.shape[1], path)
    return df


def get_X_y(
    df: pd.DataFrame,
    task: str = "multiclass",
) -> Tuple[pd.DataFrame, pd.Series]:
    """Extract feature matrix and target from a loaded DataFrame.

    Parameters
    ----------
    df:
        Output of ``load_split``.
    task:
        ``'multiclass'`` → y is attack_cat (string labels).
        ``'binary'``     → y is label (0/1 integers).

    Returns
    -------
    X : pd.DataFrame  — shape (n, len(FEATURE_COLS)), no label columns, no id
    y : pd.Series     — integer-encoded labels (for compatibility with sklearn)
    """
    if task not in {"multiclass", "binary"}:
        raise ValueError(f"task must be 'multiclass' or 'binary', got {task!r}")

    # Guard: label columns must never appear in X
    feature_set = set(FEATURE_COLS)
    label_set = set(LABEL_COLS) | {"id"}
    overlap = feature_set & label_set
    if overlap:
        raise RuntimeError(
            f"BUG: FEATURE_COLS and LABEL_COLS overlap: {overlap}. "
            "This would be catastrophic leakage."
        )

    available_features = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available_features].copy()

    if task == "multiclass":
        from src.data.schema import MULTICLASS_LABELS
        y_raw = df["attack_cat"].astype(str).str.strip()
        label_map = {lbl: i for i, lbl in enumerate(MULTICLASS_LABELS)}
        y = y_raw.map(label_map)
        if y.isna().any():
            bad = y_raw[y.isna()].unique().tolist()
            raise ValueError(f"Unmapped attack_cat values: {bad}")
        y = y.astype(int)
    else:
        y = df["label"].astype(int)

    y.name = "target"
    return X, y


def load_official_splits(
    train_path: Path | str = _DEFAULT_TRAIN,
    test_path: Path | str = _DEFAULT_TEST,
    task: str = "multiclass",
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Load both official splits and return (X_train, y_train, X_test, y_test).

    The test set is LOCKED for final evaluation only. Never use it to make
    modelling decisions.
    """
    df_train = load_split(train_path)
    df_test = load_split(test_path)

    X_train, y_train = get_X_y(df_train, task=task)
    X_test, y_test = get_X_y(df_test, task=task)

    logger.info(
        "Train: %d samples | Test: %d samples | Task: %s",
        len(X_train), len(X_test), task,
    )
    _log_class_distribution("Train", y_train, task)
    _log_class_distribution("Test", y_test, task)

    return X_train, y_train, X_test, y_test


def _log_class_distribution(split_name: str, y: pd.Series, task: str) -> None:
    counts = y.value_counts().sort_index()
    total = len(y)
    if task == "multiclass":
        from src.data.schema import MULTICLASS_LABELS
        label_names = {i: n for i, n in enumerate(MULTICLASS_LABELS)}
    else:
        label_names = {0: "Normal", 1: "Attack"}
    parts = [
        f"{label_names.get(i, i)}={v}({100*v/total:.1f}%)"
        for i, v in counts.items()
    ]
    logger.info("%s class distribution: %s", split_name, " | ".join(parts))
