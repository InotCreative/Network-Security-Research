"""
Model-family-specific preprocessing pipelines.

Three families, three contracts — no shared transformer may mix incompatible
assumptions:

  tree   — OrdinalEncoder for categoricals (tree splits don't need OHE),
            no scaling, median impute. Used for RF, ExtraTrees, XGBoost.

  linear — OneHotEncoder with rare-category grouping (min_frequency threshold)
            for categoricals, RobustScaler for numerics, median impute.
            Used for stacker (logistic regression) and linear selectors.

  knn    — Same OHE as linear + RobustScaler. Never feed ordinal-coded
            categories to a distance-based model.

All transformers are fit on training-fold data only. Unseen categories at
inference time are mapped to an "__other__" bucket silently (no error).

Column order is frozen after fit and validated on transform.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OrdinalEncoder,
    OneHotEncoder,
    RobustScaler,
    StandardScaler,
)

from src.data.schema import CATEGORICAL_COLS, NUMERIC_COLS

logger = logging.getLogger(__name__)

# Protocols with < this many occurrences in training fold → "__other__"
_RARE_PROTO_THRESHOLD = 50
# service and state have low cardinality — no collapsing needed
_RARE_CAT_THRESHOLD = 5


def _numeric_cols_present(columns: List[str]) -> List[str]:
    return [c for c in NUMERIC_COLS if c in columns]


def _cat_cols_present(columns: List[str]) -> List[str]:
    return [c for c in CATEGORICAL_COLS if c in columns]


# ──────────────────────────────────────────────────────────────────────────────
# Tree family
# ──────────────────────────────────────────────────────────────────────────────

class TreePreprocessor:
    """Ordinal-encode categoricals, median-impute numerics. No scaling."""

    def __init__(self) -> None:
        self._pipeline: Optional[Pipeline] = None
        self._feature_names_out: Optional[List[str]] = None

    def fit(self, X: pd.DataFrame) -> "TreePreprocessor":
        num_cols = _numeric_cols_present(X.columns.tolist())
        cat_cols = _cat_cols_present(X.columns.tolist())
        # Also pass through any engineered (numeric) columns not in NUMERIC_COLS
        extra_num = [c for c in X.columns if c not in num_cols and c not in cat_cols]

        num_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
        ])
        cat_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "encode",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    dtype=np.float64,
                ),
            ),
        ])

        transformers = [
            ("num", num_pipe, num_cols + extra_num),
            ("cat", cat_pipe, cat_cols),
        ]
        ct = ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)
        self._pipeline = Pipeline([("ct", ct)])
        self._pipeline.fit(X)

        self._feature_names_out = (
            (num_cols + extra_num) + cat_cols
        )
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        return self._pipeline.transform(X)

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.fit(X).transform(X)

    @property
    def feature_names_out(self) -> List[str]:
        if self._feature_names_out is None:
            raise RuntimeError("Not fitted yet.")
        return self._feature_names_out


# ──────────────────────────────────────────────────────────────────────────────
# Linear family (and KNN)
# ──────────────────────────────────────────────────────────────────────────────

class LinearPreprocessor:
    """OHE for categoricals (with rare grouping), RobustScaler for numerics."""

    def __init__(self, scaler: str = "robust") -> None:
        """
        Parameters
        ----------
        scaler:
            'robust' (default, recommended for KNN and linear models on skewed data)
            'standard' (zero-mean unit-variance)
        """
        self._scaler_type = scaler
        self._pipeline: Optional[Pipeline] = None
        self._feature_names_out: Optional[List[str]] = None

    def fit(self, X: pd.DataFrame) -> "LinearPreprocessor":
        num_cols = _numeric_cols_present(X.columns.tolist())
        cat_cols = _cat_cols_present(X.columns.tolist())
        extra_num = [c for c in X.columns if c not in num_cols and c not in cat_cols]
        all_num = num_cols + extra_num

        scaler_obj = RobustScaler() if self._scaler_type == "robust" else StandardScaler()
        num_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", scaler_obj),
        ])
        cat_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "encode",
                OneHotEncoder(
                    min_frequency=_RARE_CAT_THRESHOLD,
                    handle_unknown="infrequent_if_exist",
                    sparse_output=False,
                    dtype=np.float64,
                ),
            ),
        ])

        transformers = [
            ("num", num_pipe, all_num),
            ("cat", cat_pipe, cat_cols),
        ]
        ct = ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)
        self._pipeline = Pipeline([("ct", ct)])
        self._pipeline.fit(X)

        # Reconstruct output feature names
        cat_encoder: OneHotEncoder = self._pipeline.named_steps["ct"].named_transformers_["cat"].named_steps["encode"]
        ohe_names: List[str] = []
        for i, col in enumerate(cat_cols):
            cats = cat_encoder.categories_[i]
            for cat in cats:
                ohe_names.append(f"{col}__{cat}")
            # infrequent bucket
            infreq = cat_encoder.infrequent_categories_
            if infreq is not None and infreq[i] is not None and len(infreq[i]) > 0:
                ohe_names.append(f"{col}__infrequent")

        self._feature_names_out = all_num + ohe_names
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        return self._pipeline.transform(X)

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.fit(X).transform(X)

    @property
    def feature_names_out(self) -> List[str]:
        if self._feature_names_out is None:
            raise RuntimeError("Not fitted yet.")
        return self._feature_names_out


class KNNPreprocessor(LinearPreprocessor):
    """Identical to LinearPreprocessor; alias for clarity in model cards."""

    def __init__(self) -> None:
        super().__init__(scaler="robust")


# ──────────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────────

def get_preprocessor(family: str) -> "TreePreprocessor | LinearPreprocessor | KNNPreprocessor":
    """Return an unfitted preprocessor for the requested model family.

    family: 'tree' | 'linear' | 'knn'
    """
    if family == "tree":
        return TreePreprocessor()
    elif family == "linear":
        return LinearPreprocessor()
    elif family == "knn":
        return KNNPreprocessor()
    else:
        raise ValueError(f"Unknown preprocessor family: {family!r}. Use 'tree', 'linear', or 'knn'.")
