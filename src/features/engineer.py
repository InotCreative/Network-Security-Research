"""
Feature engineering pipeline.

FeatureEngineer is a stateful transformer (fit/transform pattern) that:
  1. Computes all registered engineered features from raw input columns.
  2. Applies clipping thresholds fit on the TRAINING fold only.
  3. Appends new columns to the input DataFrame.
  4. Serialises clip bounds for auditability.

Never call transform without a prior fit — it will raise.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.features.registry import FEATURE_REGISTRY, ENGINEERED_FEATURE_NAMES, FeatureSpec

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Stateful feature engineering transformer.

    Parameters
    ----------
    feature_names:
        Subset of registry feature names to compute. Default: all of `registry`.
    registry:
        Optional list of FeatureSpec to use. Defaults to the UNSW-NB15 global
        `FEATURE_REGISTRY` for backward compatibility. Pass
        `adapter.feature_registry` when running on a different dataset.
    """

    def __init__(
        self,
        feature_names: Optional[List[str]] = None,
        registry: Optional[List[FeatureSpec]] = None,
    ) -> None:
        if registry is None:
            registry = FEATURE_REGISTRY
        available_names = [spec.name for spec in registry]
        if feature_names is None:
            feature_names = list(available_names)
        valid = set(available_names)
        bad = set(feature_names) - valid
        if bad:
            raise ValueError(f"Unknown engineered features: {bad}")
        self.feature_names = list(feature_names)
        self._specs: List[FeatureSpec] = [
            spec for spec in registry if spec.name in set(feature_names)
        ]
        # Filled by fit()
        self._clip_bounds: Dict[str, Tuple[float, float]] = {}
        self._is_fitted = False

    # ──────────────────────────────────────────────────────────────────────────
    # sklearn-style interface
    # ──────────────────────────────────────────────────────────────────────────

    def fit(self, X: pd.DataFrame) -> "FeatureEngineer":
        """Compute clip bounds from training data X."""
        self._clip_bounds = {}
        raw = self._compute_raw(X)
        for spec in self._specs:
            series = raw[spec.name]
            if spec.clipping is None:
                pass
            elif spec.clipping[0] == "clip_to_01":
                self._clip_bounds[spec.name] = (0.0, 1.0)
            elif spec.clipping[0] == "percentile":
                lo_q, hi_q = spec.clipping[1], spec.clipping[2]
                lo = float(np.nanpercentile(series, lo_q))
                hi = float(np.nanpercentile(series, hi_q))
                self._clip_bounds[spec.name] = (lo, hi)
            else:
                raise ValueError(
                    f"Unknown clipping policy: {spec.clipping} for {spec.name}"
                )
        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Append engineered features to X and return the augmented DataFrame."""
        if not self._is_fitted:
            raise RuntimeError("FeatureEngineer.fit() must be called before transform().")

        out = X.copy()
        raw = self._compute_raw(X)

        for spec in self._specs:
            series = raw[spec.name].astype(float)
            if spec.name in self._clip_bounds:
                lo, hi = self._clip_bounds[spec.name]
                series = series.clip(lower=lo, upper=hi)
            series = series.fillna(0.0)   # NaN → 0 after clipping
            out[spec.name] = series

        return out

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)

    # ──────────────────────────────────────────────────────────────────────────
    # Auditability
    # ──────────────────────────────────────────────────────────────────────────

    def get_clip_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Return fitted clipping bounds for artifact serialisation."""
        return dict(self._clip_bounds)

    def feature_manifest(self) -> List[dict]:
        """Machine-readable feature manifest for paper appendix."""
        rows = []
        for spec in self._specs:
            rows.append({
                "name": spec.name,
                "display_name": spec.display_name,
                "formula_desc": spec.formula_desc,
                "input_cols": list(spec.input_cols),
                "category": spec.category,
                "clipping_policy": spec.clipping,
                "clip_bounds_fitted": self._clip_bounds.get(spec.name),
            })
        return rows

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _compute_raw(self, X: pd.DataFrame) -> pd.DataFrame:
        """Compute all engineered features (raw, unclipped) from X."""
        raw: Dict[str, pd.Series] = {}
        for spec in self._specs:
            missing_inputs = set(spec.input_cols) - set(X.columns)
            if missing_inputs:
                raise ValueError(
                    f"Feature '{spec.name}' requires columns {missing_inputs} "
                    f"that are absent from the input."
                )
            try:
                raw[spec.name] = spec.compute(X)
            except Exception as exc:
                raise RuntimeError(
                    f"Error computing feature '{spec.name}': {exc}"
                ) from exc
        return pd.DataFrame(raw, index=X.index)
