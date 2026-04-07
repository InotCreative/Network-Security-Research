"""
Disagreement-aware gating model.

The gate outputs a sample-level mixing coefficient β(x) ∈ [0, 1].
Input: disagreement / uncertainty meta-features (not raw traffic features).
Final combination: p_final = β * p_stack + (1 - β) * p_weighted.

When β = 1 → trust the stacker entirely.
When β = 0 → trust the weighted average entirely.

The gate is trained to minimise the multiclass Brier score of the combined
output. We frame this as a regression problem: the target β* for each sample
is the mixing coefficient that, applied to (p_stack, p_weighted), minimises
the squared error against the true label encoding.

β*(x) = argmin_{β∈[0,1]} || one_hot(y) - (β*p_s + (1-β)*p_w) ||²
      = clamp( <one_hot(y) - p_w, p_s - p_w> / ||p_s - p_w||² , 0, 1)

The gate is a logistic/tree regressor mapping meta-features → β*.
It stays lightweight to preserve interpretability (design doc §6.5).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from src.utils.seeds import get_seed

logger = logging.getLogger(__name__)

_EPS = 1e-9


def _optimal_beta(
    p_stack: np.ndarray,
    p_weighted: np.ndarray,
    y_true: np.ndarray,
    n_classes: int,
) -> np.ndarray:
    """Compute the per-sample optimal β* analytically.

    Minimises ||one_hot(y) - (β p_s + (1-β) p_w)||² w.r.t. β ∈ [0,1].
    """
    y_oh = np.eye(n_classes)[y_true]          # (n, C) one-hot
    diff = p_stack - p_weighted               # (n, C)
    residual = y_oh - p_weighted              # (n, C)

    num = (residual * diff).sum(axis=1)       # (n,)
    denom = (diff * diff).sum(axis=1)         # (n,)

    beta = np.where(denom > _EPS, num / denom, 0.5)
    return np.clip(beta, 0.0, 1.0)


class DisagreementGate:
    """Learn β(x) from disagreement meta-features.

    Parameters
    ----------
    model:
        'ridge'   — Ridge regression (default, interpretable linear gate).
        'tree'    — Decision tree regressor (max_depth=4, shallow).
        'auto'    — Select by validation MSE.
    """

    def __init__(self, model: str = "ridge") -> None:
        self.model = model
        self._scaler = StandardScaler()
        self._gate = None
        self._chosen_model: str = model

    def fit(
        self,
        meta_X: np.ndarray,
        p_stack: np.ndarray,
        p_weighted: np.ndarray,
        y_true: np.ndarray,
        n_classes: int,
    ) -> "DisagreementGate":
        """Fit the gate.

        Parameters
        ----------
        meta_X    : (n, n_meta) disagreement/uncertainty features
        p_stack   : (n, C) stacker probabilities on OOF data
        p_weighted: (n, C) weighted-average probabilities on OOF data
        y_true    : (n,) integer labels
        n_classes : int
        """
        beta_targets = _optimal_beta(p_stack, p_weighted, y_true, n_classes)

        meta_s = self._scaler.fit_transform(meta_X)

        if self.model == "auto":
            chosen_gate = self._select_gate(meta_s, beta_targets)
        elif self.model == "ridge":
            chosen_gate = Ridge(alpha=1.0)
        elif self.model == "tree":
            chosen_gate = DecisionTreeRegressor(
                max_depth=4,
                min_samples_leaf=50,
                random_state=get_seed("gate_tree"),
            )
        else:
            raise ValueError(f"Unknown gate model: {self.model!r}")

        chosen_gate.fit(meta_s, beta_targets)
        self._gate = chosen_gate
        logger.info(
            "DisagreementGate fitted. Model: %s. β* mean=%.3f std=%.3f",
            self._chosen_model,
            float(beta_targets.mean()),
            float(beta_targets.std()),
        )
        return self

    def predict_beta(self, meta_X: np.ndarray) -> np.ndarray:
        """Return per-sample β ∈ [0, 1]."""
        if self._gate is None:
            raise RuntimeError("DisagreementGate not fitted.")
        meta_s = self._scaler.transform(meta_X)
        beta = self._gate.predict(meta_s)
        return np.clip(beta, 0.0, 1.0)

    def combine(
        self,
        p_stack: np.ndarray,
        p_weighted: np.ndarray,
        meta_X: np.ndarray,
    ) -> np.ndarray:
        """Return p_final = β * p_stack + (1 - β) * p_weighted."""
        beta = self.predict_beta(meta_X).reshape(-1, 1)
        p_final = beta * p_stack + (1.0 - beta) * p_weighted
        row_sums = p_final.sum(axis=1, keepdims=True)
        return p_final / np.where(row_sums == 0, 1.0, row_sums)

    # ──────────────────────────────────────────────────────────────────────────

    def _select_gate(self, meta_s: np.ndarray, beta_targets: np.ndarray):
        from sklearn.model_selection import cross_val_score

        ridge = Ridge(alpha=1.0)
        tree = DecisionTreeRegressor(
            max_depth=4,
            min_samples_leaf=50,
            random_state=get_seed("gate_tree"),
        )
        ridge_mse = -cross_val_score(
            ridge, meta_s, beta_targets, cv=3,
            scoring="neg_mean_squared_error",
        ).mean()
        tree_mse = -cross_val_score(
            tree, meta_s, beta_targets, cv=3,
            scoring="neg_mean_squared_error",
        ).mean()
        logger.info("Gate selection: Ridge MSE=%.5f  Tree MSE=%.5f", ridge_mse, tree_mse)
        if tree_mse < ridge_mse * 0.98:
            self._chosen_model = "tree"
            return tree
        self._chosen_model = "ridge"
        return ridge
