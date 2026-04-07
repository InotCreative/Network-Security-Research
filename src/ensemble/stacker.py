"""
Meta-learner stacker (Path B of the ensemble).

Input  : rich meta-features built from calibrated OOF probability vectors
Output : class probabilities p_stack (n_samples, n_classes)

Default stacker: Multinomial Logistic Regression with L2 regularisation.
Fallback: Gradient Boosted stacker when the linear stacker systematically
underperforms (detected by a Brier score threshold on OOF evaluation).

The stacker is trained on OOF meta-features only — it never sees raw traffic
features or test data during training.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from src.utils.seeds import get_seed

logger = logging.getLogger(__name__)

# Brier improvement threshold to prefer GB stacker over linear
# Select GB stacker when it improves Brier by even a small margin —
# GB is more expressive on high-dimensional meta-features and the cost
# of an interpretability trade-off is acceptable when it clearly fits better.
_GB_IMPROVEMENT_THRESHOLD = 0.005


class MetaStacker:
    """Multinomial logistic regression stacker with optional GB fallback.

    Parameters
    ----------
    C:
        Inverse regularisation strength for logistic regression.
    allow_gb_fallback:
        If True, compare logistic vs. a small GB stacker on OOF CV and pick
        the better one (by multiclass Brier score).
    """

    def __init__(
        self,
        C: float = 0.1,
        allow_gb_fallback: bool = True,
    ) -> None:
        self.C = C
        self.allow_gb_fallback = allow_gb_fallback
        self._scaler = StandardScaler()
        self._stacker: Optional[LogisticRegression | GradientBoostingClassifier] = None
        self._chosen_method: str = "logistic"

    def fit(
        self,
        meta_X: np.ndarray,
        y: np.ndarray,
        n_classes: int,
    ) -> "MetaStacker":
        """Fit stacker on meta-feature matrix.

        Parameters
        ----------
        meta_X : (n, n_meta_features) — built by meta_features.build_meta_features
        y      : (n,) integer labels
        n_classes: total number of classes
        """
        meta_X_s = self._scaler.fit_transform(meta_X)

        lr = LogisticRegression(
            C=self.C,
            solver="lbfgs",
            max_iter=2000,
            class_weight="balanced",
            random_state=get_seed("meta_stacker_lr"),
        )

        if self.allow_gb_fallback:
            chosen = self._select_stacker(meta_X_s, y, lr)
        else:
            chosen = lr

        chosen.fit(meta_X_s, y)
        self._stacker = chosen
        logger.info("MetaStacker fitted. Method: %s", self._chosen_method)
        return self

    def predict_proba(self, meta_X: np.ndarray) -> np.ndarray:
        if self._stacker is None:
            raise RuntimeError("MetaStacker not fitted.")
        meta_X_s = self._scaler.transform(meta_X)
        return self._stacker.predict_proba(meta_X_s)

    @property
    def chosen_method(self) -> str:
        return self._chosen_method

    # ──────────────────────────────────────────────────────────────────────────

    def _select_stacker(
        self,
        meta_X_s: np.ndarray,
        y: np.ndarray,
        lr: LogisticRegression,
    ):
        """Return the better of LR or small GB using 3-fold Brier on OOF data."""
        from sklearn.metrics import brier_score_loss
        from sklearn.model_selection import StratifiedKFold

        gb = GradientBoostingClassifier(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.8,
            random_state=get_seed("meta_stacker_gb"),
        )

        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=get_seed("stacker_sel"))

        lr_brier = _cv_brier(lr, meta_X_s, y, skf)
        gb_brier = _cv_brier(gb, meta_X_s, y, skf)

        logger.info(
            "Stacker selection: LR Brier=%.4f  GB Brier=%.4f",
            lr_brier, gb_brier,
        )

        if gb_brier + _GB_IMPROVEMENT_THRESHOLD < lr_brier:
            self._chosen_method = "gradient_boosting"
            return gb
        else:
            self._chosen_method = "logistic"
            return lr


def _cv_brier(estimator, X, y, cv) -> float:
    """Multiclass Brier score via cross-validation.

    Handles the case where a fold's training set is missing rare classes —
    the estimator's classes_ may be a subset of the full label set.
    """
    from sklearn.base import clone
    from sklearn.metrics import brier_score_loss
    all_classes = np.unique(y)
    n_classes = len(all_classes)
    brierscores = []
    for tr_idx, val_idx in cv.split(X, y):
        est = clone(estimator)
        est.fit(X[tr_idx], y[tr_idx])
        proba_raw = est.predict_proba(X[val_idx])
        est_classes = est.classes_
        y_val = y[val_idx]

        # Expand proba to full n_classes columns
        proba = np.zeros((len(val_idx), n_classes))
        for col_idx, cls in enumerate(est_classes):
            global_idx = int(cls)
            if global_idx < n_classes:
                proba[:, global_idx] = proba_raw[:, col_idx]
        row_sums = proba.sum(axis=1, keepdims=True)
        proba = proba / np.where(row_sums == 0, 1.0, row_sums)

        bs = 0.0
        for c in range(n_classes):
            y_bin = (y_val == c).astype(float)
            bs += brier_score_loss(y_bin, proba[:, c])
        brierscores.append(bs / n_classes)
    return float(np.mean(brierscores))
