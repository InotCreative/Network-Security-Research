"""
Calibration wrappers for base model probability outputs.

CalibratedModel wraps a fitted BaseModel (or any object with predict_proba)
and applies either isotonic regression or Platt scaling (sigmoid) calibration
to each class independently.

Selection heuristic: if the training sample is >= min_samples_isotonic,
prefer isotonic regression (more flexible). Otherwise sigmoid. The winning
method is chosen by comparing Brier score on a validation split derived from
the OOF data — NO test data is ever seen here.

Multi-class calibration strategy: One-vs-Rest (OvR) calibration using sklearn's
CalibratedClassifierCV in 'prefit' mode on the OOF probability data.

Why not sklearn's CalibratedClassifierCV directly?
  We already have OOF probabilities. Rather than re-running CV inside
  calibration (which would double-nest the folds), we fit the calibrator
  directly on the OOF probabilities. This is the standard research approach
  and avoids any additional leakage.
"""

from __future__ import annotations

import logging
from typing import Literal, Optional, Tuple

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss

from src.utils.seeds import get_seed

logger = logging.getLogger(__name__)

_MIN_SAMPLES_ISOTONIC = 1000


class OvRCalibrator:
    """One-vs-Rest probability calibrator.

    Fits one isotonic/sigmoid calibrator per class on OOF probabilities.
    At inference time, calibrates each class independently then re-normalises.

    Parameters
    ----------
    method:
        'isotonic', 'sigmoid', or 'auto' (select by Brier score on val split).
    """

    def __init__(self, method: Literal["isotonic", "sigmoid", "auto"] = "auto") -> None:
        self.method = method
        self._calibrators: Optional[list] = None
        self._chosen_method: Optional[str] = None
        self._n_classes: int = 0

    def fit(
        self,
        oof_proba: np.ndarray,
        oof_true: np.ndarray,
        n_classes: int,
    ) -> "OvRCalibrator":
        """Fit calibrators on OOF (out-of-fold) probabilities.

        Parameters
        ----------
        oof_proba : (n, n_classes) — uncalibrated OOF probabilities
        oof_true  : (n,)           — integer class labels
        n_classes : int
        """
        self._n_classes = n_classes
        method = self._select_method(oof_proba, oof_true, n_classes)
        self._chosen_method = method

        self._calibrators = []
        for cls_idx in range(n_classes):
            y_bin = (oof_true == cls_idx).astype(int)
            p_cls = oof_proba[:, cls_idx]

            if method == "isotonic":
                cal = IsotonicRegression(out_of_bounds="clip")
            else:
                # Sigmoid (Platt scaling) as logistic regression on log-odds
                cal = _SigmoidCalibrator()

            cal.fit(p_cls.reshape(-1, 1) if method != "isotonic" else p_cls, y_bin)
            self._calibrators.append(cal)

        return self

    def predict_proba(self, proba: np.ndarray) -> np.ndarray:
        """Apply calibration and re-normalise to probability simplex."""
        if self._calibrators is None:
            raise RuntimeError("OvRCalibrator not fitted.")
        n = proba.shape[0]
        calibrated = np.zeros_like(proba)

        for cls_idx, cal in enumerate(self._calibrators):
            p = proba[:, cls_idx]
            if isinstance(cal, IsotonicRegression):
                calibrated[:, cls_idx] = cal.predict(p)
            else:
                calibrated[:, cls_idx] = cal.predict(p)

        # Clip to [0, 1] and re-normalise
        calibrated = np.clip(calibrated, 1e-9, 1.0)
        row_sums = calibrated.sum(axis=1, keepdims=True)
        return calibrated / row_sums

    @property
    def chosen_method(self) -> str:
        return self._chosen_method or "unknown"

    # ──────────────────────────────────────────────────────────────────────────

    def _select_method(
        self,
        oof_proba: np.ndarray,
        oof_true: np.ndarray,
        n_classes: int,
    ) -> str:
        if self.method != "auto":
            return self.method
        if len(oof_true) < _MIN_SAMPLES_ISOTONIC:
            return "sigmoid"

        # Compare on a small validation split drawn from OOF data
        n_val = min(5000, int(0.2 * len(oof_true)))
        try:
            idx_tr, idx_val = train_test_split(
                np.arange(len(oof_true)),
                test_size=n_val,
                stratify=oof_true,
                random_state=get_seed("calibration_selection"),
            )
        except ValueError:
            return "isotonic"

        brier_iso = self._eval_brier(
            oof_proba[idx_tr], oof_true[idx_tr],
            oof_proba[idx_val], oof_true[idx_val],
            n_classes, "isotonic",
        )
        brier_sig = self._eval_brier(
            oof_proba[idx_tr], oof_true[idx_tr],
            oof_proba[idx_val], oof_true[idx_val],
            n_classes, "sigmoid",
        )
        chosen = "isotonic" if brier_iso <= brier_sig else "sigmoid"
        logger.debug(
            "Calibration selection: isotonic_brier=%.4f sigmoid_brier=%.4f → %s",
            brier_iso, brier_sig, chosen,
        )
        return chosen

    def _eval_brier(
        self,
        p_tr, y_tr, p_val, y_val,
        n_classes: int,
        method: str,
    ) -> float:
        """Fit calibrators on p_tr/y_tr, evaluate multiclass Brier on p_val/y_val."""
        total_brier = 0.0
        for cls_idx in range(n_classes):
            y_bin_tr = (y_tr == cls_idx).astype(int)
            y_bin_val = (y_val == cls_idx).astype(int)

            if method == "isotonic":
                cal = IsotonicRegression(out_of_bounds="clip")
                cal.fit(p_tr[:, cls_idx], y_bin_tr)
                p_cal = cal.predict(p_val[:, cls_idx])
            else:
                cal = _SigmoidCalibrator()
                cal.fit(p_tr[:, cls_idx], y_bin_tr)
                p_cal = cal.predict(p_val[:, cls_idx])

            p_cal = np.clip(p_cal, 0.0, 1.0)
            if y_bin_val.sum() > 0:
                total_brier += brier_score_loss(y_bin_val, p_cal)

        return total_brier / n_classes


class _SigmoidCalibrator:
    """Platt scaling: logistic regression on raw probability scores."""

    def __init__(self) -> None:
        self._lr = LogisticRegression(C=1.0, max_iter=1000)

    def fit(self, p: np.ndarray, y: np.ndarray) -> "_SigmoidCalibrator":
        # Add small jitter to avoid singular matrix if p is constant
        p_jit = p.ravel() + np.random.default_rng(0).normal(0, 1e-7, size=len(p))
        self._lr.fit(p_jit.reshape(-1, 1), y)
        return self

    def predict(self, p: np.ndarray) -> np.ndarray:
        return self._lr.predict_proba(p.ravel().reshape(-1, 1))[:, 1]
