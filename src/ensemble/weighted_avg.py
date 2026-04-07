"""
Calibrated weighted-average path.

Computes per-model weights from inner-fold evidence (macro F1, log loss,
Brier score, calibration quality). Weights are normalised to sum to 1.

Weight formula (per model m):
  w_raw[m] = macro_f1[m] / (log_loss[m] + ε) * (1 - ece[m])
  w[m]     = softmax(w_raw)   (to ensure positivity and normalisability)

where all metrics are computed on OOF (inner-fold held-out) predictions.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import f1_score, log_loss

from src.eval.reliability import expected_calibration_error

logger = logging.getLogger(__name__)

_EPS = 1e-9


class WeightedAverager:
    """Compute p_weighted from a list of calibrated probability arrays.

    Parameters
    ----------
    model_names:
        Names of base models in the same order as the probability lists.
    """

    def __init__(self, model_names: List[str]) -> None:
        self.model_names = model_names
        self._weights: Optional[np.ndarray] = None
        self._weight_dict: Dict[str, float] = {}

    def fit(
        self,
        calibrated_oof_probas: List[np.ndarray],
        oof_true: np.ndarray,
        n_classes: int,
    ) -> "WeightedAverager":
        """Estimate per-model weights from OOF evidence.

        Parameters
        ----------
        calibrated_oof_probas:
            List of (n, n_classes) calibrated OOF probability arrays.
        oof_true:
            (n,) integer class labels.
        n_classes:
            Total number of classes.
        """
        raw_weights = []
        for i, (proba, mname) in enumerate(
            zip(calibrated_oof_probas, self.model_names)
        ):
            macro_f1 = f1_score(
                oof_true,
                np.argmax(proba, axis=1),
                average="macro",
                zero_division=0,
            )
            ll = log_loss(oof_true, proba, labels=list(range(n_classes)))
            ece = expected_calibration_error(
                oof_true, proba, n_bins=15
            )

            # Combined weight: high F1, low log-loss, low ECE
            w = (macro_f1 + _EPS) / (ll + _EPS) * max(1.0 - ece, _EPS)
            raw_weights.append(w)
            logger.info(
                "Model %s: macro_f1=%.4f log_loss=%.4f ECE=%.4f → raw_w=%.4f",
                mname, macro_f1, ll, ece, w,
            )

        # Softmax normalisation for positive, stable weights
        raw_arr = np.array(raw_weights, dtype=float)
        raw_arr -= raw_arr.max()   # numerical stability
        exp_w = np.exp(raw_arr)
        self._weights = exp_w / exp_w.sum()
        self._weight_dict = {
            n: float(w) for n, w in zip(self.model_names, self._weights)
        }
        logger.info("Normalised model weights: %s", self._weight_dict)
        return self

    def predict_proba(
        self,
        calibrated_probas: List[np.ndarray],
    ) -> np.ndarray:
        """Return (n, n_classes) weighted-average probability matrix."""
        if self._weights is None:
            raise RuntimeError("WeightedAverager not fitted.")
        weighted = np.zeros_like(calibrated_probas[0])
        for proba, w in zip(calibrated_probas, self._weights):
            weighted += w * proba
        # Re-normalise (should already sum to 1, but guard against float drift)
        row_sums = weighted.sum(axis=1, keepdims=True)
        return weighted / np.where(row_sums == 0, 1.0, row_sums)

    def get_weights(self) -> Dict[str, float]:
        return dict(self._weight_dict)
