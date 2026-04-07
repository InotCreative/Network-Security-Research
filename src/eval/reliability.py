"""Expected Calibration Error and reliability diagram data."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def expected_calibration_error(
    y_true: np.ndarray,
    proba: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Multiclass ECE via the maximum-class confidence binning strategy.

    Bins samples by their max predicted probability, then computes the
    weighted mean |accuracy - confidence| across bins.
    """
    confidence = proba.max(axis=1)
    predictions = np.argmax(proba, axis=1)
    correctness = (predictions == y_true).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confidence >= lo) & (confidence < hi)
        if i == n_bins - 1:
            mask = (confidence >= lo) & (confidence <= hi)
        n_bin = mask.sum()
        if n_bin == 0:
            continue
        acc_bin = correctness[mask].mean()
        conf_bin = confidence[mask].mean()
        ece += (n_bin / n) * abs(acc_bin - conf_bin)

    return float(ece)


def reliability_diagram_data(
    y_true: np.ndarray,
    proba: np.ndarray,
    n_bins: int = 15,
) -> dict:
    """Return data needed to draw a reliability diagram."""
    confidence = proba.max(axis=1)
    predictions = np.argmax(proba, axis=1)
    correctness = (predictions == y_true).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centres = []
    bin_accs = []
    bin_confs = []
    bin_counts = []

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confidence >= lo) & (confidence < hi)
        if i == n_bins - 1:
            mask = (confidence >= lo) & (confidence <= hi)
        n_bin = int(mask.sum())
        if n_bin == 0:
            continue
        bin_centres.append(float((lo + hi) / 2))
        bin_accs.append(float(correctness[mask].mean()))
        bin_confs.append(float(confidence[mask].mean()))
        bin_counts.append(n_bin)

    return {
        "bin_centres": bin_centres,
        "bin_accuracies": bin_accs,
        "bin_confidences": bin_confs,
        "bin_counts": bin_counts,
        "ece": expected_calibration_error(y_true, proba, n_bins),
    }
