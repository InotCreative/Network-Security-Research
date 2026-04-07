"""
All evaluation metrics used in the paper.

Primary task: multiclass on UNSW-NB15 (10 classes).
  - Macro F1 (primary metric)
  - Weighted F1
  - Per-class recall / precision / F1
  - Multiclass log loss
  - Multiclass Brier score
  - ECE (Expected Calibration Error)
  - Macro OvR ROC-AUC
  - Confusion matrix

Secondary task: binary detection.
  - Accuracy, F1, Precision, Recall, AUC-ROC, AUC-PR

All functions accept (y_true: array, proba: array, labels: List[str]) and
return a flat dictionary for easy serialisation.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.eval.reliability import expected_calibration_error


def multiclass_metrics(
    y_true: np.ndarray,
    proba: np.ndarray,
    labels: List[str],
    prefix: str = "",
) -> Dict:
    """Compute the full multiclass metric suite."""
    y_pred = np.argmax(proba, axis=1)
    n_classes = len(labels)

    # Guard: if proba has fewer columns than n_classes, zero-pad
    if proba.shape[1] < n_classes:
        pad = np.zeros((proba.shape[0], n_classes - proba.shape[1]))
        proba = np.hstack([proba, pad])

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)

    try:
        ll = log_loss(y_true, proba, labels=list(range(n_classes)))
    except Exception:
        ll = float("nan")

    # Multiclass Brier (mean of per-class OvR Brier)
    brier = 0.0
    for c in range(n_classes):
        y_bin = (y_true == c).astype(float)
        brier += brier_score_loss(y_bin, proba[:, c])
    brier /= n_classes

    ece = expected_calibration_error(y_true, proba, n_bins=15)

    # ROC-AUC (OvR macro) — only if all classes present
    try:
        roc_auc = roc_auc_score(
            y_true, proba,
            multi_class="ovr",
            average="macro",
            labels=list(range(n_classes)),
        )
    except Exception:
        roc_auc = float("nan")

    # Per-class metrics
    per_class_recall = recall_score(
        y_true, y_pred, average=None, zero_division=0,
        labels=list(range(n_classes)),
    )
    per_class_precision = precision_score(
        y_true, y_pred, average=None, zero_division=0,
        labels=list(range(n_classes)),
    )
    per_class_f1 = f1_score(
        y_true, y_pred, average=None, zero_division=0,
        labels=list(range(n_classes)),
    )

    p = prefix + "_" if prefix else ""
    result = {
        f"{p}macro_f1": round(float(macro_f1), 6),
        f"{p}weighted_f1": round(float(weighted_f1), 6),
        f"{p}accuracy": round(float(accuracy), 6),
        f"{p}log_loss": round(float(ll), 6),
        f"{p}brier_score": round(float(brier), 6),
        f"{p}ece": round(float(ece), 6),
        f"{p}roc_auc_macro_ovr": round(float(roc_auc), 6),
    }

    for i, lbl in enumerate(labels):
        safe = lbl.replace(" ", "_").lower()
        result[f"{p}recall_{safe}"] = round(float(per_class_recall[i]), 6)
        result[f"{p}precision_{safe}"] = round(float(per_class_precision[i]), 6)
        result[f"{p}f1_{safe}"] = round(float(per_class_f1[i]), 6)

    return result


def binary_metrics(
    y_true: np.ndarray,
    proba: np.ndarray,
    prefix: str = "",
) -> Dict:
    """Compute binary detection metrics."""
    y_pred = np.argmax(proba, axis=1) if proba.ndim == 2 else (proba >= 0.5).astype(int)
    p_pos = proba[:, 1] if proba.ndim == 2 else proba

    p = prefix + "_" if prefix else ""
    return {
        f"{p}accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        f"{p}f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 6),
        f"{p}precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 6),
        f"{p}recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 6),
        f"{p}roc_auc": round(float(roc_auc_score(y_true, p_pos)), 6),
        f"{p}pr_auc": round(float(average_precision_score(y_true, p_pos)), 6),
    }


def confusion_matrix_dict(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str],
) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    return {
        "matrix": cm.tolist(),
        "labels": labels,
        "n_samples": int(len(y_true)),
    }


def sanity_check_accuracy(
    name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    warn_threshold: float = 0.99,
) -> None:
    """Raise a warning if accuracy is suspiciously high (potential leakage).

    This is the primary guard against the 100%-accuracy data-leakage bug that
    commonly afflicts UNSW-NB15 implementations.
    """
    import warnings
    acc = accuracy_score(y_true, y_pred)
    if acc >= warn_threshold:
        warnings.warn(
            f"\n{'='*60}\n"
            f"LEAKAGE ALERT: {name} accuracy = {acc:.4f} (>= {warn_threshold}).\n"
            f"This is almost certainly a data-leakage bug:\n"
            f"  1. Check that attack_cat / label are not in the feature set.\n"
            f"  2. Check that 'id' column is dropped.\n"
            f"  3. Check that preprocessing is fit inside folds only.\n"
            f"  4. Check that raw IP/port columns are not included.\n"
            f"{'='*60}",
            stacklevel=2,
        )
