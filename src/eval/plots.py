"""
Manuscript-ready figure generation.

All plots are saved to reports/ and registered with the run manifest.
Style: publication-quality (no interactive mode required).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")   # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid", context="paper")
FIGSIZE_SINGLE = (6, 4)
FIGSIZE_WIDE = (10, 4)
FIGSIZE_SQUARE = (5, 5)


def save_fig(fig: plt.Figure, path: Path, dpi: int = 150) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.debug("Saved figure → %s", path)
    return path


def plot_reliability_diagram(
    reliability_data: Dict,
    model_name: str,
    out_path: Path,
) -> Path:
    """Plot a reliability (calibration) diagram."""
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

    centres = reliability_data["bin_centres"]
    accs = reliability_data["bin_accuracies"]
    confs = reliability_data["bin_confidences"]
    ece = reliability_data["ece"]

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    ax.plot(confs, accs, "o-", color="steelblue", lw=1.5, ms=4, label=f"ECE={ece:.4f}")

    ax.fill_between(
        confs, accs, confs,
        where=[a < c for a, c in zip(accs, confs)],
        alpha=0.15, color="red", label="Over-confident",
    )
    ax.fill_between(
        confs, accs, confs,
        where=[a >= c for a, c in zip(accs, confs)],
        alpha=0.15, color="green", label="Under-confident",
    )

    ax.set_xlabel("Mean predicted confidence")
    ax.set_ylabel("Fraction correct")
    ax.set_title(f"Reliability diagram — {model_name}")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    return save_fig(fig, out_path)


def plot_confusion_matrix(
    cm_data: dict,
    out_path: Path,
    normalise: bool = True,
) -> Path:
    """Plot a labelled confusion matrix heatmap."""
    matrix = np.array(cm_data["matrix"])
    labels = cm_data["labels"]

    if normalise:
        row_sums = matrix.sum(axis=1, keepdims=True)
        matrix = np.where(row_sums > 0, matrix / row_sums, 0.0)
        fmt = ".2f"
        vmax = 1.0
    else:
        fmt = "d"
        vmax = None

    n = len(labels)
    figsize = (max(6, n * 0.8), max(5, n * 0.7))
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        matrix, annot=True, fmt=fmt,
        xticklabels=labels, yticklabels=labels,
        cmap="Blues", ax=ax, vmin=0, vmax=vmax,
        annot_kws={"size": 8},
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (row-normalised)" if normalise else "Confusion matrix")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    return save_fig(fig, out_path)


def plot_macro_f1_comparison(
    results: Dict[str, float],
    out_path: Path,
    title: str = "Macro F1 comparison",
) -> Path:
    """Bar chart comparing macro F1 of all methods."""
    methods = list(results.keys())
    f1s = [results[m] for m in methods]

    colors = ["#1f77b4"] * len(methods)
    # Highlight proposed ensemble
    for i, m in enumerate(methods):
        if "gated" in m.lower() or "ensemble" in m.lower() or "final" in m.lower():
            colors[i] = "#d62728"

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    bars = ax.barh(methods, f1s, color=colors, height=0.5)
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
    ax.set_xlabel("Macro F1")
    ax.set_title(title)
    ax.set_xlim(0, min(1.05, max(f1s) * 1.15))
    ax.invert_yaxis()
    return save_fig(fig, out_path)


def plot_per_class_metrics(
    metrics: Dict[str, float],
    class_names: List[str],
    out_path: Path,
) -> Path:
    """Grouped bar chart of per-class recall/precision/F1."""
    recalls = [metrics.get(f"recall_{c.replace(' ','_').lower()}", 0) for c in class_names]
    precisions = [metrics.get(f"precision_{c.replace(' ','_').lower()}", 0) for c in class_names]
    f1s = [metrics.get(f"f1_{c.replace(' ','_').lower()}", 0) for c in class_names]

    x = np.arange(len(class_names))
    w = 0.25
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(x - w, recalls, w, label="Recall", color="steelblue")
    ax.bar(x, precisions, w, label="Precision", color="darkorange")
    ax.bar(x + w, f1s, w, label="F1", color="green")

    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=30, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Per-class metrics")
    ax.legend(fontsize=8)
    return save_fig(fig, out_path)


def plot_confidence_histogram(
    proba: np.ndarray,
    model_name: str,
    out_path: Path,
    n_bins: int = 20,
) -> Path:
    """Histogram of max predicted confidence values."""
    confidence = proba.max(axis=1)
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    ax.hist(confidence, bins=n_bins, color="steelblue", edgecolor="white", linewidth=0.5)
    ax.axvline(confidence.mean(), color="red", ls="--", lw=1, label=f"Mean={confidence.mean():.3f}")
    ax.set_xlabel("Max predicted confidence")
    ax.set_ylabel("Count")
    ax.set_title(f"Confidence histogram — {model_name}")
    ax.legend(fontsize=8)
    return save_fig(fig, out_path)


def plot_agreement_error_slices(
    slice_rows: List[Dict],
    out_path: Path,
) -> Path:
    """Grouped bar chart: accuracy and macro F1 by agreement level."""
    labels = [r["agreement_level"] for r in slice_rows if r["n_samples"] > 0]
    accs = [r["accuracy"] for r in slice_rows if r["n_samples"] > 0]
    f1s = [r["macro_f1"] for r in slice_rows if r["n_samples"] > 0]
    counts = [r["n_samples"] for r in slice_rows if r["n_samples"] > 0]

    if not labels:
        fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return save_fig(fig, out_path)

    x = np.arange(len(labels))
    w = 0.3
    fig, ax1 = plt.subplots(figsize=FIGSIZE_SINGLE)

    bars1 = ax1.bar(x - w / 2, accs, w, label="Accuracy", color="steelblue")
    bars2 = ax1.bar(x + w / 2, f1s, w, label="Macro F1", color="darkorange")

    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{l}\n(n={c})" for l, c in zip(labels, counts)], fontsize=9)
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("Score")
    ax1.set_title("Error slices by base-model agreement level")
    ax1.legend(fontsize=8)
    ax1.bar_label(bars1, fmt="%.3f", padding=2, fontsize=7)
    ax1.bar_label(bars2, fmt="%.3f", padding=2, fontsize=7)

    return save_fig(fig, out_path)
