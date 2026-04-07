"""
Disagreement-aware meta-feature construction.

Takes the calibrated probability matrices from all base models and derives
a rich descriptor of inter-model uncertainty and disagreement. These are the
ONLY inputs to the stacker and gate — raw traffic features never flow through
to the meta-learner.

Meta-feature groups (Section 6.4 of the design document):
  1. Per-model descriptors:   max_prob, entropy, top-two margin
  2. Cross-model agreement:   per-class variance, vote entropy, vote agreement
  3. Global uncertainty:      mean/variance of confidence across models
  4. Path gap (gate only):    argmax gap between p_weighted and p_stack
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

_EPS = 1e-9


def entropy(proba: np.ndarray) -> np.ndarray:
    """Shannon entropy per row. proba: (n, C)."""
    p = np.clip(proba, _EPS, 1.0)
    return -(p * np.log(p)).sum(axis=1)


def top_two_margin(proba: np.ndarray) -> np.ndarray:
    """Difference between the top-1 and top-2 probabilities per row."""
    sorted_p = np.sort(proba, axis=1)[:, ::-1]
    if sorted_p.shape[1] < 2:
        return sorted_p[:, 0]
    return sorted_p[:, 0] - sorted_p[:, 1]


def vote_entropy(vote_counts: np.ndarray, n_models: int) -> np.ndarray:
    """Normalised vote entropy. vote_counts: (n, C) — count of argmax per class."""
    p = vote_counts / (n_models + _EPS)
    p = np.clip(p, _EPS, 1.0)
    h = -(p * np.log(p)).sum(axis=1)
    max_h = np.log(vote_counts.shape[1])
    return h / (max_h + _EPS)


def build_meta_features(
    calibrated_probas: List[np.ndarray],
    model_names: Optional[List[str]] = None,
    p_weighted: Optional[np.ndarray] = None,
    p_stack: Optional[np.ndarray] = None,
    simplified: bool = False,
) -> np.ndarray:
    """
    Build the meta-feature matrix from calibrated base model probabilities.

    Parameters
    ----------
    calibrated_probas:
        List of (n_samples, n_classes) probability arrays — one per base model.
        Must all have identical shape.
    model_names:
        Optional list for documentation; not used in computation.
    p_weighted:
        (n_samples, n_classes) weighted-average path probabilities.
        If provided, the path-gap features are included.
    p_stack:
        (n_samples, n_classes) stacker path probabilities.
        Required if p_weighted is provided (and vice versa).
    simplified:
        If True, omit the full calibrated probability vectors and use only
        scalar disagreement/uncertainty descriptors. This is the ablation
        condition for "simplified meta-features" (design doc §11).

    Returns
    -------
    meta : (n_samples, n_meta_features)
    """
    n = calibrated_probas[0].shape[0]
    n_classes = calibrated_probas[0].shape[1]
    n_models = len(calibrated_probas)
    parts = []

    # ── Full calibrated probability vectors (one per model) ───────────────────
    # Omitted in simplified ablation — only scalar descriptors are used.
    if not simplified:
        for proba in calibrated_probas:
            parts.append(proba)   # (n, C) each

    # ── Per-model descriptors ─────────────────────────────────────────────────
    for proba in calibrated_probas:
        max_p = proba.max(axis=1, keepdims=True)          # (n, 1)
        ent = entropy(proba).reshape(-1, 1)                # (n, 1)
        margin = top_two_margin(proba).reshape(-1, 1)      # (n, 1)
        parts.extend([max_p, ent, margin])

    # ── Cross-model agreement features ────────────────────────────────────────
    stack_probas = np.stack(calibrated_probas, axis=0)  # (M, n, C)

    # Per-class variance across models
    per_class_var = stack_probas.var(axis=0)            # (n, C)
    parts.append(per_class_var)

    # Vote counts (argmax per model)
    vote_matrix = np.zeros((n, n_classes), dtype=float)
    for proba in calibrated_probas:
        votes = np.argmax(proba, axis=1)
        for cls_idx in range(n_classes):
            vote_matrix[:, cls_idx] += (votes == cls_idx).astype(float)

    # Normalised vote agreement (fraction of models agreeing with plurality)
    plurality_votes = vote_matrix.max(axis=1, keepdims=True)
    vote_agreement = plurality_votes / n_models             # (n, 1)
    parts.append(vote_agreement)

    # Vote entropy
    v_entropy = vote_entropy(vote_matrix, n_models).reshape(-1, 1)
    parts.append(v_entropy)

    # ── Global confidence features ────────────────────────────────────────────
    max_probs = np.stack([p.max(axis=1) for p in calibrated_probas], axis=1)  # (n, M)
    mean_conf = max_probs.mean(axis=1, keepdims=True)    # (n, 1)
    var_conf = max_probs.var(axis=1, keepdims=True)      # (n, 1)
    parts.extend([mean_conf, var_conf])

    # ── Path-gap feature (for gate training) ─────────────────────────────────
    if p_weighted is not None and p_stack is not None:
        # Probability gap between stacker and weighted-average top-1 class
        prob_gap = np.abs(
            p_stack.max(axis=1) - p_weighted.max(axis=1)
        ).reshape(-1, 1)
        # Agreement indicator: do both paths predict the same class?
        agree = (
            np.argmax(p_stack, axis=1) == np.argmax(p_weighted, axis=1)
        ).astype(float).reshape(-1, 1)
        # Per-class gap
        path_prob_diff = np.abs(p_stack - p_weighted)   # (n, C)
        parts.extend([prob_gap, agree, path_prob_diff])

    return np.hstack(parts)


def meta_feature_names(
    n_classes: int,
    n_models: int,
    model_names: Optional[List[str]] = None,
    include_path_gap: bool = False,
    simplified: bool = False,
) -> List[str]:
    """Return a list of descriptive names matching the columns of build_meta_features."""
    if model_names is None:
        model_names = [f"model_{i}" for i in range(n_models)]

    names: List[str] = []
    if not simplified:
        for mname in model_names:
            for c in range(n_classes):
                names.append(f"{mname}__prob_class{c}")
    for mname in model_names:
        names.extend([
            f"{mname}__max_prob",
            f"{mname}__entropy",
            f"{mname}__top2_margin",
        ])
    for c in range(n_classes):
        names.append(f"per_class_var__{c}")
    names.append("vote_agreement")
    names.append("vote_entropy")
    names.extend(["mean_confidence", "var_confidence"])
    if include_path_gap:
        names.extend(["path_prob_gap", "path_agree"])
        for c in range(n_classes):
            names.append(f"path_diff__class{c}")
    return names
