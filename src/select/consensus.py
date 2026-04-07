"""
Stability-weighted consensus feature selection.

Algorithm (per outer fold):
  1. Run inner stratified K-fold on the outer-training split.
  2. For each (selector, k candidate):
       a. For each inner fold, select top-k features.
       b. Compute STABILITY = mean pairwise Jaccard similarity of selected sets.
       c. Compute INNER_CV_MACRO_F1 using a fast surrogate (RF-50 on selected features).
       d. Compute selector_weight = inner_cv_f1 * stability.
  3. Aggregate feature-level CONSENSUS SCORE:
       score[feature] = Σ_{selector, fold} (selector_weight × I[feature in top-k*])
       where k* is the k that maximises inner_cv_f1 for that selector.
  4. Select the global top-k* features by consensus score.
  5. Serialise: full ranked list, selector weights, stability values, chosen k.

All inner fold statistics are computed exclusively on the outer-training split.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from src.models.preprocessing import TreePreprocessor
from src.select.selectors import BaseSelector, build_selectors
from src.utils.seeds import get_seed

logger = logging.getLogger(__name__)

# Configurable k candidates
DEFAULT_K_GRID = [8, 12, 16, 20, 24, 30, 36, 42, 50]


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def _mean_pairwise_jaccard(sets: List[set]) -> float:
    if len(sets) < 2:
        return 1.0
    sims = [_jaccard(a, b) for a, b in combinations(sets, 2)]
    return float(np.mean(sims))


@dataclass
class SelectionResult:
    """Output contract from ConsensusSelector.fit()."""
    selected_indices: List[int]      # column indices in the feature matrix
    selected_names: List[str]        # column names
    consensus_scores: Dict[str, float]   # feature → score
    full_ranking: List[str]          # all features ranked by consensus score
    chosen_k: int
    selector_weights: Dict[str, float]   # selector_name → weight
    selector_stabilities: Dict[str, float]
    inner_cv_f1s: Dict[str, float]   # selector_name → best inner F1
    k_grid: List[int]
    feature_manifest: List[dict]     # for paper appendix


class ConsensusSelector:
    """Stability-weighted consensus feature selector.

    Parameters
    ----------
    k_grid:
        Candidate feature counts to evaluate.
    n_inner_folds:
        Number of inner CV folds for stability / F1 estimation.
    surrogate_n_estimators:
        Size of the surrogate RF used for inner-fold F1 evaluation.
    """

    def __init__(
        self,
        k_grid: Optional[List[int]] = None,
        n_inner_folds: int = 3,
        surrogate_n_estimators: int = 50,
        use_stability_weighting: bool = True,
    ) -> None:
        self.k_grid = k_grid or DEFAULT_K_GRID
        self.n_inner_folds = n_inner_folds
        self.surrogate_n_estimators = surrogate_n_estimators
        self.use_stability_weighting = use_stability_weighting
        self._result: Optional[SelectionResult] = None

    # ──────────────────────────────────────────────────────────────────────────

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> SelectionResult:
        """Run consensus selection on (X, y).

        X should be the tree-preprocessed numeric array OR a DataFrame whose
        columns are the feature names. If a DataFrame is passed, feature_names
        defaults to X.columns.
        """
        if isinstance(X, pd.DataFrame):
            feature_names = feature_names or X.columns.tolist()
            # Preprocess with tree pipeline (needed for selectors)
            prep = TreePreprocessor()
            X_arr = prep.fit_transform(X)
            feature_names = prep.feature_names_out
        else:
            X_arr = np.asarray(X)
            if feature_names is None:
                feature_names = [f"f{i}" for i in range(X_arr.shape[1])]

        n_features = X_arr.shape[1]
        y_arr = np.asarray(y)

        selectors = build_selectors()
        inner_skf = StratifiedKFold(
            n_splits=self.n_inner_folds,
            shuffle=True,
            random_state=get_seed("consensus_inner_cv"),
        )

        # selector_name → {k → [set_of_selected_per_fold]}
        sel_results: Dict[str, Dict[int, List[set]]] = {
            s.name: {k: [] for k in self.k_grid} for s in selectors
        }
        # selector_name → {k → [inner_macro_f1]}
        sel_f1s: Dict[str, Dict[int, List[float]]] = {
            s.name: {k: [] for k in self.k_grid} for s in selectors
        }

        for fold_idx, (tr_idx, val_idx) in enumerate(inner_skf.split(X_arr, y_arr)):
            X_tr, X_val = X_arr[tr_idx], X_arr[val_idx]
            y_tr, y_val = y_arr[tr_idx], y_arr[val_idx]

            for sel in selectors:
                try:
                    sel.fit(X_tr, y_tr)
                except Exception as exc:
                    logger.warning("Selector %s failed on fold %d: %s", sel.name, fold_idx, exc)
                    continue

                for k in self.k_grid:
                    if k > n_features:
                        continue
                    top_k_idx = sel.get_ranking(k)
                    top_k_set = set(top_k_idx)
                    sel_results[sel.name][k].append(top_k_set)

                    # Quick surrogate RF for inner-fold F1
                    f1 = self._surrogate_f1(
                        X_tr[:, top_k_idx], y_tr,
                        X_val[:, top_k_idx], y_val,
                    )
                    sel_f1s[sel.name][k].append(f1)

            logger.debug("Consensus inner fold %d/%d done", fold_idx + 1, self.n_inner_folds)

        # ── Compute selector weights and chosen k ────────────────────────────
        selector_weights: Dict[str, float] = {}
        selector_stabilities: Dict[str, float] = {}
        selector_best_k: Dict[str, int] = {}
        inner_cv_f1s: Dict[str, float] = {}

        for sel in selectors:
            best_f1 = -1.0
            best_k_for_sel = self.k_grid[0]
            best_stability = 0.0

            for k in self.k_grid:
                if k > n_features:
                    continue
                sets = sel_results[sel.name][k]
                f1s = sel_f1s[sel.name][k]
                if not sets:
                    continue
                stability = _mean_pairwise_jaccard(sets)
                mean_f1 = float(np.mean(f1s)) if f1s else 0.0
                # Joint criterion: weight by stability only when enabled.
                # Ablation: use_stability_weighting=False → F1 only (no stability penalty).
                combined = mean_f1 * (stability if self.use_stability_weighting else 1.0)
                if combined > best_f1:
                    best_f1 = combined
                    best_k_for_sel = k
                    best_stability = stability

            selector_weights[sel.name] = max(best_f1, 0.0)
            selector_stabilities[sel.name] = best_stability
            selector_best_k[sel.name] = best_k_for_sel
            inner_cv_f1s[sel.name] = max(best_f1, 0.0)

        # ── Aggregate consensus scores ────────────────────────────────────────
        consensus_scores = np.zeros(n_features)

        total_weight = sum(selector_weights.values())
        if total_weight == 0:
            logger.warning("All selector weights are zero — falling back to equal weights.")
            total_weight = len(selectors)
            selector_weights = {s.name: 1.0 for s in selectors}

        for sel in selectors:
            w = selector_weights[sel.name]
            k_chosen = selector_best_k[sel.name]
            # Aggregate votes from all inner folds for this selector at k_chosen
            for fold_set in sel_results[sel.name].get(k_chosen, []):
                for feat_idx in fold_set:
                    consensus_scores[feat_idx] += w / self.n_inner_folds

        # ── Choose global k* ──────────────────────────────────────────────────
        # Use the weighted-average best k across selectors
        if total_weight > 0:
            chosen_k = int(round(
                sum(selector_best_k[s.name] * selector_weights[s.name]
                    for s in selectors) / total_weight
            ))
        else:
            chosen_k = self.k_grid[len(self.k_grid) // 2]
        # Snap to nearest grid value
        chosen_k = min(self.k_grid, key=lambda x: abs(x - chosen_k))
        chosen_k = min(chosen_k, n_features)

        # ── Rank and select ───────────────────────────────────────────────────
        ranking_idx = np.argsort(consensus_scores)[::-1]
        full_ranking = [feature_names[i] for i in ranking_idx]
        selected_indices = ranking_idx[:chosen_k].tolist()
        selected_names = [feature_names[i] for i in selected_indices]

        # ── Build manifest ────────────────────────────────────────────────────
        manifest = [
            {
                "feature": feature_names[i],
                "consensus_score": float(consensus_scores[i]),
                "rank": int(rank + 1),
                "selected": feature_names[i] in set(selected_names),
            }
            for rank, i in enumerate(ranking_idx)
        ]

        self._result = SelectionResult(
            selected_indices=selected_indices,
            selected_names=selected_names,
            consensus_scores={feature_names[i]: float(consensus_scores[i]) for i in range(n_features)},
            full_ranking=full_ranking,
            chosen_k=chosen_k,
            selector_weights=selector_weights,
            selector_stabilities=selector_stabilities,
            inner_cv_f1s=inner_cv_f1s,
            k_grid=self.k_grid,
            feature_manifest=manifest,
        )
        logger.info(
            "Consensus selection: chosen_k=%d | top features: %s",
            chosen_k, selected_names[:5],
        )
        return self._result

    def _surrogate_f1(
        self,
        X_tr: np.ndarray, y_tr: np.ndarray,
        X_val: np.ndarray, y_val: np.ndarray,
    ) -> float:
        rf = RandomForestClassifier(
            n_estimators=self.surrogate_n_estimators,
            max_depth=8,
            class_weight="balanced",
            n_jobs=-1,
            random_state=get_seed("surrogate_rf"),
        )
        rf.fit(X_tr, y_tr)
        y_pred = rf.predict(X_val)
        return f1_score(y_val, y_pred, average="macro", zero_division=0)
