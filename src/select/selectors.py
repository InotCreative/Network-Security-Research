"""
Individual feature selectors used in the consensus pipeline.

Four selectors (each fit on the TREE-preprocessed feature matrix):
  1. ANOVA F-test         (f_classif)
  2. Mutual Information   (mutual_info_classif)
  3. RFE with RandomForest
  4. RFE with SGDClassifier (L1/Elastic-Net linear surrogate)

Each exposes:
  fit(X, y) → self
  get_ranking(k) → List[int]  — indices of top-k features (in X columns)

Selectors operate on the numeric+encoded representation produced by
TreePreprocessor because RFE with forest needs consistent column ordering.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
from sklearn.feature_selection import (
    SelectKBest,
    f_classif,
    mutual_info_classif,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier

from src.utils.seeds import get_seed

logger = logging.getLogger(__name__)


class BaseSelector(ABC):
    name: str

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseSelector":
        ...

    @abstractmethod
    def get_ranking(self, k: int) -> List[int]:
        """Return indices of the top-k features (0-based column indices)."""
        ...

    @abstractmethod
    def feature_scores(self) -> np.ndarray:
        """Return a score array of length n_features (higher = more important)."""
        ...


class ANOVASelector(BaseSelector):
    """ANOVA F-test selector."""
    name = "anova_f"

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ANOVASelector":
        self._skb = SelectKBest(score_func=f_classif, k="all")
        self._skb.fit(X, y)
        return self

    def feature_scores(self) -> np.ndarray:
        scores = np.nan_to_num(self._skb.scores_, nan=0.0)
        return scores

    def get_ranking(self, k: int) -> List[int]:
        scores = self.feature_scores()
        return list(np.argsort(scores)[::-1][:k])


class MutualInfoSelector(BaseSelector):
    """Mutual information selector."""
    name = "mutual_info"

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MutualInfoSelector":
        seed = get_seed("mi_selector")
        scores = mutual_info_classif(
            X, y, discrete_features=False, random_state=seed, n_neighbors=5
        )
        self._scores = scores
        return self

    def feature_scores(self) -> np.ndarray:
        return self._scores.copy()

    def get_ranking(self, k: int) -> List[int]:
        return list(np.argsort(self._scores)[::-1][:k])


class RFImportanceSelector(BaseSelector):
    """Random Forest feature-importance ranking.

    Fits a small RF and ranks features by mean decrease in Gini impurity
    (feature_importances_). This is a single-pass importance ranking, not
    recursive feature elimination — it is faster and more stable than
    step-wise RFE while capturing non-linear interactions implicitly.
    """
    name = "rf_importance"

    def __init__(self, n_estimators: int = 50) -> None:
        self._n_estimators = n_estimators

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RFImportanceSelector":
        base = RandomForestClassifier(
            n_estimators=self._n_estimators,
            max_depth=10,
            class_weight="balanced",
            n_jobs=-1,
            random_state=get_seed("rf_importance"),
        )
        base.fit(X, y)
        self._importances = base.feature_importances_
        return self

    def feature_scores(self) -> np.ndarray:
        return self._importances.copy()

    def get_ranking(self, k: int) -> List[int]:
        return list(np.argsort(self._importances)[::-1][:k])


class SGDCoefficientSelector(BaseSelector):
    """SGD ElasticNet coefficient-magnitude ranking.

    Fits a sparse linear SGD classifier (modified Huber loss, ElasticNet
    penalty) and ranks features by mean absolute coefficient across classes.
    Provides a complementary linear-separability view to the tree-based
    selectors.
    """
    name = "sgd_coef"

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SGDCoefficientSelector":
        base = SGDClassifier(
            loss="modified_huber",
            penalty="elasticnet",
            l1_ratio=0.5,
            max_iter=500,
            tol=1e-3,
            class_weight="balanced",
            random_state=get_seed("sgd_coef"),
            n_jobs=-1,
        )
        base.fit(X, y)
        # Multi-class: average absolute coefficient across classes
        if base.coef_.ndim == 2:
            self._scores = np.abs(base.coef_).mean(axis=0)
        else:
            self._scores = np.abs(base.coef_).ravel()
        return self

    def feature_scores(self) -> np.ndarray:
        return self._scores.copy()

    def get_ranking(self, k: int) -> List[int]:
        return list(np.argsort(self._scores)[::-1][:k])


ALL_SELECTORS = [
    ANOVASelector,
    MutualInfoSelector,
    RFImportanceSelector,
    SGDCoefficientSelector,
]

SELECTOR_NAMES = {cls().name: cls for cls in ALL_SELECTORS}


def build_selectors(names: Optional[List[str]] = None) -> List[BaseSelector]:
    """Return selector instances.

    Parameters
    ----------
    names:
        If None (default), return all four selectors.
        If a list of selector names is given, return only those selectors
        in the canonical order (anova_f, mutual_info, rf_importance, sgd_coef).
        This is used by the single_selector ablation to test whether the
        consensus of four heterogeneous selectors outperforms one alone.

    Valid names: 'anova_f', 'mutual_info', 'rf_importance', 'sgd_coef'
    """
    if names is None:
        return [cls() for cls in ALL_SELECTORS]
    name_set = set(names)
    unknown = name_set - set(SELECTOR_NAMES)
    if unknown:
        raise ValueError(f"Unknown selector names: {unknown}. "
                         f"Valid: {set(SELECTOR_NAMES)}")
    # Preserve canonical order
    return [cls() for cls in ALL_SELECTORS if cls().name in name_set]
