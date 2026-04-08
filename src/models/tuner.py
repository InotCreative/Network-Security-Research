"""
Hyperparameter search for base learners.

Uses RandomizedSearchCV with stratified inner K-fold, optimising macro F1.
All searches run on the preprocessed feature matrix inside an outer fold,
so tuned parameters are never informed by validation or test data.

Design
------
* Each model class has its own search space (SEARCH_SPACES).
* Preprocessing is performed once per model family before the search, to
  avoid re-running it inside every CV iteration.
* XGBoost objective/num_class are fixed at init time (not searched).
* A try/except around each search falls back to default params if the
  search crashes (e.g. rare class absent from a stratified fold).

Usage (inside fold_runner)
--------------------------
    tuner = HyperparameterTuner(n_iter=20, n_cv=3)
    best_params = tuner.tune_all(X_tr_sel, y_tr, n_classes=10)
    base_models = build_models_from_params(best_params)
"""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from src.models.preprocessing import get_preprocessor
from src.utils.seeds import get_seed

logger = logging.getLogger(__name__)

# ── Search spaces ──────────────────────────────────────────────────────────────
# Each dict maps sklearn estimator param names → list of candidate values.
# Designed to bracket the defaults used in base_models.py, so the default
# is always a reachable point in the search space.

SEARCH_SPACES: Dict[str, Dict] = {
    "random_forest": {
        "n_estimators": [100, 200, 300, 400, 500],
        "max_depth": [None, 10, 15, 20, 30],
        "min_samples_split": [2, 5, 10, 20],
        "max_features": ["sqrt", "log2"],
    },
    "extra_trees": {
        "n_estimators": [100, 200, 300, 400, 500],
        "max_depth": [None, 10, 15, 20, 30],
        "min_samples_split": [2, 5, 10, 20],
        "max_features": [0.3, 0.5, 0.75, 1.0],
    },
    "xgboost": {
        "n_estimators": [200, 300, 400, 500],
        "max_depth": [4, 5, 6, 8, 10],
        "learning_rate": [0.01, 0.05, 0.1, 0.15, 0.2],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        "min_child_weight": [1, 3, 5, 7],
    },
    "knn": {
        "n_neighbors": [5, 10, 15, 20, 25, 30],
        "metric": ["euclidean", "manhattan"],
    },
}

# Map model name → preprocessing family
_FAMILIES: Dict[str, str] = {
    "random_forest": "tree",
    "extra_trees": "tree",
    "xgboost": "tree",
    "knn": "knn",
}

# Default params to fall back on if a model's search fails
_DEFAULT_PARAMS: Dict[str, Dict] = {
    "random_forest": dict(n_estimators=200, max_depth=None, min_samples_split=5, max_features="sqrt"),
    "extra_trees": dict(n_estimators=300, max_depth=None, min_samples_split=5, max_features=1.0),
    "xgboost": dict(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8,
                    colsample_bytree=0.8, min_child_weight=3),
    "knn": dict(n_neighbors=10, metric="euclidean"),
}


class HyperparameterTuner:
    """Runs RandomizedSearchCV for each base learner inside an outer fold.

    Parameters
    ----------
    n_iter:
        Random parameter settings sampled per model.
    n_cv:
        Stratified inner-fold count for the search.
    scoring:
        Sklearn scoring string. Default 'f1_macro' matches the paper's
        primary metric.
    """

    def __init__(
        self,
        n_iter: int = 20,
        n_cv: int = 3,
        scoring: str = "f1_macro",
    ) -> None:
        self.n_iter = n_iter
        self.n_cv = n_cv
        self.scoring = scoring

    # ──────────────────────────────────────────────────────────────────────────

    def tune_all(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        n_classes: int,
    ) -> Dict[str, dict]:
        """Tune all base models.

        Returns
        -------
        Dict mapping model_name → best_params dict ready for model __init__.
        """
        results: Dict[str, dict] = {}
        for model_name in SEARCH_SPACES:
            logger.info(
                "HP search: %-20s  n_iter=%d  n_cv=%d",
                model_name, self.n_iter, self.n_cv,
            )
            results[model_name] = self._tune_one(model_name, X, y, n_classes)
        return results

    def _tune_one(
        self,
        model_name: str,
        X: pd.DataFrame,
        y: np.ndarray,
        n_classes: int,
    ) -> dict:
        """Search hyperparameters for one model. Falls back to defaults on error."""
        try:
            family = _FAMILIES[model_name]
            prep = get_preprocessor(family)
            X_arr = prep.fit_transform(X)
            y_arr = np.asarray(y)

            estimator = _build_search_estimator(model_name, n_classes)
            space = SEARCH_SPACES[model_name]

            cv = StratifiedKFold(
                n_splits=self.n_cv,
                shuffle=True,
                random_state=get_seed(f"hp_cv_{model_name}"),
            )

            rscv = RandomizedSearchCV(
                estimator=estimator,
                param_distributions=space,
                n_iter=self.n_iter,
                scoring=self.scoring,
                cv=cv,
                n_jobs=-1,
                random_state=get_seed(f"hp_rscv_{model_name}"),
                refit=False,
                error_score=0.0,   # treat crash in a single fold as score=0
                verbose=0,
            )
            rscv.fit(X_arr, y_arr)
            best = dict(rscv.best_params_)
            logger.info(
                "  %s → score=%.4f  params=%s",
                model_name, rscv.best_score_, best,
            )
            return best

        except Exception as exc:
            logger.warning(
                "HP search failed for %s (%s). Using defaults.", model_name, exc
            )
            return dict(_DEFAULT_PARAMS[model_name])


# ── Helpers ────────────────────────────────────────────────────────────────────

def _build_search_estimator(model_name: str, n_classes: int):
    """Build a bare sklearn estimator for use inside RandomizedSearchCV."""
    seed = get_seed(f"hp_est_{model_name}")

    if model_name == "random_forest":
        return RandomForestClassifier(
            class_weight="balanced", n_jobs=-1, random_state=seed
        )

    if model_name == "extra_trees":
        return ExtraTreesClassifier(
            class_weight="balanced", n_jobs=-1, random_state=seed
        )

    if model_name == "xgboost":
        # Fix objective and num_class; only vary other hyperparams.
        xgb = XGBClassifier(
            tree_method="hist",
            n_jobs=-1,
            verbosity=0,
            random_state=seed,
        )
        if n_classes > 2:
            xgb.set_params(
                objective="multi:softprob",
                eval_metric="mlogloss",
                num_class=n_classes,
            )
        else:
            xgb.set_params(
                objective="binary:logistic",
                eval_metric="logloss",
            )
        return xgb

    if model_name == "knn":
        return KNeighborsClassifier(weights="distance", n_jobs=-1)

    raise ValueError(f"Unknown model name: {model_name!r}")
