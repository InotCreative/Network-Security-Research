"""
Base learner definitions and OOF (out-of-fold) probability collection.

Each BaseModel wraps a scikit-learn-compatible estimator and its preprocessing
family. The collect_oof method runs inner cross-validation on the outer-training
split to produce leak-free probability estimates for calibration and stacking.

Models
------
  RandomForestModel   — tree family
  ExtraTreesModel     — tree family
  XGBoostModel        — tree family
  KNNModel            — knn family

All models expose:
  fit(X, y)           — fit on full data (array or DataFrame)
  predict_proba(X)    — calibrated-ready softmax probabilities (n_samples, n_classes)
  collect_oof(X, y, cv) → (oof_proba, oof_true)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from src.models.preprocessing import get_preprocessor, TreePreprocessor, LinearPreprocessor, KNNPreprocessor
from src.utils.seeds import get_seed

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Abstract base for all learners in the ensemble."""

    name: str
    family: str   # 'tree' | 'linear' | 'knn'

    def __init__(self) -> None:
        self._preprocessor = get_preprocessor(self.family)
        self._estimator = self._build_estimator()
        self._n_classes: Optional[int] = None
        self._classes: Optional[np.ndarray] = None
        self._is_fitted = False

    @abstractmethod
    def _build_estimator(self):
        """Return an unfitted sklearn-compatible estimator."""
        ...

    # ──────────────────────────────────────────────────────────────────────────

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "BaseModel":
        """Fit preprocessor + estimator on (X, y)."""
        X_t = self._preprocessor.fit_transform(X)
        self._classes = np.unique(y)
        self._n_classes = len(self._classes)
        self._estimator.fit(X_t, y)
        self._is_fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return (n_samples, n_classes) probability matrix."""
        self._check_fitted()
        X_t = self._preprocessor.transform(X)
        return self._estimator.predict_proba(X_t)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        X_t = self._preprocessor.transform(X)
        return self._estimator.predict(X_t)

    def collect_oof(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        n_splits: int = 5,
        n_classes: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Collect out-of-fold probability predictions.

        Each fold: fit preprocessor + model on k-1 folds, predict proba on
        the held-out fold. Preprocessing is ALWAYS fit inside the fold.

        Returns
        -------
        oof_proba : (n_samples, n_classes) — NaN-free after all folds
        oof_true  : (n_samples,)            — true labels in original order
        """
        n = len(y)
        oof_proba = np.full((n, n_classes), np.nan)
        oof_true = np.asarray(y).copy()

        skf = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=get_seed(f"oof_{self.name}"),
        )

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr = np.asarray(y)[train_idx]

            # Fresh preprocessor per fold — never leak validation stats
            fold_prep = get_preprocessor(self.family)
            X_tr_t = fold_prep.fit_transform(X_tr)
            X_val_t = fold_prep.transform(X_val)

            # Fresh estimator per fold
            fold_est = self._build_estimator()
            fold_est.fit(X_tr_t, y_tr)

            fold_proba = fold_est.predict_proba(X_val_t)

            # Align to global class ordering (class index = class label)
            est_classes = fold_est.classes_
            for col_idx, cls in enumerate(est_classes):
                oof_proba[val_idx, cls] = fold_proba[:, col_idx]

            logger.debug(
                "%s OOF fold %d/%d done. Val size=%d",
                self.name, fold_idx + 1, n_splits, len(val_idx),
            )

        # Any class absent from a fold's training data → probability = 0
        oof_proba = np.nan_to_num(oof_proba, nan=0.0)
        # Re-normalise rows to sum to 1
        row_sums = oof_proba.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        oof_proba = oof_proba / row_sums

        return oof_proba, oof_true

    # ──────────────────────────────────────────────────────────────────────────

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(f"{self.name}: call fit() before predict_proba().")

    def get_feature_importances(self) -> Optional[np.ndarray]:
        """Return feature importances if available (tree models), else None."""
        return getattr(self._estimator, "feature_importances_", None)


# ──────────────────────────────────────────────────────────────────────────────
# Concrete model definitions
# ──────────────────────────────────────────────────────────────────────────────

class RandomForestModel(BaseModel):
    name = "random_forest"
    family = "tree"

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: Optional[int] = None,
        min_samples_split: int = 5,
        max_features: str = "sqrt",
    ) -> None:
        self._params = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            max_features=max_features,
        )
        super().__init__()

    def _build_estimator(self) -> RandomForestClassifier:
        return RandomForestClassifier(
            **self._params,
            class_weight="balanced",
            n_jobs=-1,
            random_state=get_seed("random_forest"),
        )


class ExtraTreesModel(BaseModel):
    name = "extra_trees"
    family = "tree"

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: Optional[int] = None,
        min_samples_split: int = 5,
        # Use ALL features with random thresholds — maximises diversity from RF
        # (RF uses sqrt features + optimal thresholds; ET uses all features +
        #  random thresholds → genuinely orthogonal inductive bias)
        max_features: float = 1.0,
    ) -> None:
        self._params = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            max_features=max_features,
        )
        super().__init__()

    def _build_estimator(self) -> ExtraTreesClassifier:
        return ExtraTreesClassifier(
            **self._params,
            class_weight="balanced",
            n_jobs=-1,
            random_state=get_seed("extra_trees"),
        )


class XGBoostModel(BaseModel):
    name = "xgboost"
    family = "tree"

    def __init__(
        self,
        n_estimators: int = 400,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: int = 3,
    ) -> None:
        self._params = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight,
        )
        super().__init__()

    def _build_estimator(self) -> XGBClassifier:
        # Objective is set dynamically in fit() once we know n_classes.
        # Default to multiclass; binary fit() overrides to binary:logistic.
        return XGBClassifier(
            **self._params,
            tree_method="hist",
            n_jobs=-1,
            verbosity=0,
            random_state=get_seed("xgboost"),
        )

    def _configure_objective(self, n_classes: int) -> None:
        if n_classes == 2:
            self._estimator.set_params(
                objective="binary:logistic",
                eval_metric="logloss",
            )
        else:
            self._estimator.set_params(
                objective="multi:softprob",
                eval_metric="mlogloss",
                num_class=n_classes,
            )

    def fit(self, X, y) -> "XGBoostModel":
        X_t = self._preprocessor.fit_transform(X)
        self._classes = np.unique(y)
        self._n_classes = len(self._classes)
        self._configure_objective(self._n_classes)
        self._estimator.fit(X_t, y)
        self._is_fitted = True
        return self

    def collect_oof(self, X, y, n_splits=5, n_classes=10):
        """Override to configure XGBoost objective per fold."""
        n = len(y)
        oof_proba = np.full((n, n_classes), np.nan)
        oof_true = np.asarray(y).copy()

        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(
            n_splits=n_splits, shuffle=True,
            random_state=get_seed(f"oof_{self.name}"),
        )
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr = np.asarray(y)[train_idx]

            fold_prep = get_preprocessor(self.family)
            X_tr_t = fold_prep.fit_transform(X_tr)
            X_val_t = fold_prep.transform(X_val)

            fold_est = self._build_estimator()
            fold_n_classes = len(np.unique(y_tr))
            if fold_n_classes == 2:
                fold_est.set_params(objective="binary:logistic", eval_metric="logloss")
            else:
                fold_est.set_params(objective="multi:softprob", eval_metric="mlogloss",
                                    num_class=fold_n_classes)
            fold_est.fit(X_tr_t, y_tr)

            fold_proba = fold_est.predict_proba(X_val_t)
            est_classes = fold_est.classes_
            for col_idx, cls in enumerate(est_classes):
                oof_proba[val_idx, cls] = fold_proba[:, col_idx]

        oof_proba = np.nan_to_num(oof_proba, nan=0.0)
        row_sums = oof_proba.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        return oof_proba / row_sums, oof_true


class KNNModel(BaseModel):
    name = "knn"
    family = "knn"

    def __init__(
        self,
        n_neighbors: int = 10,
        metric: str = "euclidean",
    ) -> None:
        self._params = dict(
            n_neighbors=n_neighbors,
            metric=metric,
        )
        super().__init__()

    def _build_estimator(self) -> KNeighborsClassifier:
        # KNN has no random_state; reproducibility guaranteed by data order + seed
        return KNeighborsClassifier(
            **self._params,
            weights="distance",
            n_jobs=-1,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Registry helpers
# ──────────────────────────────────────────────────────────────────────────────

ALL_MODEL_CLASSES = {
    "random_forest": RandomForestModel,
    "extra_trees": ExtraTreesModel,
    "xgboost": XGBoostModel,
    "knn": KNNModel,
}


def build_default_models() -> list:
    """Return one instance of each base learner with default hyperparameters."""
    return [cls() for cls in ALL_MODEL_CLASSES.values()]
