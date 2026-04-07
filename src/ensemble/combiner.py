"""
EnsembleCombiner: orchestrates the full two-path + gate ensemble.

Encapsulates:
  - Base model OOF collection
  - Calibration fitting per base model
  - Weighted-average path (p_weighted)
  - Stacker path (p_stack)
  - Gate fitting and final combination (p_final)
  - Refitting base models on full outer-training data

This is the single entry point the fold_runner calls. It emits serialisable
artifacts and a rich model card for each run.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.ensemble.calibration import OvRCalibrator
from src.ensemble.gate import DisagreementGate
from src.ensemble.meta_features import build_meta_features, meta_feature_names
from src.ensemble.stacker import MetaStacker
from src.ensemble.weighted_avg import WeightedAverager
from src.models.base_models import BaseModel

logger = logging.getLogger(__name__)


@dataclass
class EnsembleArtifacts:
    """All serialisable components produced in one outer fold."""
    calibrators: Dict[str, OvRCalibrator] = field(default_factory=dict)
    averager: Optional[WeightedAverager] = None
    stacker: Optional[MetaStacker] = None
    gate: Optional[DisagreementGate] = None
    model_weights: Dict[str, float] = field(default_factory=dict)
    chosen_calibration_methods: Dict[str, str] = field(default_factory=dict)


class EnsembleCombiner:
    """
    Parameters
    ----------
    base_models:
        List of fitted (or unfitted) BaseModel instances.
    n_classes:
        Number of output classes.
    n_oof_splits:
        Inner CV splits for OOF collection.
    gate_model:
        'ridge' | 'tree' | 'auto'
    ablation:
        Dict of flags to disable components for ablation studies.
        Keys: 'no_calibration', 'no_gate', 'fixed_beta'
    """

    def __init__(
        self,
        base_models: List[BaseModel],
        n_classes: int,
        n_oof_splits: int = 5,
        gate_model: str = "auto",
        ablation: Optional[Dict] = None,
    ) -> None:
        self.base_models = base_models
        self.n_classes = n_classes
        self.n_oof_splits = n_oof_splits
        self.gate_model = gate_model
        self.ablation = ablation or {}
        self._artifacts: Optional[EnsembleArtifacts] = None

    # ──────────────────────────────────────────────────────────────────────────

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray) -> "EnsembleCombiner":
        """Full ensemble training pipeline on the outer-training split.

        Steps:
          1. Collect OOF probabilities for each base model (inner CV on X_train).
          2. Fit calibrators per model on OOF probabilities.
          3. Calibrate OOF probabilities.
          4. Fit weighted averager.
          5. Build meta-features (without path gap yet).
          6. Get stacker OOF predictions via inner CV on meta-features.
          7. Build path-gap augmented meta-features.
          8. Fit gate.
          9. Refit all base models on full X_train.
        """
        art = EnsembleArtifacts()
        model_names = [m.name for m in self.base_models]

        # ── Step 1: OOF probabilities ─────────────────────────────────────────
        logger.info("Collecting OOF probabilities (%d models)…", len(self.base_models))
        raw_oof_probas: List[np.ndarray] = []
        for model in self.base_models:
            oof_proba, oof_true = model.collect_oof(
                X_train, y_train,
                n_splits=self.n_oof_splits,
                n_classes=self.n_classes,
            )
            raw_oof_probas.append(oof_proba)
        oof_true_arr = np.asarray(y_train)

        # ── Step 2 & 3: Calibration ───────────────────────────────────────────
        cal_oof_probas: List[np.ndarray] = []
        for model, raw_oof in zip(self.base_models, raw_oof_probas):
            if self.ablation.get("no_calibration"):
                cal_oof_probas.append(raw_oof)
                art.calibrators[model.name] = None
                art.chosen_calibration_methods[model.name] = "none"
            else:
                cal = OvRCalibrator(method="auto")
                cal.fit(raw_oof, oof_true_arr, self.n_classes)
                cal_proba = cal.predict_proba(raw_oof)
                cal_oof_probas.append(cal_proba)
                art.calibrators[model.name] = cal
                art.chosen_calibration_methods[model.name] = cal.chosen_method
                logger.info("%s calibration method: %s", model.name, cal.chosen_method)

        # ── Step 4: Weighted averager ─────────────────────────────────────────
        averager = WeightedAverager(model_names)
        averager.fit(cal_oof_probas, oof_true_arr, self.n_classes)
        p_weighted_oof = averager.predict_proba(cal_oof_probas)
        art.averager = averager
        art.model_weights = averager.get_weights()

        # ── Step 5: Meta-features (without path gap) ──────────────────────────
        simplified = bool(self.ablation.get("simplified_meta_features"))
        meta_X_no_gap = build_meta_features(cal_oof_probas, simplified=simplified)

        # ── Step 6: Stacker OOF predictions ──────────────────────────────────
        stacker = MetaStacker(allow_gb_fallback=True)
        p_stack_oof = self._get_stacker_oof(meta_X_no_gap, oof_true_arr, stacker)
        stacker.fit(meta_X_no_gap, oof_true_arr, self.n_classes)
        art.stacker = stacker

        # ── Step 7: Augmented meta-features (with path gap) ───────────────────
        meta_X_full = build_meta_features(
            cal_oof_probas,
            model_names=model_names,
            p_weighted=p_weighted_oof,
            p_stack=p_stack_oof,
            simplified=simplified,
        )

        # ── Step 8: Gate ──────────────────────────────────────────────────────
        gate = DisagreementGate(model=self.gate_model)
        if self.ablation.get("no_gate") or self.ablation.get("fixed_beta") is not None:
            # Gate is disabled; fixed β handled in predict_proba
            pass
        else:
            gate.fit(
                meta_X_full,
                p_stack_oof,
                p_weighted_oof,
                oof_true_arr,
                self.n_classes,
            )
        art.gate = gate

        self._artifacts = art

        # ── Step 9: Refit base models on full outer_train ─────────────────────
        logger.info("Refitting %d base models on full outer-training set…", len(self.base_models))
        for model in self.base_models:
            model.fit(X_train, y_train)

        logger.info("EnsembleCombiner.fit() complete.")
        return self

    def predict_proba_all(
        self,
        X: pd.DataFrame,
    ) -> Dict[str, np.ndarray]:
        """Return all probability paths for evaluation and ablation.

        Returns dict with keys:
          'p_weighted'   — calibrated weighted average
          'p_stack'      — stacker output
          'p_final'      — gated combination
          'base_{name}'  — raw base model proba (uncalibrated) for each model
          'cal_{name}'   — calibrated base model proba
        """
        art = self._check_fitted()

        # Base model probabilities
        raw_probas: List[np.ndarray] = []
        for model in self.base_models:
            raw_probas.append(model.predict_proba(X))

        # Calibrate
        cal_probas: List[np.ndarray] = []
        for model, raw_p in zip(self.base_models, raw_probas):
            cal = art.calibrators.get(model.name)
            if cal is not None:
                cal_probas.append(cal.predict_proba(raw_p))
            else:
                cal_probas.append(raw_p)

        # Path A: weighted average
        p_weighted = art.averager.predict_proba(cal_probas)

        # Meta-features (no path gap at first pass)
        simplified = bool(self.ablation.get("simplified_meta_features"))
        meta_no_gap = build_meta_features(cal_probas, simplified=simplified)

        # Path B: stacker (align to full n_classes space)
        p_stack_raw = art.stacker.predict_proba(meta_no_gap)
        if p_stack_raw.shape[1] == self.n_classes:
            p_stack = p_stack_raw
        else:
            p_stack = np.zeros((len(X), self.n_classes))
            stacker_classes = art.stacker._stacker.classes_
            for col_idx, cls in enumerate(stacker_classes):
                p_stack[:, cls] = p_stack_raw[:, col_idx]
            row_sums = p_stack.sum(axis=1, keepdims=True)
            p_stack = p_stack / np.where(row_sums == 0, 1.0, row_sums)

        # Full meta-features with path gap
        meta_full = build_meta_features(
            cal_probas,
            model_names=[m.name for m in self.base_models],
            p_weighted=p_weighted,
            p_stack=p_stack,
            simplified=simplified,
        )

        # Gate combination
        if self.ablation.get("no_gate"):
            fixed_beta = 0.5
            beta = np.full(len(X), fixed_beta)
            p_final = beta.reshape(-1, 1) * p_stack + (1 - beta.reshape(-1, 1)) * p_weighted
        elif self.ablation.get("fixed_beta") is not None:
            fixed_beta = float(self.ablation["fixed_beta"])
            beta = np.full(len(X), fixed_beta)
            p_final = beta.reshape(-1, 1) * p_stack + (1 - beta.reshape(-1, 1)) * p_weighted
        else:
            p_final = art.gate.combine(p_stack, p_weighted, meta_full)

        out = {
            "p_weighted": p_weighted,
            "p_stack": p_stack,
            "p_final": p_final,
        }
        for model, raw_p, cal_p in zip(self.base_models, raw_probas, cal_probas):
            out[f"base_{model.name}"] = raw_p
            out[f"cal_{model.name}"] = cal_p
        return out

    def get_artifacts(self) -> EnsembleArtifacts:
        return self._check_fitted()

    # ──────────────────────────────────────────────────────────────────────────

    def _check_fitted(self) -> EnsembleArtifacts:
        if self._artifacts is None:
            raise RuntimeError("EnsembleCombiner.fit() must be called first.")
        return self._artifacts

    def _get_stacker_oof(
        self,
        meta_X: np.ndarray,
        y: np.ndarray,
        stacker_template: MetaStacker,
    ) -> np.ndarray:
        """Get stacker OOF predictions by 3-fold CV on meta-features."""
        from sklearn.model_selection import StratifiedKFold
        from src.utils.seeds import get_seed

        n = len(y)
        oof = np.zeros((n, self.n_classes))
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=get_seed("stacker_oof"))

        for tr_idx, val_idx in skf.split(meta_X, y):
            s = MetaStacker(C=stacker_template.C, allow_gb_fallback=False)
            s.fit(meta_X[tr_idx], y[tr_idx], self.n_classes)
            fold_proba = s.predict_proba(meta_X[val_idx])
            fold_classes = s._stacker.classes_
            # Align to full class space (rare classes absent from fold → 0)
            for col_idx, cls in enumerate(fold_classes):
                oof[val_idx, cls] = fold_proba[:, col_idx]

        # Re-normalise rows (handle zero rows from absent classes)
        row_sums = oof.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        return oof / row_sums
