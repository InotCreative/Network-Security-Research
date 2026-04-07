"""
Outer cross-validation fold runner.

For each outer fold on the training set:
  1. Split outer_train | outer_valid.
  2. Fit FeatureEngineer inside the fold.
  3. Run ConsensusSelector inside the fold.
  4. Restrict to selected features.
  5. Train EnsembleCombiner (includes inner OOF collection, calibration,
     stacker, gate, base-model refit).
  6. Evaluate all methods on outer_valid.
  7. Emit per-fold metrics + artifacts.

After all folds:
  - Refit on full training set.
  - Evaluate on the locked test set.
  - Export all paper artifacts.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.data.loader import load_official_splits
from src.data.schema import MULTICLASS_LABELS
from src.ensemble.combiner import EnsembleCombiner
from src.eval.exporters import ResultsExporter
from src.eval.metrics import multiclass_metrics, sanity_check_accuracy
from src.eval.reliability import reliability_diagram_data
from src.features.engineer import FeatureEngineer
from src.models.base_models import build_default_models
from src.select.consensus import ConsensusSelector
from src.utils.manifests import RunManifest
from src.utils.seeds import get_seed, get_global_seed, seed_summary
from src.utils.serialization import save_json, save_model

logger = logging.getLogger(__name__)


class FoldRunner:
    """Outer K-fold evaluation loop.

    Parameters
    ----------
    config:
        Parsed experiment configuration dict.
    manifest:
        RunManifest instance for artifact registration.
    """

    def __init__(self, config: dict, manifest: RunManifest) -> None:
        self.config = config
        self.manifest = manifest
        self.n_outer_folds: int = config.get("n_outer_folds", 5)
        self.n_oof_splits: int = config.get("n_oof_splits", 5)
        self.task: str = config.get("task", "multiclass")
        self.ablation: dict = config.get("ablation", {})
        self.artifact_dir = Path(config.get("artifact_dir", "artifacts"))
        self.reports_dir = Path(config.get("reports_dir", "reports"))
        self._class_names = MULTICLASS_LABELS if self.task == "multiclass" else ["Normal", "Attack"]
        self._n_classes = len(self._class_names)

    def run(self) -> None:
        """Execute full outer CV + final test evaluation."""
        X_train, y_train, X_test, y_test = load_official_splits(
            train_path=self.config.get("train_path", "data/UNSW_NB15_training-set.csv"),
            test_path=self.config.get("test_path", "data/UNSW_NB15_testing-set.csv"),
            task=self.task,
        )

        exporter = ResultsExporter(self.artifact_dir, self.reports_dir, self.manifest)

        outer_skf = StratifiedKFold(
            n_splits=self.n_outer_folds,
            shuffle=True,
            random_state=get_seed("outer_cv"),
        )

        for fold_idx, (tr_idx, val_idx) in enumerate(
            outer_skf.split(X_train, y_train)
        ):
            logger.info(
                "═══ Outer fold %d/%d  (train=%d, val=%d) ═══",
                fold_idx + 1, self.n_outer_folds, len(tr_idx), len(val_idx),
            )
            X_outer_tr = X_train.iloc[tr_idx].reset_index(drop=True)
            y_outer_tr = y_train.values[tr_idx]
            X_outer_val = X_train.iloc[val_idx].reset_index(drop=True)
            y_outer_val = y_train.values[val_idx]

            fold_results = self._run_fold(
                fold_idx, X_outer_tr, y_outer_tr, X_outer_val, y_outer_val, exporter
            )
            for method, metrics in fold_results.items():
                exporter.add_fold_result(fold_idx, method, metrics)

        # ── Final evaluation on locked test set ────────────────────────────────
        logger.info("═══ Final evaluation on locked test set ═══")
        final_ensemble = self._train_final_ensemble(X_train, y_train.values, exporter)
        self._evaluate_test(final_ensemble, X_test, y_test.values, exporter)

        # ── Export all artifacts ───────────────────────────────────────────────
        exporter.export_all(self._class_names)
        self.manifest.save(self.artifact_dir)
        logger.info("Run complete. Seed registry: %s", seed_summary())

    # ──────────────────────────────────────────────────────────────────────────

    def _run_fold(
        self,
        fold_idx: int,
        X_tr: pd.DataFrame, y_tr: np.ndarray,
        X_val: pd.DataFrame, y_val: np.ndarray,
        exporter: ResultsExporter,
    ) -> Dict[str, dict]:
        """Train and evaluate all methods for one outer fold."""

        # ── 1. Feature engineering (fit on outer_train only) ──────────────────
        eng = FeatureEngineer()
        X_tr_eng = eng.fit_transform(X_tr)
        X_val_eng = eng.transform(X_val)

        # ── 2. Consensus feature selection (fit on outer_train only) ──────────
        selector = ConsensusSelector(
            k_grid=self.config.get("k_grid", None),
            n_inner_folds=self.config.get("n_inner_folds", 3),
            use_stability_weighting=not self.ablation.get("no_stability_weighting", False),
        )
        sel_result = selector.fit(X_tr_eng, y_tr)
        exporter.export_feature_artifacts(sel_result, fold_idx)

        # Restrict to selected features
        sel_cols = sel_result.selected_names
        X_tr_sel = X_tr_eng[sel_cols] if not self.ablation.get("no_feature_selection") else X_tr_eng
        X_val_sel = X_val_eng[sel_cols] if not self.ablation.get("no_feature_selection") else X_val_eng

        if self.ablation.get("no_engineered_features"):
            from src.data.schema import FEATURE_COLS
            base_cols = [c for c in FEATURE_COLS if c in X_tr_sel.columns]
            X_tr_sel = X_tr_sel[base_cols]
            X_val_sel = X_val_sel[base_cols]

        logger.info(
            "Fold %d: selected %d features. Top-5: %s",
            fold_idx, len(sel_cols), sel_cols[:5],
        )

        # ── 3. Build and train ensemble ────────────────────────────────────────
        base_models = build_default_models()
        combiner = EnsembleCombiner(
            base_models=base_models,
            n_classes=self._n_classes,
            n_oof_splits=self.n_oof_splits,
            gate_model=self.config.get("gate_model", "auto"),
            ablation=self.ablation,
        )
        combiner.fit(X_tr_sel, y_tr)

        # ── 4. Evaluate all methods on outer_valid ─────────────────────────────
        with warnings.catch_warnings(record=True):
            all_probas = combiner.predict_proba_all(X_val_sel)

        fold_metrics: Dict[str, dict] = {}
        for method, proba in all_probas.items():
            if proba.shape[1] < self._n_classes:
                continue
            y_pred = np.argmax(proba, axis=1)
            sanity_check_accuracy(f"Fold {fold_idx} / {method}", y_val, y_pred)
            m = multiclass_metrics(proba=proba, y_true=y_val, labels=self._class_names)
            fold_metrics[method] = m
            logger.info(
                "Fold %d | %-30s macro_F1=%.4f  weighted_F1=%.4f  ECE=%.4f",
                fold_idx, method,
                m["macro_f1"], m["weighted_f1"], m["ece"],
            )

        # ── 5. Calibration artifacts ───────────────────────────────────────────
        art = combiner.get_artifacts()
        for mname, cal in art.calibrators.items():
            if cal is None:
                continue
            # Need calibrated proba on val for reliability diagram
            val_proba = all_probas.get(f"cal_{mname}")
            if val_proba is not None:
                rel_data = reliability_diagram_data(y_val, val_proba)
                exporter.add_calibration_data(mname, {
                    "method": cal.chosen_method,
                    "fold": fold_idx,
                    "reliability": rel_data,
                })

        # ── 6. Model cards ─────────────────────────────────────────────────────
        for model in base_models:
            exporter.write_model_card(
                model.name,
                config={
                    "family": model.family,
                    "fold": fold_idx,
                    "calibration": art.chosen_calibration_methods.get(model.name, "n/a"),
                    "weight": art.model_weights.get(model.name, "n/a"),
                    "global_seed": get_global_seed(),
                },
                metrics=fold_metrics.get(f"cal_{model.name}", {}),
                fold_idx=fold_idx,
            )

        return fold_metrics

    def _train_final_ensemble(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        exporter: ResultsExporter,
    ) -> "EnsembleCombiner":
        """Refit on the full training set for locked test evaluation."""
        eng = FeatureEngineer()
        X_eng = eng.fit_transform(X_train)

        selector = ConsensusSelector(
            k_grid=self.config.get("k_grid", None),
            n_inner_folds=self.config.get("n_inner_folds", 3),
            use_stability_weighting=not self.ablation.get("no_stability_weighting", False),
        )
        sel_result = selector.fit(X_eng, y_train)

        sel_cols = sel_result.selected_names
        X_sel = X_eng[sel_cols] if not self.ablation.get("no_feature_selection") else X_eng

        if self.ablation.get("no_engineered_features"):
            from src.data.schema import FEATURE_COLS
            base_cols = [c for c in FEATURE_COLS if c in X_sel.columns]
            X_sel = X_sel[base_cols]

        base_models = build_default_models()
        combiner = EnsembleCombiner(
            base_models=base_models,
            n_classes=self._n_classes,
            n_oof_splits=self.n_oof_splits,
            gate_model=self.config.get("gate_model", "auto"),
            ablation=self.ablation,
        )
        combiner.fit(X_sel, y_train)

        # Save the final combiner
        model_path = self.artifact_dir / "final_ensemble.joblib"
        save_model(combiner, model_path)
        self.manifest.register(
            "final_ensemble", model_path,
            "Final ensemble trained on full training set",
            produced_by="fold_runner",
        )

        # Save feature engineering + selection metadata
        feat_meta = {
            "engineered_features": eng.feature_manifest(),
            "clip_bounds": eng.get_clip_bounds(),
            "selected_features": sel_result.selected_names,
            "chosen_k": sel_result.chosen_k,
            "selector_weights": sel_result.selector_weights,
        }
        save_json(feat_meta, self.artifact_dir / "final_feature_metadata.json")

        # Store eng and selector on combiner so test transform can use them
        combiner._final_engineer = eng
        combiner._final_sel_cols = X_sel.columns.tolist()
        combiner._final_ablation = self.ablation
        return combiner

    def _evaluate_test(
        self,
        combiner: "EnsembleCombiner",
        X_test: pd.DataFrame,
        y_test: np.ndarray,
        exporter: ResultsExporter,
    ) -> None:
        eng = combiner._final_engineer
        sel_cols = combiner._final_sel_cols
        ablation = combiner._final_ablation

        X_test_eng = eng.transform(X_test)
        X_test_sel = X_test_eng[sel_cols] if not ablation.get("no_feature_selection") else X_test_eng

        if ablation.get("no_engineered_features"):
            from src.data.schema import FEATURE_COLS
            base_cols = [c for c in FEATURE_COLS if c in X_test_sel.columns]
            X_test_sel = X_test_sel[base_cols]

        all_probas = combiner.predict_proba_all(X_test_sel)

        for method, proba in all_probas.items():
            if proba.shape[1] < self._n_classes:
                continue
            y_pred = np.argmax(proba, axis=1)
            sanity_check_accuracy(f"TEST / {method}", y_test, y_pred)
            exporter.set_final_test_result(method, proba, y_test)

        logger.info("Locked test evaluation complete.")
