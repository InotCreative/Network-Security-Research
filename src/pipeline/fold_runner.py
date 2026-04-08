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
from src.models.base_models import build_default_models, build_models_from_params
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

        # ── Scientific integrity: warn if ensemble fails to beat stacker ─────
        self._check_ensemble_vs_stacker(exporter)

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
            active_selector_names=self.config.get("selector_names"),
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

        # ── 3. Hyperparameter search (optional, fit on outer_train only) ──────
        base_models = self._build_base_models(X_tr_sel, y_tr, fold_idx)
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

    def _check_ensemble_vs_stacker(self, exporter: "ResultsExporter") -> None:
        """Scientific integrity check required by CLAUDE.md.

        If the gated ensemble (p_final) fails to beat plain stacking (p_stack)
        on mean macro F1 across outer folds, emit a prominent WARNING.
        Do NOT suppress or hide this — report honestly in the paper.
        """
        df = pd.DataFrame(exporter._fold_results)
        if "macro_f1" not in df.columns:
            return

        proposed = df[df.method == "p_final"].sort_values("fold")["macro_f1"].values
        stacker = df[df.method == "p_stack"].sort_values("fold")["macro_f1"].values
        wa = df[df.method == "p_weighted"].sort_values("fold")["macro_f1"].values

        if len(proposed) == 0 or len(stacker) == 0:
            logger.warning("Could not find p_final or p_stack in fold results — skipping integrity check.")
            return

        mean_vs_stack = float(proposed.mean()) - float(stacker.mean())
        mean_vs_wa = float(proposed.mean()) - float(wa.mean()) if len(wa) > 0 else None

        if mean_vs_stack < 0:
            logger.warning(
                "\n"
                "╔══════════════════════════════════════════════════════════════════╗\n"
                "║  ⚠  SCIENTIFIC INTEGRITY WARNING                                ║\n"
                "║                                                                  ║\n"
                "║  p_final (gated ensemble) FAILED to beat p_stack (plain         ║\n"
                "║  stacker) on multiclass macro F1.                                ║\n"
                "║                                                                  ║\n"
                "║  p_final mean macro_F1 = %.4f                                  ║\n"
                "║  p_stack  mean macro_F1 = %.4f                                  ║\n"
                "║  Difference             = %.4f (ensemble − stacker)             ║\n"
                "║                                                                  ║\n"
                "║  Per-fold results: %s\n"
                "║                                                                  ║\n"
                "║  ACTION REQUIRED: Do NOT adjust code to hide this result.       ║\n"
                "║  Report it honestly in the paper. Investigate gate behaviour    ║\n"
                "║  and whether calibration is genuinely improving p_weighted.     ║\n"
                "╚══════════════════════════════════════════════════════════════════╝",
                proposed.mean(), stacker.mean(), mean_vs_stack,
                str((proposed - stacker).round(4)),
            )
            self.manifest.register(
                "integrity_warning",
                self.artifact_dir / "INTEGRITY_WARNING.txt",
                "Ensemble failed to beat plain stacker on macro F1",
                produced_by="fold_runner",
                metadata={
                    "p_final_mean": round(float(proposed.mean()), 6),
                    "p_stack_mean": round(float(stacker.mean()), 6),
                    "diff": round(float(mean_vs_stack), 6),
                },
            )
            # Write a plain-text warning file so it is visible in the artifact dir
            warning_txt = (
                f"INTEGRITY WARNING\n"
                f"p_final mean macro_F1 = {proposed.mean():.6f}\n"
                f"p_stack mean macro_F1 = {stacker.mean():.6f}\n"
                f"Difference = {mean_vs_stack:.6f} (ensemble failed to beat stacker)\n"
            )
            (Path(self.artifact_dir) / "INTEGRITY_WARNING.txt").write_text(
                warning_txt, encoding="utf-8"
            )
        else:
            logger.info(
                "Integrity check PASSED: p_final (%.4f) beats p_stack (%.4f) "
                "by %.4f macro F1 over %d outer folds.",
                proposed.mean(), stacker.mean(), mean_vs_stack, len(proposed),
            )
            if mean_vs_wa is not None and mean_vs_wa < 0:
                logger.warning(
                    "Note: p_final (%.4f) does not beat p_weighted (%.4f) — "
                    "the stacker path may be dominating gating.",
                    proposed.mean(), wa.mean(),
                )

    def _build_base_models(
        self,
        X_sel: pd.DataFrame,
        y: np.ndarray,
        fold_idx,   # int for outer folds, "final" for full-training refit
    ) -> list:
        """Build base models, optionally running HP search first.

        When hp_search.enabled is True in the config, RandomizedSearchCV is
        run on (X_sel, y) inside this outer fold before model construction.
        Tuned parameters are saved to artifacts/fold_{fold_idx}/hp_search_results.json
        and registered with the manifest.
        """
        hp_cfg = self.config.get("hp_search", {})
        if not hp_cfg.get("enabled", False):
            return build_default_models()

        from src.models.tuner import HyperparameterTuner
        tuner = HyperparameterTuner(
            n_iter=hp_cfg.get("n_iter", 20),
            n_cv=hp_cfg.get("n_cv", 3),
            scoring=hp_cfg.get("scoring", "f1_macro"),
        )
        logger.info(
            "Fold %d: Running HP search (n_iter=%d, n_cv=%d) on %d samples, %d features.",
            fold_idx, tuner.n_iter, tuner.n_cv, len(y), X_sel.shape[1],
        )
        best_params = tuner.tune_all(X_sel, y, n_classes=self._n_classes)

        # Persist tuned parameters as a traceable artifact
        fold_dir = Path(self.artifact_dir) / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        hp_path = fold_dir / "hp_search_results.json"
        save_json(best_params, hp_path)
        self.manifest.register(
            f"hp_search_fold{fold_idx}", hp_path,
            "Tuned base-model hyperparameters from RandomizedSearchCV",
            produced_by="HyperparameterTuner",
            metadata={"fold": fold_idx, "n_iter": tuner.n_iter, "n_cv": tuner.n_cv},
        )
        logger.info("Fold %d: HP search complete. Tuned params: %s", fold_idx, best_params)
        return build_models_from_params(best_params)

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
            active_selector_names=self.config.get("selector_names"),
        )
        sel_result = selector.fit(X_eng, y_train)

        sel_cols = sel_result.selected_names
        X_sel = X_eng[sel_cols] if not self.ablation.get("no_feature_selection") else X_eng

        if self.ablation.get("no_engineered_features"):
            from src.data.schema import FEATURE_COLS
            base_cols = [c for c in FEATURE_COLS if c in X_sel.columns]
            X_sel = X_sel[base_cols]

        base_models = self._build_base_models(X_sel, y_train, fold_idx="final")
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
