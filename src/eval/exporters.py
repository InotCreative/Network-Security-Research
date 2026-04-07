"""
Artifact exporters.

Produces all required paper-facing artifacts from one pipeline run:
  - results_main_multiclass.csv / .json
  - ablation_results.csv
  - feature_stability.csv
  - selected_features_multiclass.txt / .json
  - calibration_report.json + reliability diagram plots
  - model_cards/*.md
  - run_manifest.yaml (written by manifests.py)

All paths are relative to the run's artifact directory.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.eval.metrics import confusion_matrix_dict
from src.eval.plots import (
    plot_confidence_histogram,
    plot_confusion_matrix,
    plot_macro_f1_comparison,
    plot_per_class_metrics,
    plot_reliability_diagram,
)
from src.eval.reliability import reliability_diagram_data
from src.utils.manifests import RunManifest
from src.utils.serialization import save_json

logger = logging.getLogger(__name__)


class ResultsExporter:
    """Collects fold results and writes all paper artifacts at run end.

    Usage:
        exporter = ResultsExporter(artifact_dir, reports_dir, manifest)
        # Per-fold:
        exporter.add_fold_result(fold_idx, method_name, metrics_dict, proba, y_true)
        # At end:
        exporter.export_all(class_names)
    """

    def __init__(
        self,
        artifact_dir: Path,
        reports_dir: Path,
        manifest: RunManifest,
    ) -> None:
        self.artifact_dir = Path(artifact_dir)
        self.reports_dir = Path(reports_dir)
        self.manifest = manifest
        self._fold_results: List[dict] = []   # raw per-fold metric dicts
        self._final_probas: Dict[str, np.ndarray] = {}  # method → proba (test)
        self._final_y: Optional[np.ndarray] = None
        self._calibration_data: Dict[str, dict] = {}

    def add_fold_result(
        self,
        fold_idx: int,
        method: str,
        metrics: Dict,
        proba: Optional[np.ndarray] = None,
        y_true: Optional[np.ndarray] = None,
    ) -> None:
        row = {"fold": fold_idx, "method": method, **metrics}
        self._fold_results.append(row)

    def set_final_test_result(
        self,
        method: str,
        proba: np.ndarray,
        y_true: np.ndarray,
    ) -> None:
        self._final_probas[method] = proba
        self._final_y = y_true

    def add_calibration_data(self, model_name: str, cal_data: dict) -> None:
        self._calibration_data[model_name] = cal_data

    # ──────────────────────────────────────────────────────────────────────────

    def export_all(self, class_names: List[str]) -> None:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        self._export_fold_results()
        self._export_statistical_tests()
        self._export_test_metrics(class_names)
        self._export_calibration_report(class_names)
        logger.info("All artifacts exported to %s", self.artifact_dir)

    def export_feature_artifacts(
        self,
        selection_result,   # ConsensusSelector.SelectionResult
        fold_idx: int,
    ) -> None:
        """Write feature stability and selected features artifacts."""
        out_dir = self.artifact_dir / f"fold_{fold_idx}"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Feature stability CSV
        stability_df = pd.DataFrame(selection_result.feature_manifest)
        stab_path = out_dir / "feature_stability.csv"
        stability_df.to_csv(stab_path, index=False)
        self.manifest.register(
            "feature_stability", stab_path,
            "Feature stability and consensus scores per fold",
            produced_by="ConsensusSelector",
            metadata={"fold": fold_idx, "chosen_k": selection_result.chosen_k},
        )

        # Selected features
        sel_txt = out_dir / "selected_features.txt"
        sel_txt.write_text("\n".join(selection_result.selected_names), encoding="utf-8")

        sel_json = out_dir / "selected_features.json"
        save_json({
            "chosen_k": selection_result.chosen_k,
            "selected_features": selection_result.selected_names,
            "full_ranking": selection_result.full_ranking,
            "consensus_scores": {k: round(v, 6) for k, v in selection_result.consensus_scores.items()},
            "selector_weights": {k: round(v, 6) for k, v in selection_result.selector_weights.items()},
            "selector_stabilities": {k: round(v, 6) for k, v in selection_result.selector_stabilities.items()},
        }, sel_json)

        self.manifest.register(
            "selected_features", sel_json,
            "Consensus-selected feature set with rankings",
            produced_by="ConsensusSelector",
            metadata={"fold": fold_idx, "n_selected": len(selection_result.selected_names)},
        )

    def write_model_card(
        self,
        model_name: str,
        config: dict,
        metrics: dict,
        fold_idx: int,
    ) -> None:
        card_dir = self.artifact_dir / "model_cards"
        card_dir.mkdir(parents=True, exist_ok=True)
        card_path = card_dir / f"{model_name}_fold{fold_idx}.md"

        lines = [
            f"# Model Card: {model_name} (Fold {fold_idx})\n",
            "## Configuration\n",
            "```yaml",
        ]
        for k, v in config.items():
            lines.append(f"{k}: {v}")
        lines += ["```\n", "## Metrics\n"]
        for k, v in metrics.items():
            if not isinstance(v, dict):
                lines.append(f"- **{k}**: {v}")
        lines.append("\n")

        card_path.write_text("\n".join(lines), encoding="utf-8")
        self.manifest.register(
            f"model_card_{model_name}_fold{fold_idx}",
            card_path,
            "Model card with config and metrics",
            produced_by=model_name,
            metadata={"fold": fold_idx},
        )

    # ──────────────────────────────────────────────────────────────────────────

    def _export_statistical_tests(self) -> None:
        """Pairwise paired t-tests and Wilcoxon signed-rank tests across folds.

        Compares p_final against every other method on macro_f1.
        Produces a paper-ready significance table.
        """
        if not self._fold_results:
            return
        from scipy import stats as sp_stats

        df = pd.DataFrame(self._fold_results)
        if "macro_f1" not in df.columns:
            return

        proposed = "p_final"
        baselines = [m for m in df["method"].unique() if m != proposed]
        if proposed not in df["method"].unique():
            return

        f1_proposed = df[df.method == proposed].sort_values("fold")["macro_f1"].values
        rows = []
        for baseline in baselines:
            sub = df[df.method == baseline].sort_values("fold")["macro_f1"].values
            if len(sub) != len(f1_proposed):
                continue
            diff = f1_proposed - sub
            mean_diff = float(diff.mean())
            std_diff = float(diff.std())
            t_stat, t_p = sp_stats.ttest_rel(f1_proposed, sub)
            try:
                w_stat, w_p = sp_stats.wilcoxon(f1_proposed, sub, alternative="greater")
            except ValueError:
                w_stat, w_p = float("nan"), float("nan")
            rows.append({
                "baseline": baseline,
                "mean_diff": round(mean_diff, 6),
                "std_diff": round(std_diff, 6),
                "t_statistic": round(float(t_stat), 4),
                "t_pvalue": round(float(t_p), 4),
                "wilcoxon_statistic": round(float(w_stat), 4) if not np.isnan(w_stat) else "n/a",
                "wilcoxon_pvalue": round(float(w_p), 4) if not np.isnan(w_p) else "n/a",
                "significant_005": bool(t_p < 0.05),
                "significant_010": bool(t_p < 0.10),
            })

        if rows:
            sig_df = pd.DataFrame(rows)
            path = self.artifact_dir / "statistical_tests.csv"
            sig_df.to_csv(path, index=False)
            save_json(rows, self.artifact_dir / "statistical_tests.json")
            self.manifest.register(
                "statistical_tests", path,
                "Pairwise significance tests: p_final vs all baselines",
                produced_by="exporters",
            )
            logger.info(
                "Statistical tests:\n%s",
                sig_df[["baseline", "mean_diff", "t_pvalue", "significant_005"]].to_string(index=False),
            )

    def _export_fold_results(self) -> None:
        if not self._fold_results:
            return
        df = pd.DataFrame(self._fold_results)
        path = self.artifact_dir / "fold_results.csv"
        df.to_csv(path, index=False)
        self.manifest.register(
            "fold_results", path, "Per-fold metrics for all methods",
            produced_by="fold_runner",
        )

        # Aggregated means/stds
        numeric_cols = df.select_dtypes(include="number").columns.difference(["fold"])
        agg = df.groupby("method")[numeric_cols].agg(["mean", "std"]).round(6)
        agg_path = self.artifact_dir / "results_main_multiclass.csv"
        agg.to_csv(agg_path)
        self.manifest.register(
            "results_main_multiclass", agg_path,
            "Aggregated cross-fold metrics for paper table",
            produced_by="fold_runner",
        )
        # JSON summary
        summary = {}
        for method, grp in df.groupby("method"):
            summary[method] = {
                col: {"mean": float(grp[col].mean()), "std": float(grp[col].std())}
                for col in numeric_cols
                if col in grp.columns
            }
        save_json(summary, self.artifact_dir / "results_main_multiclass_summary.json")

    def _export_test_metrics(self, class_names: List[str]) -> None:
        if not self._final_probas or self._final_y is None:
            return

        from src.eval.metrics import multiclass_metrics
        test_results = {}
        for method, proba in self._final_probas.items():
            m = multiclass_metrics(self._final_y, proba, class_names, prefix=method)
            test_results[method] = m
            y_pred = np.argmax(proba, axis=1)
            cm = confusion_matrix_dict(self._final_y, y_pred, class_names)

            # Confusion matrix plot
            cm_path = self.reports_dir / f"cm_{method}.png"
            plot_confusion_matrix(cm, cm_path)
            self.manifest.register(
                f"confusion_matrix_{method}", cm_path,
                "Confusion matrix heatmap",
                produced_by="exporters",
            )

            # Per-class bar chart
            pc_path = self.reports_dir / f"per_class_{method}.png"
            plot_per_class_metrics(m, class_names, pc_path)

            # Confidence histogram
            conf_path = self.reports_dir / f"confidence_{method}.png"
            plot_confidence_histogram(proba, method, conf_path)

        df = pd.DataFrame(test_results).T
        path = self.artifact_dir / "results_test_locked.csv"
        df.to_csv(path)
        save_json(test_results, self.artifact_dir / "results_test_locked.json")
        self.manifest.register(
            "results_test_locked", path,
            "Locked test-set metrics (final evaluation only)",
            produced_by="exporters",
        )

        # Macro F1 comparison bar chart
        f1_dict = {m: v.get(f"{m}_macro_f1", v.get("macro_f1", 0)) for m, v in test_results.items()}
        plot_macro_f1_comparison(
            f1_dict,
            self.reports_dir / "macro_f1_comparison.png",
        )

    def _export_calibration_report(self, class_names: List[str]) -> None:
        if not self._calibration_data:
            return
        save_json(self._calibration_data, self.artifact_dir / "calibration_report.json")
        for model_name, data in self._calibration_data.items():
            if "reliability" in data:
                rel_path = self.reports_dir / f"reliability_{model_name}.png"
                plot_reliability_diagram(data["reliability"], model_name, rel_path)
                self.manifest.register(
                    f"reliability_diagram_{model_name}", rel_path,
                    "Reliability diagram",
                    produced_by="exporters",
                )
