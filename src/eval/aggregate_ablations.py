"""
Ablation results aggregator.

Scans all subdirectories of an artifact root, identifies each run's
experiment label from its run_manifest.yaml, reads the p_final metrics from
results_main_multiclass_summary.json, and writes:

  - ablation_results.csv   — paper-ready comparison table (primary key metric per experiment)
  - ablation_results.json  — machine-readable version of the same
  - ablation_f1_comparison.png — bar chart of macro F1 across all experiments

This produces the ``ablation_results.csv`` required output artifact from CLAUDE.md.
"No result table should require manual copying from notebooks."

Usage
-----
    python -m src.eval.aggregate_ablations \\
        --artifacts-dir artifacts \\
        --output-dir reports

Called automatically at the end of run_all.sh after all experiments complete.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# ── Experiment catalogue ───────────────────────────────────────────────────────
# Maps the config file stem → human-readable label for the paper table.
# Any experiment not listed here is labelled by its config stem.

EXPERIMENT_LABELS: Dict[str, str] = {
    "multiclass_main":           "Full system (gated ensemble)",
    "binary_sanity":             "Binary sanity check",
    "no_calibration":            "No calibration",
    "no_gate":                   "No gate (β = 0.5, fixed)",
    "stacker_only":              "Stacker only (β = 1.0)",
    "weighted_avg_only":         "Weighted avg only (β = 0.0)",
    "no_feature_selection":      "No feature selection",
    "no_engineered_features":    "No engineered features",
    "simplified_meta_features":  "Simplified meta-features",
    "no_stability_weighting":    "No stability weighting",
    "single_selector":           "Single selector (mutual info only)",
}

# Row order in the output table. Rows not listed appear after these, alphabetically.
DISPLAY_ORDER: List[str] = [
    "multiclass_main",
    # Calibration
    "no_calibration",
    # Path-mixing comparison (β sweep)
    "stacker_only",
    "no_gate",
    "weighted_avg_only",
    # Meta-feature design
    "simplified_meta_features",
    # Feature engineering / selection
    "no_engineered_features",
    "no_feature_selection",
    "no_stability_weighting",
    "single_selector",
]

# Primary method to extract per experiment (always p_final for the proposed system).
# For ablation experiments this is still p_final, which reflects the modified design.
PRIMARY_METHOD = "p_final"

# Metrics to include in the output table (from results_main_multiclass_summary.json).
# Keys must match exactly what multiclass_metrics() emits in src/eval/metrics.py.
METRICS = [
    "macro_f1",
    "weighted_f1",
    "accuracy",
    "log_loss",
    "ece",
    "roc_auc_macro_ovr",
]


# ── Core aggregation ───────────────────────────────────────────────────────────

def find_run_dirs(artifacts_root: Path) -> List[Path]:
    """Return all subdirectories that contain a run_manifest.yaml."""
    return sorted(
        p.parent
        for p in artifacts_root.rglob("run_manifest.yaml")
    )


def parse_manifest(run_dir: Path) -> Optional[dict]:
    """Load run_manifest.yaml from a run directory."""
    manifest_path = run_dir / "run_manifest.yaml"
    if not manifest_path.exists():
        return None
    try:
        with open(manifest_path, encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as exc:
        logger.warning("Could not parse manifest %s: %s", manifest_path, exc)
        return None


def config_stem(config_path: str) -> str:
    """Extract the config file stem (without extension) from a path string.

    Handles both POSIX and Windows separators, since manifests recorded on
    Windows store paths with backslashes that pathlib.PurePosixPath does not
    split on.
    """
    normalised = config_path.replace("\\", "/")
    return Path(normalised).stem


def load_primary_metrics(run_dir: Path, method: str = PRIMARY_METHOD) -> Optional[dict]:
    """
    Load macro-averaged metrics for ``method`` from results_main_multiclass_summary.json.

    Falls back to results_test_locked.json if the CV summary is absent.
    Returns None if neither file exists or method is not present.
    """
    summary_path = run_dir / "results_main_multiclass_summary.json"
    if summary_path.exists():
        try:
            with open(summary_path, encoding="utf-8") as f:
                data = json.load(f)
            if method in data:
                return data[method]
        except Exception as exc:
            logger.warning("Could not read %s: %s", summary_path, exc)

    # Fallback: test-set locked results
    test_path = run_dir / "results_test_locked.json"
    if test_path.exists():
        try:
            with open(test_path, encoding="utf-8") as f:
                data = json.load(f)
            if method in data:
                # Test-set results are flat dicts; wrap for consistency
                return {k: {"mean": v, "std": None} for k, v in data[method].items()}
        except Exception as exc:
            logger.warning("Could not read %s: %s", test_path, exc)

    return None


def aggregate(
    artifacts_root: Path,
) -> pd.DataFrame:
    """
    Scan ``artifacts_root``, collect one row per experiment, return DataFrame.

    Columns: experiment, label, run_id, config, macro_f1_mean, macro_f1_std, ...
    """
    run_dirs = find_run_dirs(artifacts_root)
    if not run_dirs:
        logger.warning("No runs found under %s", artifacts_root)
        return pd.DataFrame()

    rows = []
    # Track which config stem → most recent run (by run_id timestamp prefix)
    seen: Dict[str, Tuple[str, dict]] = {}  # stem → (run_id, row_dict)

    for run_dir in run_dirs:
        manifest = parse_manifest(run_dir)
        if manifest is None:
            continue
        run_id = manifest.get("run_id", run_dir.name)
        cfg_path = manifest.get("config_path", "")
        stem = config_stem(cfg_path)

        metrics = load_primary_metrics(run_dir)
        if metrics is None:
            logger.debug("No primary metrics found in %s (run_id=%s)", run_dir, run_id)
            continue

        row: dict = {
            "experiment": stem,
            "label": EXPERIMENT_LABELS.get(stem, stem),
            "run_id": run_id,
            "config": cfg_path,
        }
        for metric in METRICS:
            m_data = metrics.get(metric, {})
            if isinstance(m_data, dict):
                row[f"{metric}_mean"] = m_data.get("mean")
                row[f"{metric}_std"] = m_data.get("std")
            else:
                row[f"{metric}_mean"] = m_data
                row[f"{metric}_std"] = None

        # Keep only the most recent run for each config stem
        if stem not in seen:
            seen[stem] = (run_id, row)
        else:
            existing_run_id, _ = seen[stem]
            if run_id > existing_run_id:
                seen[stem] = (run_id, row)

    if not seen:
        logger.warning("No usable runs with metrics found under %s", artifacts_root)
        return pd.DataFrame()

    # Build final rows in display order
    rows = []
    ordered_stems = DISPLAY_ORDER + sorted(
        s for s in seen if s not in DISPLAY_ORDER
    )
    for stem in ordered_stems:
        if stem in seen:
            rows.append(seen[stem][1])

    return pd.DataFrame(rows)


# ── Cross-run integrity guard ──────────────────────────────────────────────────
# The per-run guard in fold_runner only checks `p_final ≥ p_stack` inside a
# single run. A second failure mode is an *ablation* beating the full system —
# e.g., `no_calibration` with mean macro F1 > `multiclass_main` mean macro F1.
# When this happens we write `ABLATION_WARNING.txt` next to the ablation CSV
# so it is visible in code review and manuscript preparation.

def check_ablation_integrity(
    df: pd.DataFrame,
    output_dir: Path,
    *,
    reference_stem: str = "multiclass_main",
    metric: str = "macro_f1",
) -> Optional[Path]:
    """Warn when any ablation's mean metric exceeds the full system by more
    than one combined-std.

    Returns the path to the warning file if written, else None.
    """
    if df.empty:
        return None

    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"
    if mean_col not in df.columns:
        return None

    ref_rows = df[df["experiment"] == reference_stem]
    if ref_rows.empty:
        logger.debug("No %s row found — skipping cross-run integrity check.", reference_stem)
        return None
    ref_row = ref_rows.iloc[0]
    ref_mean = float(ref_row[mean_col])
    ref_std = float(ref_row[std_col] or 0.0)

    offenders: List[dict] = []
    for _, row in df.iterrows():
        stem = row["experiment"]
        if stem == reference_stem:
            continue
        # Skip experiments that are intentionally a different task (e.g. binary)
        if stem == "binary_sanity":
            continue
        cand_mean = row.get(mean_col)
        cand_std = row.get(std_col) or 0.0
        if cand_mean is None:
            continue
        # Joint uncertainty (conservative, assumes independence across folds)
        import math
        combined_std = math.sqrt(ref_std ** 2 + (cand_std or 0.0) ** 2)
        delta = float(cand_mean) - ref_mean
        if delta > combined_std and delta > 0:
            offenders.append({
                "experiment": stem,
                "label": row.get("label", stem),
                "mean": round(float(cand_mean), 6),
                "std": round(float(cand_std or 0.0), 6),
                "delta_vs_reference": round(delta, 6),
                "combined_std": round(combined_std, 6),
            })

    if not offenders:
        logger.info(
            "Cross-run integrity PASSED: no ablation exceeds %s mean %s (%.4f ± %.4f) "
            "by more than combined std.",
            reference_stem, metric, ref_mean, ref_std,
        )
        return None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    warning_path = output_dir / "ABLATION_WARNING.txt"

    lines = [
        "ABLATION INTEGRITY WARNING",
        "",
        f"Reference: {reference_stem} {metric} = {ref_mean:.6f} ± {ref_std:.6f}",
        "",
        "The following ablations exceed the full system by more than the combined",
        "outer-fold standard deviation. Investigate before publication — the component",
        "being ablated may not be net-positive for this metric at this sample size.",
        "",
    ]
    for o in offenders:
        lines.append(
            f"  - {o['label']:<40s} "
            f"{metric} = {o['mean']:.4f} ± {o['std']:.4f}   "
            f"Δ = +{o['delta_vs_reference']:.4f}  (combined σ = {o['combined_std']:.4f})"
        )
    lines.append("")
    lines.append(
        "Note: this warning is informational. Ablations that beat the full system "
        "reveal real scientific findings that should be reported honestly in the paper."
    )

    warning_path.write_text("\n".join(lines), encoding="utf-8")
    logger.warning(
        "ABLATION WARNING: %d ablation(s) beat %s on %s. Details → %s",
        len(offenders), reference_stem, metric, warning_path,
    )
    return warning_path


# ── Output writers ─────────────────────────────────────────────────────────────

def write_ablation_results(
    df: pd.DataFrame,
    output_dir: Path,
) -> Tuple[Path, Path]:
    """Write ablation_results.csv and ablation_results.json."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "ablation_results.csv"
    json_path = output_dir / "ablation_results.json"

    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records", indent=2)

    logger.info("Ablation results → %s (%d experiments)", csv_path, len(df))
    return csv_path, json_path


def write_ablation_plot(
    df: pd.DataFrame,
    output_dir: Path,
) -> Optional[Path]:
    """Bar chart of macro F1 mean ± std for all experiments."""
    if df.empty or "macro_f1_mean" not in df.columns:
        return None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        labels = df["label"].tolist()
        means = df["macro_f1_mean"].fillna(0).tolist()
        stds = [
            v if v is not None else 0.0
            for v in df["macro_f1_std"].fillna(0).tolist()
        ]

        colors = []
        for stem in df["experiment"].tolist():
            if stem == "multiclass_main":
                colors.append("#d62728")    # red for proposed system
            elif "stacker" in stem or "weighted_avg" in stem or "no_gate" in stem:
                colors.append("#ff7f0e")    # orange for path-mixing ablations
            else:
                colors.append("#1f77b4")    # blue for component ablations

        fig, ax = plt.subplots(figsize=(12, max(4, len(labels) * 0.55)))
        y = np.arange(len(labels))
        bars = ax.barh(y, means, xerr=stds, color=colors, height=0.6,
                       capsize=3, ecolor="black", error_kw={"linewidth": 1})

        # Value labels
        for bar, mean in zip(bars, means):
            if mean:
                ax.text(
                    bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                    f"{mean:.4f}", va="center", ha="left", fontsize=8,
                )

        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("Macro F1 (mean ± std over outer folds)", fontsize=10)
        ax.set_title("Ablation study — p_final macro F1 comparison", fontsize=11)
        ax.set_xlim(0, min(1.05, (max(means) if means else 1.0) * 1.18))
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.3)

        from matplotlib.patches import Patch
        legend = [
            Patch(color="#d62728", label="Proposed system"),
            Patch(color="#ff7f0e", label="Path-mixing ablations (β sweep)"),
            Patch(color="#1f77b4", label="Component ablations"),
        ]
        ax.legend(handles=legend, fontsize=8, loc="lower right")

        plt.tight_layout()
        plot_path = output_dir / "ablation_f1_comparison.png"
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Ablation bar chart → %s", plot_path)
        return plot_path

    except Exception as exc:
        logger.warning("Could not generate ablation plot: %s", exc)
        return None


# ── CLI ────────────────────────────────────────────────────────────────────────

def main(artifacts_dir: str = "artifacts", output_dir: str = "reports") -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    artifacts_root = Path(artifacts_dir)
    out_dir = Path(output_dir)

    df = aggregate(artifacts_root)
    if df.empty:
        logger.error(
            "No experiment results found under %s. "
            "Run the full suite first: bash run_all.sh",
            artifacts_root,
        )
        return

    # Print summary table to console
    display_cols = ["label", "macro_f1_mean", "macro_f1_std", "weighted_f1_mean", "ece_mean"]
    display_cols = [c for c in display_cols if c in df.columns]
    print("\n" + "=" * 70)
    print("ABLATION STUDY — p_final macro F1 summary")
    print("=" * 70)
    print(df[display_cols].to_string(index=False, float_format="%.4f"))
    print("=" * 70 + "\n")

    csv_path, json_path = write_ablation_results(df, out_dir)
    plot_path = write_ablation_plot(df, out_dir)
    check_ablation_integrity(df, out_dir)

    print(f"Outputs:")
    print(f"  CSV:  {csv_path}")
    print(f"  JSON: {json_path}")
    if plot_path:
        print(f"  Plot: {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate ablation results across all experiment runs."
    )
    parser.add_argument(
        "--artifacts-dir", default="artifacts",
        help="Root directory containing per-run artifact subdirectories (default: artifacts)",
    )
    parser.add_argument(
        "--output-dir", default="reports",
        help="Directory to write ablation_results.csv and comparison plot (default: reports)",
    )
    args = parser.parse_args()
    main(args.artifacts_dir, args.output_dir)
