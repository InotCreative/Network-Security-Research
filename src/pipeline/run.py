"""
Main pipeline entry point.

Usage:
    python -m src.pipeline.run --config configs/multiclass_main.yaml
    python -m src.pipeline.run --config configs/ablations/no_calibration.yaml
    python -m src.pipeline.run --config configs/binary_sanity.yaml

The config YAML drives everything — no hardcoded paths or hyperparameters
appear in this file.
"""

from __future__ import annotations

import argparse
import datetime
import logging
import sys
import uuid
from pathlib import Path

import yaml

from src.pipeline.fold_runner import FoldRunner
from src.utils.logging_utils import setup_logging
from src.utils.manifests import RunManifest
from src.utils.seeds import set_global_seed


def main(config_path: str) -> None:
    config_path = Path(config_path)
    if not config_path.exists():
        print(f"ERROR: config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Seed setup
    global_seed = config.get("global_seed", 42)
    set_global_seed(global_seed)

    # Logging setup
    run_id = config.get("run_id") or f"{datetime.datetime.utcnow():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"
    artifact_dir = Path(config.get("artifact_dir", "artifacts")) / run_id
    reports_dir = Path(config.get("reports_dir", "reports")) / run_id
    config["artifact_dir"] = str(artifact_dir)
    config["reports_dir"] = str(reports_dir)

    log_file = artifact_dir / "run.log"
    setup_logging(level=logging.INFO, log_file=log_file)
    logger = logging.getLogger(__name__)
    logger.info("Run ID: %s", run_id)
    logger.info("Config: %s", config_path)
    logger.info("Global seed: %d", global_seed)

    # Manifest
    manifest = RunManifest(
        run_id=run_id,
        config_path=str(config_path),
        global_seed=global_seed,
    )

    # Run
    runner = FoldRunner(config=config, manifest=manifest)
    runner.run()

    logger.info("Pipeline finished. Artifacts at %s", artifact_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UNSW-NB15 Ensemble IDS pipeline")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()
    main(args.config)
