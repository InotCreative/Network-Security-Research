"""Run manifest: single ledger of every artifact emitted in a pipeline run.

Every emitted file is registered here with its path, purpose, and the config +
seed that produced it. The manifest itself is written to artifacts/run_manifest.yaml
at the end of the run.
"""

from __future__ import annotations

import datetime
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass
class ArtifactEntry:
    name: str
    path: str
    purpose: str
    produced_by: str
    timestamp: str = field(
        default_factory=lambda: datetime.datetime.utcnow().isoformat()
    )
    metadata: Dict[str, Any] = field(default_factory=dict)


class RunManifest:
    """Accumulates artifact registrations for one pipeline run."""

    def __init__(self, run_id: str, config_path: str, global_seed: int) -> None:
        self.run_id = run_id
        self.config_path = config_path
        self.global_seed = global_seed
        self.started_at = datetime.datetime.utcnow().isoformat()
        self._entries: List[ArtifactEntry] = []

    def register(
        self,
        name: str,
        path: Path | str,
        purpose: str,
        produced_by: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        entry = ArtifactEntry(
            name=name,
            path=str(path),
            purpose=purpose,
            produced_by=produced_by,
            metadata=metadata or {},
        )
        self._entries.append(entry)
        logger.debug("Manifest: registered %s → %s", name, path)

    def save(self, out_dir: Path | str) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = out_dir / "run_manifest.yaml"

        data = {
            "run_id": self.run_id,
            "config_path": self.config_path,
            "global_seed": self.global_seed,
            "started_at": self.started_at,
            "completed_at": datetime.datetime.utcnow().isoformat(),
            "artifacts": [
                {
                    "name": e.name,
                    "path": e.path,
                    "purpose": e.purpose,
                    "produced_by": e.produced_by,
                    "timestamp": e.timestamp,
                    "metadata": e.metadata,
                }
                for e in self._entries
            ],
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

        logger.info("Run manifest saved → %s (%d artifacts)", manifest_path, len(self._entries))
        return manifest_path
