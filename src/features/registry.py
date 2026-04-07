"""
Audited flow-feature registry.

Every engineered feature is registered with:
  - name           : Python-safe column name
  - display_name   : Label for paper tables / figures
  - formula_desc   : Human-readable formula string
  - input_cols     : Raw columns consumed (validation hook)
  - category       : Semantic grouping (ratio / asymmetry / tcp / loss / load)
  - clipping       : (low, high) or None — applied AFTER computation
  - compute        : Callable (pd.DataFrame) -> pd.Series

This module is the single source of truth for feature formulas. No formula
should be duplicated in preprocessing scripts or notebooks.

Clipping uses quantile-derived bounds fit on the TRAINING fold only (see
engineer.py). The registry stores the clipping *policy* (e.g., clip_to_01),
not the actual threshold values — those are computed at fit time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np
import pandas as pd

_EPS = 1e-9   # prevents division by zero without shifting scale


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    display_name: str
    formula_desc: str
    input_cols: Tuple[str, ...]
    category: str
    # ('clip_to_01',) or ('percentile', low_q, high_q) or None
    clipping: Optional[Tuple] = None
    compute: Callable[[pd.DataFrame], pd.Series] = field(
        default=None, compare=False, hash=False, repr=False
    )


def _ratio(a: pd.Series, b: pd.Series) -> pd.Series:
    """a / (a + b + ε), result in [0, 1)."""
    return a / (a + b + _EPS)


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    """a / (b + ε)."""
    return a / (b + _EPS)


# ──────────────────────────────────────────────────────────────────────────────
# Feature definitions
# ──────────────────────────────────────────────────────────────────────────────

FEATURE_REGISTRY: List[FeatureSpec] = [
    # ── Byte / packet ratios ──────────────────────────────────────────────────
    FeatureSpec(
        name="byte_ratio",
        display_name="Source Byte Dominance Ratio",
        formula_desc="sbytes / (sbytes + dbytes + ε)",
        input_cols=("sbytes", "dbytes"),
        category="ratio",
        clipping=("clip_to_01",),
        compute=lambda df: _ratio(df["sbytes"].astype(float),
                                  df["dbytes"].astype(float)),
    ),
    FeatureSpec(
        name="pkt_ratio",
        display_name="Source Packet Dominance Ratio",
        formula_desc="spkts / (spkts + dpkts + ε)",
        input_cols=("spkts", "dpkts"),
        category="ratio",
        clipping=("clip_to_01",),
        compute=lambda df: _ratio(df["spkts"].astype(float),
                                  df["dpkts"].astype(float)),
    ),
    FeatureSpec(
        name="load_ratio",
        display_name="Source Load Dominance Ratio",
        formula_desc="sload / (sload + dload + ε)",
        input_cols=("sload", "dload"),
        category="ratio",
        clipping=("clip_to_01",),
        compute=lambda df: _ratio(df["sload"].astype(float),
                                  df["dload"].astype(float)),
    ),
    # ── Asymmetry features ────────────────────────────────────────────────────
    FeatureSpec(
        name="ttl_diff",
        display_name="TTL Asymmetry",
        formula_desc="sttl - dttl",
        input_cols=("sttl", "dttl"),
        category="asymmetry",
        clipping=("percentile", 0.5, 99.5),
        compute=lambda df: (df["sttl"].astype(float) - df["dttl"].astype(float)),
    ),
    FeatureSpec(
        name="inter_pkt_ratio",
        display_name="Source Interpacket Dominance",
        formula_desc="sinpkt / (sinpkt + dinpkt + ε)",
        input_cols=("sinpkt", "dinpkt"),
        category="asymmetry",
        clipping=("clip_to_01",),
        compute=lambda df: _ratio(df["sinpkt"].astype(float),
                                  df["dinpkt"].astype(float)),
    ),
    FeatureSpec(
        name="jit_ratio",
        display_name="Source Jitter Dominance",
        formula_desc="sjit / (sjit + djit + ε)",
        input_cols=("sjit", "djit"),
        category="asymmetry",
        clipping=("clip_to_01",),
        compute=lambda df: _ratio(df["sjit"].astype(float),
                                  df["djit"].astype(float)),
    ),
    # ── Packet-size features ──────────────────────────────────────────────────
    FeatureSpec(
        name="mean_pkt_src",
        display_name="Mean Source Packet Size",
        formula_desc="sbytes / (spkts + ε)",
        input_cols=("sbytes", "spkts"),
        category="packet_size",
        clipping=("percentile", 0.0, 99.5),
        compute=lambda df: _safe_div(df["sbytes"].astype(float),
                                     df["spkts"].astype(float)),
    ),
    FeatureSpec(
        name="mean_pkt_dst",
        display_name="Mean Destination Packet Size",
        formula_desc="dbytes / (dpkts + ε)",
        input_cols=("dbytes", "dpkts"),
        category="packet_size",
        clipping=("percentile", 0.0, 99.5),
        compute=lambda df: _safe_div(df["dbytes"].astype(float),
                                     df["dpkts"].astype(float)),
    ),
    FeatureSpec(
        name="pkt_size_asym",
        display_name="Packet Size Asymmetry",
        formula_desc="mean_pkt_src / (mean_pkt_src + mean_pkt_dst + ε)",
        input_cols=("sbytes", "spkts", "dbytes", "dpkts"),
        category="packet_size",
        clipping=("clip_to_01",),
        compute=lambda df: _ratio(
            _safe_div(df["sbytes"].astype(float), df["spkts"].astype(float)),
            _safe_div(df["dbytes"].astype(float), df["dpkts"].astype(float)),
        ),
    ),
    FeatureSpec(
        name="bytes_per_pkt_total",
        display_name="Total Mean Packet Size",
        formula_desc="(sbytes + dbytes) / (spkts + dpkts + ε)",
        input_cols=("sbytes", "dbytes", "spkts", "dpkts"),
        category="packet_size",
        clipping=("percentile", 0.0, 99.5),
        compute=lambda df: _safe_div(
            df["sbytes"].astype(float) + df["dbytes"].astype(float),
            df["spkts"].astype(float) + df["dpkts"].astype(float),
        ),
    ),
    # ── Loss features ─────────────────────────────────────────────────────────
    FeatureSpec(
        name="loss_rate",
        display_name="Total Packet Loss Rate",
        formula_desc="(sloss + dloss) / (spkts + dpkts + ε)",
        input_cols=("sloss", "dloss", "spkts", "dpkts"),
        category="loss",
        clipping=("clip_to_01",),
        compute=lambda df: _safe_div(
            df["sloss"].astype(float) + df["dloss"].astype(float),
            df["spkts"].astype(float) + df["dpkts"].astype(float),
        ),
    ),
    FeatureSpec(
        name="src_loss_rate",
        display_name="Source Packet Loss Rate",
        formula_desc="sloss / (spkts + ε)",
        input_cols=("sloss", "spkts"),
        category="loss",
        clipping=("clip_to_01",),
        compute=lambda df: _safe_div(df["sloss"].astype(float),
                                     df["spkts"].astype(float)),
    ),
    FeatureSpec(
        name="dst_loss_rate",
        display_name="Destination Packet Loss Rate",
        formula_desc="dloss / (dpkts + ε)",
        input_cols=("dloss", "dpkts"),
        category="loss",
        clipping=("clip_to_01",),
        compute=lambda df: _safe_div(df["dloss"].astype(float),
                                     df["dpkts"].astype(float)),
    ),
    # ── TCP features ──────────────────────────────────────────────────────────
    FeatureSpec(
        name="tcp_setup_ratio",
        display_name="SYN-ACK Proportion of RTT",
        formula_desc="synack / (tcprtt + ε)",
        input_cols=("synack", "tcprtt"),
        category="tcp",
        clipping=("clip_to_01",),
        compute=lambda df: _safe_div(df["synack"].astype(float),
                                     df["tcprtt"].astype(float)),
    ),
    FeatureSpec(
        name="win_ratio",
        display_name="Window Advertisement Asymmetry",
        formula_desc="swin / (swin + dwin + ε)",
        input_cols=("swin", "dwin"),
        category="tcp",
        clipping=("clip_to_01",),
        compute=lambda df: _ratio(df["swin"].astype(float),
                                  df["dwin"].astype(float)),
    ),
    # ── Rate / load features ──────────────────────────────────────────────────
    FeatureSpec(
        name="rate_per_pkt",
        display_name="Flow Rate per Packet",
        formula_desc="rate / (spkts + dpkts + ε)",
        input_cols=("rate", "spkts", "dpkts"),
        category="load",
        clipping=("percentile", 0.0, 99.5),
        compute=lambda df: _safe_div(
            df["rate"].astype(float),
            df["spkts"].astype(float) + df["dpkts"].astype(float),
        ),
    ),
    FeatureSpec(
        name="load_per_pkt",
        display_name="Total Load per Packet",
        formula_desc="(sload + dload) / (spkts + dpkts + ε)",
        input_cols=("sload", "dload", "spkts", "dpkts"),
        category="load",
        clipping=("percentile", 0.0, 99.5),
        compute=lambda df: _safe_div(
            df["sload"].astype(float) + df["dload"].astype(float),
            df["spkts"].astype(float) + df["dpkts"].astype(float),
        ),
    ),
]

# Fast name → spec lookup
REGISTRY_MAP = {spec.name: spec for spec in FEATURE_REGISTRY}

# All engineered feature names in registration order
ENGINEERED_FEATURE_NAMES: List[str] = [spec.name for spec in FEATURE_REGISTRY]
