"""
Single source of truth for UNSW-NB15 schema: columns, dtypes, label vocabulary,
and the explicit separation of feature columns from label columns.

The official train/test splits do NOT include srcip, dstip, sport, dsport, or
timestamps — those appeared only in the raw pcap-derived files. We therefore
operate entirely on the 42-column official splits.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List

# ──────────────────────────────────────────────────────────────────────────────
# Label vocabularies
# ──────────────────────────────────────────────────────────────────────────────

MULTICLASS_LABELS: List[str] = [
    "Normal",
    "Analysis",
    "Backdoor",
    "DoS",
    "Exploits",
    "Fuzzers",
    "Generic",
    "Reconnaissance",
    "Shellcode",
    "Worms",
]

BINARY_LABELS: List[int] = [0, 1]   # 0 = Normal, 1 = Attack

# ──────────────────────────────────────────────────────────────────────────────
# Column groups
# ──────────────────────────────────────────────────────────────────────────────

# Columns present in the official CSV that must be dropped before modelling
DROP_COLS: List[str] = ["id"]

# Label columns — never permitted as model input features
LABEL_COLS: List[str] = ["attack_cat", "label"]

# Categorical (nominal) feature columns and their handling notes
CATEGORICAL_COLS: List[str] = ["proto", "service", "state"]

# Binary indicator columns (already 0/1/2 integers; treat as numeric)
BINARY_INDICATOR_COLS: List[str] = ["is_ftp_login", "ct_ftp_cmd", "is_sm_ips_ports"]

# All numeric feature columns (excludes categoricals, drop cols, label cols)
NUMERIC_COLS: List[str] = [
    "dur",
    "spkts", "dpkts",
    "sbytes", "dbytes",
    "rate",
    "sttl", "dttl",
    "sload", "dload",
    "sloss", "dloss",
    "sinpkt", "dinpkt",
    "sjit", "djit",
    "swin", "stcpb", "dtcpb", "dwin",
    "tcprtt", "synack", "ackdat",
    "smean", "dmean",
    "trans_depth", "response_body_len",
    "ct_srv_src", "ct_state_ttl", "ct_dst_ltm",
    "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
    "is_ftp_login", "ct_ftp_cmd", "ct_flw_http_mthd",
    "ct_src_ltm", "ct_srv_dst",
    "is_sm_ips_ports",
]

# Canonical feature column order (all input features, before engineering)
FEATURE_COLS: List[str] = NUMERIC_COLS + CATEGORICAL_COLS

# Expected full column set in the raw CSV (using BOM-stripped names)
EXPECTED_COLS: List[str] = (
    DROP_COLS + NUMERIC_COLS + CATEGORICAL_COLS + LABEL_COLS
)

# Dtype mapping for loading
DTYPE_MAP: Dict[str, str] = {
    "id": "Int64",
    "dur": "float64",
    "proto": "category",
    "service": "category",
    "state": "category",
    "spkts": "Int64", "dpkts": "Int64",
    "sbytes": "Int64", "dbytes": "Int64",
    "rate": "float64",
    "sttl": "Int64", "dttl": "Int64",
    "sload": "float64", "dload": "float64",
    "sloss": "Int64", "dloss": "Int64",
    "sinpkt": "float64", "dinpkt": "float64",
    "sjit": "float64", "djit": "float64",
    "swin": "Int64", "stcpb": "Int64", "dtcpb": "Int64", "dwin": "Int64",
    "tcprtt": "float64", "synack": "float64", "ackdat": "float64",
    "smean": "Int64", "dmean": "Int64",
    "trans_depth": "Int64", "response_body_len": "Int64",
    "ct_srv_src": "Int64", "ct_state_ttl": "Int64", "ct_dst_ltm": "Int64",
    "ct_src_dport_ltm": "Int64", "ct_dst_sport_ltm": "Int64",
    "ct_dst_src_ltm": "Int64",
    "is_ftp_login": "Int64",
    "ct_ftp_cmd": "Int64",
    "ct_flw_http_mthd": "Int64",
    "ct_src_ltm": "Int64",
    "ct_srv_dst": "Int64",
    "is_sm_ips_ports": "Int64",
    "attack_cat": "category",
    "label": "Int64",
}


@dataclass
class SchemaSpec:
    """Immutable contract for a loaded UNSW-NB15 dataframe."""
    drop_cols: List[str] = field(default_factory=lambda: DROP_COLS)
    label_cols: List[str] = field(default_factory=lambda: LABEL_COLS)
    numeric_cols: List[str] = field(default_factory=lambda: NUMERIC_COLS)
    categorical_cols: List[str] = field(default_factory=lambda: CATEGORICAL_COLS)
    feature_cols: List[str] = field(default_factory=lambda: FEATURE_COLS)
    multiclass_labels: List[str] = field(default_factory=lambda: MULTICLASS_LABELS)
    binary_labels: List[int] = field(default_factory=lambda: BINARY_LABELS)

    def validate(self, df) -> None:
        """Raise ValueError with context if df violates the schema."""
        import pandas as pd

        missing = set(FEATURE_COLS + LABEL_COLS) - set(df.columns)
        if missing:
            raise ValueError(
                f"Schema violation — missing columns: {sorted(missing)}"
            )

        mc_col = "attack_cat"
        if mc_col in df.columns:
            observed = set(df[mc_col].dropna().astype(str).str.strip().unique())
            unexpected = observed - set(MULTICLASS_LABELS)
            if unexpected:
                raise ValueError(
                    f"Schema violation — unexpected attack_cat values: {unexpected}"
                )

        bin_col = "label"
        if bin_col in df.columns:
            observed_bin = set(df[bin_col].dropna().astype(int).unique())
            unexpected_bin = observed_bin - {0, 1}
            if unexpected_bin:
                raise ValueError(
                    f"Schema violation — unexpected label values: {unexpected_bin}"
                )


SCHEMA = SchemaSpec()
