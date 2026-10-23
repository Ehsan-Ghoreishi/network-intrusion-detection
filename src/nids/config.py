"""Project-wide paths, the NSL-KDD schema and run settings."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("NIDS_DATA_DIR", ROOT / "data"))
MODELS_DIR = Path(os.environ.get("NIDS_MODELS_DIR", ROOT / "models"))
REPORTS_DIR = Path(os.environ.get("NIDS_REPORTS_DIR", ROOT / "reports"))

SEED = 42

# The 41 connection features, in the order they appear in the raw files.
# See Tavallaee et al., "A Detailed Analysis of the KDD CUP 99 Data Set" (2009).
FEATURE_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
    "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files",
    "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
    "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
]
RAW_COLUMNS = FEATURE_COLUMNS + ["label", "difficulty"]

CATEGORICAL_COLUMNS = ["protocol_type", "service", "flag"]

# Counts and byte volumes span many orders of magnitude; log1p tames them.
LOG_COLUMNS = [
    "duration", "src_bytes", "dst_bytes", "hot", "num_compromised",
    "num_root", "num_file_creations", "num_access_files", "count",
    "srv_count", "dst_host_count", "dst_host_srv_count",
]

# num_outbound_cmds is zero in every row of both splits and carries no signal.
DROP_COLUMNS = ["num_outbound_cmds"]

CLASSES = ["normal", "dos", "probe", "r2l", "u2r"]
