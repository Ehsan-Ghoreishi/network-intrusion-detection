"""Shared fixtures: a tiny NSL-KDD-shaped dataset, so tests never hit the network."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nids.config import FEATURE_COLUMNS, RAW_COLUMNS

LABELS = ["normal", "neptune", "smurf", "ipsweep", "guess_passwd", "rootkit"]


def make_raw(n: int = 60, seed: int = 0, labels: list[str] = LABELS) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.integers(0, 1000, n) for c in FEATURE_COLUMNS})
    df["protocol_type"] = rng.choice(["tcp", "udp", "icmp"], n)
    df["service"] = rng.choice(["http", "ftp", "smtp", "private"], n)
    df["flag"] = rng.choice(["SF", "S0", "REJ"], n)
    df["label"] = [labels[i % len(labels)] for i in range(n)]
    df["difficulty"] = rng.integers(0, 22, n)
    return df[RAW_COLUMNS]


@pytest.fixture
def raw_dir(tmp_path):
    """A data directory holding train/test files in the raw headerless format."""
    make_raw(60, seed=0).to_csv(tmp_path / "KDDTrain+.txt", header=False, index=False)
    make_raw(30, seed=1).to_csv(tmp_path / "KDDTest+.txt", header=False, index=False)
    return tmp_path


@pytest.fixture
def frame(raw_dir):
    from nids.data import load_split
    return load_split("train", raw_dir)

