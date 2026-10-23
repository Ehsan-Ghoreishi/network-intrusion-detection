"""Download, verify and load the NSL-KDD dataset."""

from __future__ import annotations

import hashlib
import logging
import urllib.request
from pathlib import Path

import pandas as pd

from nids.attacks import to_category
from nids.config import DATA_DIR, RAW_COLUMNS

log = logging.getLogger(__name__)

BASE_URL = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/"

# SHA-256 of the canonical files, so a corrupted or tampered mirror is caught.
SPLITS: dict[str, tuple[str, str]] = {
    "train": ("KDDTrain+.txt",
              "1b86d2f957b33082081bba410fe129b475efebcc13c9014c3f447c8271aadf95"),
    "test": ("KDDTest+.txt",
             "fa46b0935342616aa83b7c2578db355b6a7aaabbc492248172c7a1e8b7ab8f84"),
    "test21": ("KDDTest-21.txt",
               "746993ac9e25868827cacf09eab450050a2a1056e1ce48a1ad39f5dc801d531d"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(data_dir: Path | None = None, base_url: str = BASE_URL) -> None:
    """Fetch every split into ``data_dir``, skipping files already verified."""
    data_dir = data_dir or DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)
    for filename, expected in SPLITS.values():
        path = data_dir / filename
        if path.exists() and sha256(path) == expected:
            log.info("%s already present", filename)
            continue
        log.info("Downloading %s", filename)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(base_url + filename, tmp)
        actual = sha256(tmp)
        if actual != expected:
            tmp.unlink()
            raise RuntimeError(
                f"Checksum mismatch for {filename}: expected {expected}, got {actual}"
            )
        tmp.replace(path)


def load_split(split: str, data_dir: Path | None = None) -> pd.DataFrame:
    """Load one split with ``category`` and ``is_attack`` columns added."""
    data_dir = data_dir or DATA_DIR
    filename, _ = SPLITS[split]
    path = data_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `nids download` first")
    df = pd.read_csv(path, header=None, names=RAW_COLUMNS)
    df["category"] = df["label"].map(to_category)
    df["is_attack"] = (df["category"] != "normal").astype(int)
    return df
