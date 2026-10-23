"""Small shared helpers: seeding and JSON output."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import keras

from nids.config import SEED


def set_seed(seed: int = SEED) -> None:
    """Seed Python, NumPy and the Keras backend in one call."""
    keras.utils.set_random_seed(seed)


def write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n")


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
