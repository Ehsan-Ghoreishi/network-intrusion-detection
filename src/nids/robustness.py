"""Retrain both deep models across seeds and report mean ± std.

A single run on NSL-KDD can swing test accuracy by several points depending
on initialisation, so headline numbers are only meaningful with their spread.
"""

from __future__ import annotations

import logging

import numpy as np

from nids import anomaly, supervised
from nids.config import REPORTS_DIR
from nids.utils import write_json

log = logging.getLogger(__name__)

TRACKED = {
    "supervised_nn": [("test_multiclass", "accuracy"), ("test_multiclass", "macro_f1"),
                      ("test_binary", "detection_rate"), ("test_binary", "false_alarm_rate"),
                      ("test_binary", "roc_auc"), ("test_breakdown", "unseen_attack_types")],
    "autoencoder": [("test_binary", "detection_rate"), ("test_binary", "false_alarm_rate"),
                    ("test_binary", "roc_auc"), ("test_breakdown", "unseen_attack_types")],
}


def summarise(runs: list[dict], keys: list[tuple[str, str]]) -> dict:
    out = {}
    for section, metric in keys:
        vals = np.array([r[section][metric] for r in runs], dtype=float)
        stats = {"mean": vals.mean(), "std": vals.std(ddof=1), "min": vals.min(), "max": vals.max()}
        out[metric] = {k: round(float(v), 4) for k, v in stats.items()}
    return out


def run(n_seeds: int = 5) -> dict:
    seeds = list(range(n_seeds))
    results = {"seeds": seeds}
    for name, module in [("supervised_nn", supervised), ("autoencoder", anomaly)]:
        runs = []
        for seed in seeds:
            log.info("%s seed %d", name, seed)
            runs.append(module.run(seed=seed, save=False))
        results[name] = summarise(runs, TRACKED[name])
    write_json(results, REPORTS_DIR / "seed_variance.json")
    return results
