"""Classical baselines on the same features and split as the neural network.

A deep model is only worth its complexity if it beats strong simple models,
so every result in the README is reported next to these.
"""

from __future__ import annotations

import logging

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from nids.config import CLASSES, REPORTS_DIR, SEED
from nids.data import load_split
from nids.features import build_preprocessor, encode_labels
from nids.metrics import attack_breakdown, binary_metrics, multiclass_metrics
from nids.utils import write_json

log = logging.getLogger(__name__)


def candidates() -> dict:
    return {
        "logistic_regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=SEED),
        "random_forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced_subsample",
            n_jobs=-1, random_state=SEED),
    }


def run() -> dict:
    train, test = load_split("train"), load_split("test")
    y_train, y_test = encode_labels(train), encode_labels(test)
    results = {}
    for name, clf in candidates().items():
        log.info("Fitting %s", name)
        pipe = make_pipeline(build_preprocessor(), clf).fit(train, y_train)
        proba = pipe.predict_proba(test)
        pred = proba.argmax(axis=1)
        pred_bin = (pred != 0).astype(int)
        results[name] = {
            "test_multiclass": multiclass_metrics(y_test, pred, CLASSES),
            "test_binary": binary_metrics(test["is_attack"], pred_bin, 1.0 - proba[:, 0]),
            "test_breakdown": attack_breakdown(test, pred_bin, set(train["label"])),
        }
        log.info("%s test accuracy %.3f", name, results[name]["test_multiclass"]["accuracy"])
    write_json(results, REPORTS_DIR / "baseline_metrics.json")
    return results
