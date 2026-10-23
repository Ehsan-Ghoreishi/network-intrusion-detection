"""Train and evaluate the supervised multi-class intrusion classifier.

Trains on KDDTrain+ and reports on the official KDDTest+ split, whose traffic
distribution differs from training and which contains 17 attack types the
model never sees. That is why test scores sit well below validation scores:
the gap is the honest measure of generalisation, not a bug.
"""

from __future__ import annotations

import logging

import joblib
import keras
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from nids.config import CLASSES, MODELS_DIR, REPORTS_DIR, SEED
from nids.data import load_split
from nids.features import build_preprocessor, encode_labels
from nids.metrics import attack_breakdown, binary_metrics, multiclass_metrics
from nids.models import build_classifier
from nids.utils import set_seed, write_json

log = logging.getLogger(__name__)


def balanced_class_weights(y: np.ndarray) -> dict[int, float]:
    """Inverse-frequency weights so rare families (u2r is 0.04%) still shape the loss.

    Capping the u2r weight (~480x normal) was tried at 3, 10 and 50; the
    differences were smaller than seed-to-seed variance, so it stays uncapped.
    """
    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    return {int(c): float(w) for c, w in zip(classes, weights, strict=True)}


def run(epochs: int = 100, batch_size: int = 256, seed: int = SEED, save: bool = True) -> dict:
    set_seed(seed)
    train, test = load_split("train"), load_split("test")

    fit_df, val_df = train_test_split(
        train, test_size=0.1, stratify=train["category"], random_state=seed
    )
    pre = build_preprocessor().fit(fit_df)
    X_fit, X_val, X_test = (pre.transform(d).astype("float32") for d in (fit_df, val_df, test))
    y_fit, y_val, y_test = (encode_labels(d) for d in (fit_df, val_df, test))
    log.info("Feature matrix: %d columns", X_fit.shape[1])

    model = build_classifier(X_fit.shape[1], len(CLASSES))
    history = model.fit(
        X_fit, y_fit,
        validation_data=(X_val, y_val),
        epochs=epochs, batch_size=batch_size,
        class_weight=balanced_class_weights(y_fit),
        callbacks=[keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True)],
        verbose=2,
    )

    proba = model.predict(X_test, batch_size=2048, verbose=0)
    pred = proba.argmax(axis=1)
    attack_score = 1.0 - proba[:, 0]  # P(not normal)
    pred_bin = (pred != 0).astype(int)

    val_pred = model.predict(X_val, batch_size=2048, verbose=0).argmax(axis=1)
    summary = {
        "model": "supervised_nn",
        "epochs_trained": len(history.history["loss"]),
        "validation": multiclass_metrics(y_val, val_pred, CLASSES)["accuracy"],
        "test_multiclass": multiclass_metrics(y_test, pred, CLASSES),
        "test_binary": binary_metrics(test["is_attack"], pred_bin, attack_score),
        "test_breakdown": attack_breakdown(test, pred_bin, set(train["label"])),
    }
    log.info("Test accuracy %.3f | detection %.3f | false alarms %.3f",
             summary["test_multiclass"]["accuracy"],
             summary["test_binary"]["detection_rate"],
             summary["test_binary"]["false_alarm_rate"])

    if not save:
        return summary
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODELS_DIR / "classifier.keras")
    joblib.dump(pre, MODELS_DIR / "classifier_preprocessor.joblib")
    write_json(summary, REPORTS_DIR / "supervised_metrics.json")
    write_json({k: [float(v) for v in vals] for k, vals in history.history.items()},
               REPORTS_DIR / "supervised_history.json")
    return summary
