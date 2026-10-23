"""Train and evaluate the autoencoder anomaly detector.

The autoencoder sees only normal training traffic and learns to reconstruct
it. Anything it reconstructs poorly is flagged, so it needs no attack labels
and can, in principle, catch attack types nobody has labelled yet.

The alert threshold is chosen on held-out *training* normals to hit a target
false-alarm rate. The test set is never consulted, so the reported numbers
are what you would see when deploying the model.
"""

from __future__ import annotations

import logging

import joblib
import keras
import numpy as np
from sklearn.model_selection import train_test_split

from nids.config import MODELS_DIR, REPORTS_DIR, SEED
from nids.data import load_split
from nids.features import build_preprocessor
from nids.metrics import attack_breakdown, binary_metrics, threshold_at_fpr
from nids.models import build_autoencoder, reconstruction_error
from nids.utils import set_seed, write_json

log = logging.getLogger(__name__)


def run(epochs: int = 100, batch_size: int = 256, target_fpr: float = 0.05,
        seed: int = SEED, save: bool = True) -> dict:
    set_seed(seed)
    train, test = load_split("train"), load_split("test")
    normal = train[train["is_attack"] == 0]
    fit_df, val_df = train_test_split(normal, test_size=0.2, random_state=seed)

    pre = build_preprocessor().fit(fit_df)
    X_fit, X_val, X_test = (pre.transform(d).astype("float32") for d in (fit_df, val_df, test))

    model = build_autoencoder(X_fit.shape[1])
    history = model.fit(
        X_fit, X_fit,
        validation_data=(X_val, X_val),
        epochs=epochs, batch_size=batch_size,
        callbacks=[keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True)],
        verbose=2,
    )

    val_err = reconstruction_error(model, X_val)
    threshold = threshold_at_fpr(val_err, target_fpr)
    test_err = reconstruction_error(model, X_test)
    flagged = (test_err >= threshold).astype(int)
    y_test = test["is_attack"].to_numpy()

    summary = {
        "model": "autoencoder",
        "epochs_trained": len(history.history["loss"]),
        "target_fpr": target_fpr,
        "threshold": round(threshold, 6),
        "test_binary": binary_metrics(y_test, flagged, test_err),
        "test_breakdown": attack_breakdown(test, flagged, set(train["label"])),
        "mean_error": {
            "normal": round(float(test_err[y_test == 0].mean()), 4),
            "attack": round(float(test_err[y_test == 1].mean()), 4),
        },
    }
    log.info("ROC-AUC %.3f | detection %.3f | false alarms %.3f",
             summary["test_binary"]["roc_auc"],
             summary["test_binary"]["detection_rate"],
             summary["test_binary"]["false_alarm_rate"])

    if not save:
        return summary
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODELS_DIR / "autoencoder.keras")
    joblib.dump(pre, MODELS_DIR / "autoencoder_preprocessor.joblib")
    np.savez_compressed(MODELS_DIR / "autoencoder_scores.npz",
                        test_err=test_err, y_test=y_test, threshold=threshold)
    write_json(summary, REPORTS_DIR / "autoencoder_metrics.json")
    return summary
