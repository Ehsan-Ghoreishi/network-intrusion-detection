"""
Supervised deep-learning intrusion detector.

A feed-forward neural network that classifies each network connection as
normal or one of four attack families. This is the "detect known attacks"
half of the project.

WHY A NEURAL NETWORK (and how to explain it in an interview):
A network intrusion is a *non-linear* pattern across many features — no single
threshold on "byte count" or "failed logins" separates attacks from normal
traffic. A neural network learns combinations of features automatically
through its hidden layers, which is why it outperforms a simple rule set.

The network:
  input (12 features)
    -> Dense(64) + ReLU + Dropout      # learns feature combinations
    -> Dense(32) + ReLU + Dropout      # learns higher-level patterns
    -> Dense(5)  + softmax             # probability over the 5 classes

Key techniques used and why:
  - StandardScaler: neural nets train badly on unscaled features.
  - Dropout: randomly switches off neurons during training to prevent
    overfitting (the net can't just memorise the training data).
  - class_weight: u2r attacks are 1% of data; without weighting the net
    would ignore them. Weighting forces it to care about rare attacks.
  - EarlyStopping: stops training when validation loss stops improving,
    so we don't overfit.
"""

from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "network_traffic.csv"
REPORTS = ROOT / "reports"

FEATURES = [
    "duration", "src_bytes", "dst_bytes", "count", "srv_count",
    "dst_host_count", "failed_logins", "logged_in", "same_srv_rate",
    "diff_srv_rate", "serror_rate", "root_shell",
]
CLASSES = ["normal", "dos", "probe", "r2l", "u2r"]

tf.random.set_seed(42)
np.random.seed(42)


def load_data():
    df = pd.read_csv(DATA)
    # log-transform the heavy-tailed byte/count columns (common in net data)
    for col in ["src_bytes", "dst_bytes", "duration", "count", "srv_count",
                "dst_host_count"]:
        df[col] = np.log1p(df[col])
    X = df[FEATURES].values
    y = df["attack_type"].map({c: i for i, c in enumerate(CLASSES)}).values
    return X, y


def build_model(n_features: int, n_classes: int) -> keras.Model:
    model = keras.Sequential([
        layers.Input(shape=(n_features,)),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(n_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    scaler = StandardScaler().fit(X_train)
    X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

    # weight rare attack classes so the network doesn't ignore them
    weights = compute_class_weight("balanced", classes=np.unique(y_train),
                                   y=y_train)
    class_weight = dict(enumerate(weights))

    model = build_model(X_train.shape[1], len(CLASSES))
    model.summary()

    early = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=50, batch_size=256,
        class_weight=class_weight,
        callbacks=[early], verbose=2,
    )

    # ---- evaluation ---------------------------------------------------------
    proba = model.predict(X_test, verbose=0)
    preds = proba.argmax(axis=1)

    print("\nClassification report (per attack family):\n")
    report = classification_report(y_test, preds, target_names=CLASSES,
                                   digits=3, output_dict=True)
    print(classification_report(y_test, preds, target_names=CLASSES, digits=3))

    # binary view: attack vs normal (what a SOC analyst ultimately cares about)
    y_bin = (y_test != 0).astype(int)
    pred_bin = (preds != 0).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_bin, pred_bin).ravel()
    detection_rate = tp / (tp + fn)
    false_alarm = fp / (fp + tn)
    print(f"\n--- Binary (attack vs normal) ---")
    print(f"Attack detection rate: {detection_rate:.1%}")
    print(f"False alarm rate:      {false_alarm:.1%}")

    REPORTS.mkdir(exist_ok=True)
    summary = {
        "overall_accuracy": round(report["accuracy"], 4),
        "attack_detection_rate": round(detection_rate, 4),
        "false_alarm_rate": round(false_alarm, 4),
        "per_class": {c: {"precision": round(report[c]["precision"], 3),
                          "recall": round(report[c]["recall"], 3),
                          "f1": round(report[c]["f1-score"], 3)}
                      for c in CLASSES},
        "epochs_trained": len(history.history["loss"]),
    }
    (REPORTS / "supervised_metrics.json").write_text(json.dumps(summary, indent=2))

    model.save(REPORTS / "intrusion_model.keras")
    np.save(REPORTS / "history.npy", history.history, allow_pickle=True)
    print("\nSaved model + metrics to reports/")
    return summary


if __name__ == "__main__":
    main()
