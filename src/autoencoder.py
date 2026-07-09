"""
Unsupervised anomaly detection with an autoencoder.

This is the "catch UNKNOWN attacks" half of the project — and the more
advanced idea.

THE PROBLEM WITH SUPERVISED DETECTION:
A supervised classifier can only recognise attacks it was trained on. But
real threats include zero-day attacks nobody has labelled yet. How do you
catch an attack you've never seen?

THE AUTOENCODER IDEA (how to explain it in an interview):
An autoencoder is a neural network trained to *reconstruct its own input*.
It has a bottleneck in the middle, so it's forced to learn a compressed
representation of the data. We train it ONLY on normal traffic, so it becomes
very good at reconstructing normal connections — and very bad at
reconstructing anything that doesn't look normal.

At inference time we measure the reconstruction error. Normal traffic -> low
error. Anything anomalous (an attack, even a brand-new one) -> high error.
We flag connections whose error exceeds a threshold.

  input (12) -> Dense(8) -> Dense(4)  [bottleneck]  -> Dense(8) -> output (12)
                \_____ encoder _____/  \_____ decoder _____/

This is exactly how modern anomaly-based IDS and fraud systems work.
"""

from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_recall_curve

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

tf.random.set_seed(42)
np.random.seed(42)


def load_data():
    df = pd.read_csv(DATA)
    for col in ["src_bytes", "dst_bytes", "duration", "count", "srv_count",
                "dst_host_count"]:
        df[col] = np.log1p(df[col])
    return df


def build_autoencoder(n_features: int) -> keras.Model:
    model = keras.Sequential([
        layers.Input(shape=(n_features,)),
        layers.Dense(8, activation="relu"),
        layers.Dense(4, activation="relu"),      # bottleneck: compressed code
        layers.Dense(8, activation="relu"),
        layers.Dense(n_features, activation="linear"),
    ])
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return model


def main():
    df = load_data()

    # split by label: train the autoencoder on NORMAL traffic only
    normal = df[df.is_attack == 0]
    attack = df[df.is_attack == 1]

    X_normal = normal[FEATURES].values
    Xn_train, Xn_test = train_test_split(X_normal, test_size=0.25,
                                         random_state=42)

    scaler = StandardScaler().fit(Xn_train)
    Xn_train = scaler.transform(Xn_train)
    Xn_test = scaler.transform(Xn_test)
    X_attack = scaler.transform(attack[FEATURES].values)

    model = build_autoencoder(len(FEATURES))
    model.summary()

    early = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5,
                                          restore_best_weights=True)
    model.fit(Xn_train, Xn_train,          # target == input: reconstruction
              validation_split=0.2, epochs=50, batch_size=256,
              callbacks=[early], verbose=2)

    # ---- reconstruction error as an anomaly score --------------------------
    def recon_error(X):
        recon = model.predict(X, verbose=0)
        return np.mean((X - recon) ** 2, axis=1)

    err_normal = recon_error(Xn_test)
    err_attack = recon_error(X_attack)

    # threshold: 95th percentile of NORMAL error (only 5% false alarms by design)
    threshold = np.percentile(err_normal, 95)

    # build a combined test set to score properly
    y_true = np.r_[np.zeros(len(err_normal)), np.ones(len(err_attack))]
    scores = np.r_[err_normal, err_attack]
    preds = (scores >= threshold).astype(int)

    roc_auc = roc_auc_score(y_true, scores)
    detection = ((preds == 1) & (y_true == 1)).sum() / (y_true == 1).sum()
    false_alarm = ((preds == 1) & (y_true == 0)).sum() / (y_true == 0).sum()

    print(f"\n--- Autoencoder anomaly detection ---")
    print(f"ROC-AUC:               {roc_auc:.4f}")
    print(f"Anomaly threshold:     {threshold:.4f} (95th pct of normal error)")
    print(f"Attack detection rate: {detection:.1%}")
    print(f"False alarm rate:      {false_alarm:.1%}")
    print(f"\nMean reconstruction error:")
    print(f"  normal traffic: {err_normal.mean():.4f}")
    print(f"  attack traffic: {err_attack.mean():.4f}  "
          f"({err_attack.mean()/err_normal.mean():.1f}x higher)")

    REPORTS.mkdir(exist_ok=True)
    summary = {
        "roc_auc": round(float(roc_auc), 4),
        "threshold": round(float(threshold), 4),
        "attack_detection_rate": round(float(detection), 4),
        "false_alarm_rate": round(float(false_alarm), 4),
        "mean_error_normal": round(float(err_normal.mean()), 4),
        "mean_error_attack": round(float(err_attack.mean()), 4),
    }
    (REPORTS / "autoencoder_metrics.json").write_text(json.dumps(summary, indent=2))

    # save error arrays for the plots
    np.savez(REPORTS / "recon_errors.npz",
             err_normal=err_normal, err_attack=err_attack,
             threshold=threshold)
    print("\nSaved autoencoder metrics to reports/")
    return summary


if __name__ == "__main__":
    main()
