"""Generate all figures for the README: training curve, confusion matrix,
per-class performance, and the autoencoder reconstruction-error separation."""

from __future__ import annotations
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow import keras

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
NAVY, TEAL, RED = "#1F3864", "#2E86AB", "#C0392B"
CLASSES = ["normal", "dos", "probe", "r2l", "u2r"]
FEATURES = [
    "duration", "src_bytes", "dst_bytes", "count", "srv_count",
    "dst_host_count", "failed_logins", "logged_in", "same_srv_rate",
    "diff_srv_rate", "serror_rate", "root_shell",
]

plt.rcParams.update({"figure.dpi": 120, "font.size": 10, "axes.titleweight": "bold",
                     "axes.spines.top": False, "axes.spines.right": False})


def training_curve():
    history = np.load(REPORTS / "history.npy", allow_pickle=True).item()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(history["loss"], color=NAVY, lw=2, label="Train loss")
    ax.plot(history["val_loss"], color=RED, lw=2, label="Validation loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Neural Network Training")
    ax.legend()
    fig.tight_layout(); fig.savefig(REPORTS / "training_curve.png"); plt.close(fig)


def confusion():
    df = pd.read_csv(ROOT / "data" / "network_traffic.csv")
    for c in ["src_bytes", "dst_bytes", "duration", "count", "srv_count", "dst_host_count"]:
        df[c] = np.log1p(df[c])
    X = df[FEATURES].values
    y = df["attack_type"].map({c: i for i, c in enumerate(CLASSES)}).values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
    scaler = StandardScaler().fit(X_tr)
    X_te = scaler.transform(X_te)
    model = keras.models.load_model(REPORTS / "intrusion_model.keras")
    preds = model.predict(X_te, verbose=0).argmax(axis=1)
    cm = confusion_matrix(y_te, preds)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)

    cmap = LinearSegmentedColormap.from_list("navy", ["#EAF0F7", NAVY])
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_norm, cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(5), CLASSES); ax.set_yticks(range(5), CLASSES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix (row-normalised)")
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{cm_norm[i,j]:.2f}", ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else NAVY, fontsize=9)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(REPORTS / "confusion_matrix.png"); plt.close(fig)


def per_class():
    m = json.loads((REPORTS / "supervised_metrics.json").read_text())
    pc = m["per_class"]
    metrics = ["precision", "recall", "f1"]
    x = np.arange(len(CLASSES)); w = 0.25
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, met in enumerate(metrics):
        vals = [pc[c][met] for c in CLASSES]
        ax.bar(x + i*w, vals, w, label=met.capitalize(),
               color=[NAVY, TEAL, RED][i])
    ax.set_xticks(x + w, CLASSES); ax.set_ylim(0, 1.05)
    ax.set_title("Per-Attack-Family Performance")
    ax.set_ylabel("Score"); ax.legend()
    fig.tight_layout(); fig.savefig(REPORTS / "per_class.png"); plt.close(fig)


def recon_separation():
    d = np.load(REPORTS / "recon_errors.npz")
    en, ea, thr = d["err_normal"], d["err_attack"], float(d["threshold"])
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.logspace(-2, 3, 60)
    ax.hist(en, bins=bins, alpha=0.7, color=TEAL, label="Normal", density=True)
    ax.hist(ea, bins=bins, alpha=0.7, color=RED, label="Attack", density=True)
    ax.axvline(thr, color="black", ls="--", lw=1.5, label=f"Threshold")
    ax.set_xscale("log")
    ax.set_xlabel("Reconstruction error (log scale)")
    ax.set_ylabel("Density")
    ax.set_title("Autoencoder Separates Attacks by Reconstruction Error")
    ax.legend()
    fig.tight_layout(); fig.savefig(REPORTS / "reconstruction_error.png"); plt.close(fig)


if __name__ == "__main__":
    training_curve(); confusion(); per_class(); recon_separation()
    print("All figures written to reports/")
