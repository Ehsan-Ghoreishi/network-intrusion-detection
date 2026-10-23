"""Render the README figures from saved metrics and scores.

Every figure is built from files in ``reports/`` and ``models/``, so plots can
be regenerated without retraining.
"""

from __future__ import annotations

import json
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

from nids.config import CLASSES, MODELS_DIR, REPORTS_DIR  # noqa: E402

log = logging.getLogger(__name__)

# First three slots of a colour-blind-validated categorical palette.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SEQ = ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]

plt.rcParams.update({
    "figure.dpi": 130, "savefig.bbox": "tight", "font.size": 10,
    "axes.titleweight": "bold", "axes.titlesize": 11, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "legend.frameon": False,
})


def _load(name: str) -> dict:
    return json.loads((REPORTS_DIR / name).read_text())


def _save(fig, name: str) -> None:
    fig.savefig(REPORTS_DIR / name)
    plt.close(fig)
    log.info("Wrote reports/%s", name)


def training_curve() -> None:
    h = _load("supervised_history.json")
    fig, ax = plt.subplots(figsize=(6, 3.8))
    epochs = np.arange(1, len(h["loss"]) + 1)
    ax.plot(epochs, h["loss"], color=BLUE, lw=2, label="Train")
    ax.plot(epochs, h["val_loss"], color=ORANGE, lw=2, label="Validation")
    ax.set(xlabel="Epoch", ylabel="Cross-entropy loss", title="Classifier training")
    ax.legend()
    _save(fig, "training_curve.png")


def confusion() -> None:
    cm = np.array(_load("supervised_metrics.json")["test_multiclass"]["confusion_matrix"])
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    im = ax.imshow(cm_norm, cmap=LinearSegmentedColormap.from_list("seq", SEQ), vmin=0, vmax=1)
    ax.grid(False)
    ax.set_xticks(range(len(CLASSES)), CLASSES)
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set(xlabel="Predicted", ylabel="Actual",
           title="Confusion matrix on KDDTest+ (row-normalised)")
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            v = cm_norm[i, j]
            ax.text(j, i, f"{v:.2f}\n({cm[i, j]:,})", ha="center", va="center", fontsize=7.5,
                    color="white" if v > 0.5 else INK)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    _save(fig, "confusion_matrix.png")


def per_class() -> None:
    pc = _load("supervised_metrics.json")["test_multiclass"]["per_class"]
    x, w = np.arange(len(CLASSES)), 0.26
    fig, ax = plt.subplots(figsize=(7, 3.8))
    bars = [("precision", "Precision", BLUE), ("recall", "Recall", ORANGE), ("f1", "F1", AQUA)]
    for i, (met, label, color) in enumerate(bars):
        ax.bar(x + (i - 1) * w, [pc[c][met] for c in CLASSES], w * 0.92, label=label, color=color)
    ax.set_xticks(x, [f"{c}\nn={pc[c]['support']:,}" for c in CLASSES])
    ax.set(ylim=(0, 1.05), ylabel="Score", title="Per-family performance on KDDTest+")
    ax.legend(ncol=3, loc="upper right")
    _save(fig, "per_class.png")


def reconstruction_error() -> None:
    d = np.load(MODELS_DIR / "autoencoder_scores.npz")
    err, y, thr = d["test_err"], d["y_test"], float(d["threshold"])
    # Bin in log10 space so both the tight normal peak and the long attack
    # tail get equal-width bins and comparable densities.
    log_err = np.log10(np.clip(err, 1e-5, None))
    bins = np.linspace(log_err.min(), log_err.max(), 70)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.hist(log_err[y == 0], bins=bins, color=BLUE, alpha=0.65, label="Normal", density=True)
    ax.hist(log_err[y == 1], bins=bins, color=ORANGE, alpha=0.65, label="Attack", density=True)
    ax.axvline(np.log10(thr), color=INK, ls="--", lw=1.4)
    ax.annotate("threshold\n(5% FPR on train normals)", xy=(np.log10(thr), ax.get_ylim()[1] * 0.8),
                xytext=(6, 0), textcoords="offset points", fontsize=8, color=INK)
    ticks = np.arange(np.ceil(bins[0]), np.floor(bins[-1]) + 1)
    ax.set_xticks(ticks, [f"$10^{{{int(t)}}}$" for t in ticks])
    ax.set(xlabel="Reconstruction error (log scale)", ylabel="Density",
           title="Autoencoder anomaly scores on KDDTest+")
    ax.legend(loc="upper right")
    _save(fig, "reconstruction_error.png")


def roc() -> None:
    d = np.load(MODELS_DIR / "autoencoder_scores.npz")
    fpr, tpr, _ = roc_curve(d["y_test"], d["test_err"])
    auc = _load("autoencoder_metrics.json")["test_binary"]["roc_auc"]
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot(fpr, tpr, color=BLUE, lw=2, label=f"Autoencoder (AUC {auc:.3f})")
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls=":", label="Chance")
    ax.set(xlabel="False alarm rate", ylabel="Detection rate", title="ROC: unsupervised detector",
           xlim=(0, 1), ylim=(0, 1.01))
    ax.legend(loc="lower right")
    _save(fig, "roc_autoencoder.png")


def model_comparison() -> None:
    sup, ae = _load("supervised_metrics.json"), _load("autoencoder_metrics.json")
    rows = {"Neural net": sup, "Autoencoder": ae}
    if (REPORTS_DIR / "baseline_metrics.json").exists():
        base = _load("baseline_metrics.json")
        rows["Random forest"] = base["random_forest"]
        rows["Logistic reg."] = base["logistic_regression"]
    names = list(rows)
    series = [
        ("Detection rate", BLUE, [rows[n]["test_binary"]["detection_rate"] for n in names]),
        ("Unseen attack types", ORANGE,
         [rows[n]["test_breakdown"]["unseen_attack_types"] for n in names]),
        ("ROC-AUC", AQUA, [rows[n]["test_binary"]["roc_auc"] for n in names]),
    ]
    x, w = np.arange(len(names)), 0.26
    fig, ax = plt.subplots(figsize=(7.5, 3.9))
    for i, (label, color, vals) in enumerate(series):
        bars = ax.bar(x + (i - 1) * w, vals, w * 0.92, label=label, color=color)
        ax.bar_label(bars, fmt="%.2f", fontsize=7, color=MUTED, padding=2)
    ax.set_xticks(x, names)
    ax.set(ylim=(0, 1.12), ylabel="Score", title="Attack-vs-normal on KDDTest+")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    _save(fig, "model_comparison.png")


def run() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    training_curve()
    confusion()
    per_class()
    reconstruction_error()
    roc()
    model_comparison()
