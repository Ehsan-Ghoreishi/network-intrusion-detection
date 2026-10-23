"""Evaluation helpers framed the way a SOC measures a detector."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def binary_metrics(y_true, y_pred, scores=None) -> dict[str, float]:
    """Detection rate, false-alarm rate and friends for attack-vs-normal.

    ``y_true``/``y_pred`` are 1 for attack. ``scores`` (higher = more
    suspicious) adds threshold-free ROC-AUC and PR-AUC.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out = {
        "detection_rate": tp / (tp + fn) if tp + fn else 0.0,
        "false_alarm_rate": fp / (fp + tn) if fp + tn else 0.0,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "accuracy": (tp + tn) / (tp + tn + fp + fn),
    }
    if scores is not None:
        out["roc_auc"] = roc_auc_score(y_true, scores)
        out["pr_auc"] = average_precision_score(y_true, scores)
    return {k: round(float(v), 4) for k, v in out.items()}


def multiclass_metrics(y_true, y_pred, classes: list[str]) -> dict:
    """Accuracy, macro-F1 and per-family precision/recall/F1."""
    labels = list(range(len(classes)))
    report = classification_report(
        y_true, y_pred, labels=labels, target_names=classes,
        output_dict=True, zero_division=0,
    )
    return {
        "accuracy": round(report["accuracy"], 4),
        "macro_f1": round(report["macro avg"]["f1-score"], 4),
        "per_class": {
            c: {
                "precision": round(report[c]["precision"], 4),
                "recall": round(report[c]["recall"], 4),
                "f1": round(report[c]["f1-score"], 4),
                "support": int(report[c]["support"]),
            }
            for c in classes
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def attack_breakdown(df, flagged, seen_labels: set[str]) -> dict:
    """Detection rate per attack family, and for attack types seen vs unseen in training.

    The "unseen" group is the closest NSL-KDD gets to measuring zero-day
    performance: 17 attack types in KDDTest+ never appear in KDDTrain+.
    """
    flagged = np.asarray(flagged).astype(bool)
    attacks = df["is_attack"].to_numpy() == 1
    unseen = ~df["label"].isin(seen_labels).to_numpy() & attacks
    out = {
        "by_family": {
            fam: round(float(flagged[(df["category"] == fam).to_numpy()].mean()), 4)
            for fam in sorted(df.loc[attacks, "category"].unique())
        },
        "seen_attack_types": round(float(flagged[attacks & ~unseen].mean()), 4),
        "unseen_attack_types": round(float(flagged[unseen].mean()), 4) if unseen.any() else None,
        "n_unseen": int(unseen.sum()),
    }
    return out


def threshold_at_fpr(normal_scores: np.ndarray, target_fpr: float) -> float:
    """Score threshold that flags ``target_fpr`` of known-normal traffic."""
    if not 0 < target_fpr < 1:
        raise ValueError("target_fpr must be in (0, 1)")
    return float(np.quantile(normal_scores, 1 - target_fpr))
