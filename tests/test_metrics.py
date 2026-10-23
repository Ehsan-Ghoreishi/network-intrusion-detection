import numpy as np
import pytest

from nids.metrics import binary_metrics, multiclass_metrics, threshold_at_fpr


def test_binary_metrics_hand_computed():
    y_true = [0, 0, 0, 0, 1, 1, 1, 1]
    y_pred = [0, 0, 0, 1, 1, 1, 1, 0]
    m = binary_metrics(y_true, y_pred)
    assert m["detection_rate"] == 0.75
    assert m["false_alarm_rate"] == 0.25
    assert m["precision"] == 0.75
    assert m["accuracy"] == 0.75


def test_binary_metrics_with_scores():
    m = binary_metrics([0, 0, 1, 1], [0, 0, 1, 1], scores=[0.1, 0.2, 0.8, 0.9])
    assert m["roc_auc"] == 1.0 and m["pr_auc"] == 1.0


def test_multiclass_includes_absent_classes():
    m = multiclass_metrics([0, 1, 1], [0, 1, 0], ["normal", "dos", "u2r"])
    assert m["per_class"]["u2r"]["support"] == 0
    assert np.array(m["confusion_matrix"]).shape == (3, 3)


def test_threshold_at_fpr():
    normal = np.arange(100, dtype=float)
    thr = threshold_at_fpr(normal, 0.05)
    assert np.mean(normal >= thr) == pytest.approx(0.05, abs=0.01)


@pytest.mark.parametrize("bad", [0, 1, -0.1])
def test_threshold_rejects_bad_fpr(bad):
    with pytest.raises(ValueError):
        threshold_at_fpr(np.ones(10), bad)
