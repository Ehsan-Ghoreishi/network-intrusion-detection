import numpy as np

from nids.config import CLASSES
from nids.features import NUMERIC_COLUMNS, build_preprocessor, encode_labels


def test_output_is_dense_and_finite(frame):
    X = build_preprocessor().fit_transform(frame)
    assert isinstance(X, np.ndarray)
    assert np.isfinite(X).all()


def test_width_matches_categories(frame):
    pre = build_preprocessor().fit(frame)
    n_cat = sum(frame[c].nunique() for c in ["protocol_type", "service", "flag"])
    assert pre.transform(frame).shape[1] == n_cat + 12 + len(NUMERIC_COLUMNS)


def test_constant_column_dropped(frame):
    names = build_preprocessor().fit(frame).get_feature_names_out()
    assert "num_outbound_cmds" not in names
    assert "label" not in names and "difficulty" not in names


def test_unseen_category_is_ignored(frame):
    pre = build_preprocessor().fit(frame)
    row = frame.iloc[[0]].copy()
    row["service"] = "never_seen_service"
    out = pre.transform(row)
    names = pre.get_feature_names_out()
    service_cols = [i for i, n in enumerate(names) if n.startswith("service_")]
    assert out[0, service_cols].sum() == 0


def test_encode_labels_follows_class_order(frame):
    y = encode_labels(frame)
    assert y.min() >= 0 and y.max() < len(CLASSES)
    assert (y[frame["category"].to_numpy() == "normal"] == 0).all()
