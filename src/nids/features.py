"""Turn raw NSL-KDD rows into a dense, scaled feature matrix."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from nids.config import CATEGORICAL_COLUMNS, CLASSES, DROP_COLUMNS, FEATURE_COLUMNS, LOG_COLUMNS

NUMERIC_COLUMNS = [
    c for c in FEATURE_COLUMNS
    if c not in CATEGORICAL_COLUMNS and c not in LOG_COLUMNS and c not in DROP_COLUMNS
]


def build_preprocessor() -> ColumnTransformer:
    """One-hot the categoricals, log-scale the heavy tails, standardise the rest.

    Categories seen only at inference time are encoded as all zeros instead of
    raising, because new services do show up in live traffic.
    """
    return ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             CATEGORICAL_COLUMNS),
            ("log", make_pipeline(FunctionTransformer(np.log1p, feature_names_out="one-to-one"),
                                  StandardScaler()),
             LOG_COLUMNS),
            ("num", StandardScaler(), NUMERIC_COLUMNS),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def encode_labels(df: pd.DataFrame) -> np.ndarray:
    """Integer-encode the attack family in ``CLASSES`` order."""
    return df["category"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
