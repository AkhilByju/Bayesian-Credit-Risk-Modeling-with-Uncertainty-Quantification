from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Builds a sklearn ColumnTransformer that:
    - imputes missing values
    - scales numeric
    - one-hot encodes categorical
    """
    cat_cols = []
    num_cols = []

    for col in X.columns:
        unique_count = X[col].nunique(dropna=True)
        if col.upper() in {"SEX", "EDUCATION", "MARRIAGE"}:
            cat_cols.append(col)
        elif X[col].dtype == "object":
            cat_cols.append(col)
        elif unique_count <= 20 and np.issubdtype(X[col].dtype, np.integer):
            # keep most integer features numeric, but allow low-cardinality to be categorical
            cat_cols.append(col)
        else:
            num_cols.append(col)

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    return preprocessor
