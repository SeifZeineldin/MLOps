"""Model definition: feature preprocessing + classifier, bundled as one sklearn Pipeline."""
from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def split_columns(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Split feature columns into (numeric, categorical) by dtype."""
    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = X.select_dtypes(exclude="number").columns.tolist()
    return numeric_cols, categorical_cols


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols, categorical_cols = split_columns(X)
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )


def build_model(X: pd.DataFrame, params: dict[str, Any] | None = None) -> Pipeline:
    """Build the full inference pipeline: preprocessing + RandomForestClassifier."""
    params = params or {}
    preprocessor = build_preprocessor(X)
    classifier = RandomForestClassifier(**params)
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])
