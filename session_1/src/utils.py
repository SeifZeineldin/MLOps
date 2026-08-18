"""Shared helpers: config loading, data loading, path resolution."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve_path(path: str | Path) -> Path:
    """Resolve a path from config relative to the project root."""
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(config_path: str | Path = "configs/config.yaml") -> dict[str, Any]:
    with open(resolve_path(config_path), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_data(config: dict[str, Any]) -> pd.DataFrame:
    raw_path = resolve_path(config["data"]["raw_path"])
    return pd.read_csv(raw_path)


def get_feature_target(config: dict[str, Any], df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a raw dataframe into (features, target), dropping leak/id columns."""
    target_column = config["data"]["target_column"]
    drop_columns = config["data"].get("drop_columns", [])
    X = df.drop(columns=[target_column, *drop_columns], errors="ignore")
    y = df[target_column]
    return X, y
