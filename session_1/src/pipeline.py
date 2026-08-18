"""Training pipeline: load data -> train -> evaluate -> save (joblib + ONNX)."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.model import build_model, split_columns
from src.utils import get_feature_target, load_config, load_data, resolve_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def split_data(config: dict[str, Any], X: pd.DataFrame, y: pd.Series):
    return train_test_split(
        X,
        y,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
        stratify=y,
    )


def train(config: dict[str, Any], X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    pipeline = build_model(X_train, config["model"].get("params"))
    pipeline.fit(X_train, y_train)
    return pipeline


def evaluate(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
    preds = pipeline.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, preds),
        "f1_macro": f1_score(y_test, preds, average="macro"),
        "report": classification_report(y_test, preds, output_dict=True),
    }


def save_model(pipeline: Pipeline, path: str | Path) -> Path:
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    return path


def load_model(path: str | Path) -> Pipeline:
    return joblib.load(resolve_path(path))


def _build_initial_types(X: pd.DataFrame):
    """One ONNX graph input per column, typed to match its dtype (skl2onnx's
    convention for pipelines whose ColumnTransformer selects columns by name)."""
    from skl2onnx.common.data_types import FloatTensorType, StringTensorType

    numeric_cols, categorical_cols = split_columns(X)
    initial_types = [(col, FloatTensorType([None, 1])) for col in numeric_cols]
    initial_types += [(col, StringTensorType([None, 1])) for col in categorical_cols]
    return initial_types, numeric_cols, categorical_cols


def convert_to_onnx(pipeline: Pipeline, X: pd.DataFrame, path: str | Path) -> Path:
    from skl2onnx import convert_sklearn

    initial_types, _, _ = _build_initial_types(X)
    onnx_model = convert_sklearn(
        pipeline,
        initial_types=initial_types,
        target_opset=17,
        options={id(pipeline.named_steps["classifier"]): {"zipmap": False}},
    )
    # Record class order so consumers can label the raw "probabilities" output
    # (the ONNX graph itself has no notion of the sklearn classifier's classes_).
    classes = pipeline.named_steps["classifier"].classes_.tolist()
    meta = onnx_model.metadata_props.add()
    meta.key = "classes"
    meta.value = json.dumps(classes)

    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    return path


def dataframe_to_onnx_inputs(X: pd.DataFrame) -> dict[str, np.ndarray]:
    """Build the {column_name: ndarray} feed dict onnxruntime expects, matching
    the per-column inputs produced by `convert_to_onnx`."""
    numeric_cols, categorical_cols = split_columns(X)
    feed: dict[str, np.ndarray] = {}
    for col in numeric_cols:
        feed[col] = X[[col]].to_numpy(dtype=np.float32)
    for col in categorical_cols:
        feed[col] = X[[col]].astype(str).to_numpy(dtype=object)
    return feed


def onnx_predict(onnx_path: str | Path, X: pd.DataFrame) -> np.ndarray:
    import onnxruntime as rt

    session = rt.InferenceSession(str(resolve_path(onnx_path)), providers=["CPUExecutionProvider"])
    feed = dataframe_to_onnx_inputs(X)
    (labels,) = session.run(["label"], feed)
    return labels


def run(config_path: str | Path = "configs/config.yaml") -> dict[str, Any]:
    config = load_config(config_path)
    df = load_data(config)
    X, y = get_feature_target(config, df)
    X_train, X_test, y_train, y_test = split_data(config, X, y)

    logger.info("Training %s on %d rows (%d features)", config["model"]["type"], len(X_train), X.shape[1])
    pipeline = train(config, X_train, y_train)

    metrics = evaluate(pipeline, X_test, y_test)
    logger.info("accuracy=%.4f f1_macro=%.4f", metrics["accuracy"], metrics["f1_macro"])

    model_path = save_model(pipeline, config["artifacts"]["model_path"])
    onnx_path = convert_to_onnx(pipeline, X_train, config["artifacts"]["onnx_path"])

    onnx_preds = onnx_predict(onnx_path, X_test)
    parity = float(np.mean(onnx_preds.astype(str) == y_test.to_numpy().astype(str)))
    metrics["onnx_vs_test_accuracy"] = parity
    logger.info("onnx accuracy on held-out test set=%.4f (sklearn=%.4f)", parity, metrics["accuracy"])

    metrics_path = resolve_path(config["artifacts"]["metrics_path"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=float)

    logger.info("saved sklearn model -> %s", model_path)
    logger.info("saved onnx model -> %s", onnx_path)
    return metrics


if __name__ == "__main__":
    run()
