from __future__ import annotations

import numpy as np
import pytest
import yaml

from src.model import build_model
from src.pipeline import (
    convert_to_onnx,
    dataframe_to_onnx_inputs,
    evaluate,
    load_model,
    onnx_predict,
    run,
    save_model,
    split_data,
    train,
)


@pytest.fixture
def trained_synthetic_pipeline(synthetic_X_y):
    X, y = synthetic_X_y
    pipeline = build_model(X, params={"n_estimators": 20, "random_state": 0})
    pipeline.fit(X, y)
    return pipeline, X, y


def test_split_data_respects_test_size(real_config, synthetic_X_y):
    X, y = synthetic_X_y
    config = {**real_config, "data": {**real_config["data"], "test_size": 0.25}}
    X_train, X_test, y_train, y_test = split_data(config, X, y)
    assert len(X_test) == round(len(X) * 0.25)
    assert len(X_train) + len(X_test) == len(X)


def test_train_returns_fitted_pipeline(real_config, synthetic_X_y):
    X, y = synthetic_X_y
    config = {**real_config, "model": {"type": "random_forest", "params": {"n_estimators": 10, "random_state": 0}}}
    pipeline = train(config, X, y)
    preds = pipeline.predict(X)
    assert len(preds) == len(X)


def test_evaluate_returns_accuracy_between_zero_and_one(trained_synthetic_pipeline):
    pipeline, X, y = trained_synthetic_pipeline
    metrics = evaluate(pipeline, X, y)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["f1_macro"] <= 1.0
    assert "report" in metrics


def test_save_and_load_model_roundtrip(tmp_path, trained_synthetic_pipeline):
    pipeline, X, _ = trained_synthetic_pipeline
    path = save_model(pipeline, tmp_path / "model.joblib")
    assert path.exists()
    reloaded = load_model(path)
    np.testing.assert_array_equal(reloaded.predict(X), pipeline.predict(X))


def test_onnx_export_predictions_match_sklearn(tmp_path, trained_synthetic_pipeline):
    pipeline, X, y = trained_synthetic_pipeline
    onnx_path = convert_to_onnx(pipeline, X, tmp_path / "model.onnx")
    assert onnx_path.exists()

    sklearn_preds = pipeline.predict(X).astype(str)
    onnx_preds = onnx_predict(onnx_path, X).astype(str)
    agreement = np.mean(sklearn_preds == onnx_preds)
    assert agreement == 1.0, f"onnx/sklearn predictions diverged (agreement={agreement})"


def test_dataframe_to_onnx_inputs_shapes(synthetic_X_y):
    X, _ = synthetic_X_y
    feed = dataframe_to_onnx_inputs(X)
    assert set(feed.keys()) == set(X.columns)
    for col, arr in feed.items():
        assert arr.shape == (len(X), 1)


def test_run_end_to_end_produces_artifacts(tmp_path, real_config):
    config = {
        **real_config,
        "model": {"type": "random_forest", "params": {"n_estimators": 30, "random_state": 0}},
        "artifacts": {
            "dir": str(tmp_path),
            "model_path": str(tmp_path / "model.joblib"),
            "onnx_path": str(tmp_path / "model.onnx"),
            "metrics_path": str(tmp_path / "metrics.json"),
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    metrics = run(config_path)

    assert (tmp_path / "model.joblib").exists()
    assert (tmp_path / "model.onnx").exists()
    assert (tmp_path / "metrics.json").exists()
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert metrics["onnx_vs_test_accuracy"] == pytest.approx(metrics["accuracy"], abs=0.05)
