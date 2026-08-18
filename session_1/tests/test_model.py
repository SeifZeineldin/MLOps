from __future__ import annotations

from sklearn.pipeline import Pipeline

from src.model import build_model, split_columns


def test_split_columns_separates_numeric_and_categorical(synthetic_X_y):
    X, _ = synthetic_X_y
    numeric, categorical = split_columns(X)
    assert set(numeric) == {"study_time_hours", "attendance_percent"}
    assert set(categorical) == {"gender", "internet_access"}


def test_build_model_returns_preprocessor_plus_classifier_pipeline(synthetic_X_y):
    X, _ = synthetic_X_y
    pipeline = build_model(X)
    assert isinstance(pipeline, Pipeline)
    assert list(pipeline.named_steps.keys()) == ["preprocessor", "classifier"]


def test_build_model_fit_predict_roundtrip(synthetic_X_y):
    X, y = synthetic_X_y
    pipeline = build_model(X, params={"n_estimators": 20, "random_state": 0})
    pipeline.fit(X, y)
    preds = pipeline.predict(X)
    assert len(preds) == len(y)
    assert set(preds) <= set(y.unique())


def test_build_model_passes_through_params(synthetic_X_y):
    X, _ = synthetic_X_y
    pipeline = build_model(X, params={"n_estimators": 7, "random_state": 0})
    assert pipeline.named_steps["classifier"].n_estimators == 7
