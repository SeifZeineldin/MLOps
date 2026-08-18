from __future__ import annotations

import pytest

from src import api as api_module
from src.pipeline import convert_to_onnx, save_model, train


@pytest.fixture(scope="module")
def client(tmp_path_factory, real_config, real_features_target):
    """Train a small model against the real schema and point the already-imported
    api module at it, so /predict exercises real validation + inference logic
    without touching the project's real artifacts/ directory."""
    X, y = real_features_target
    config = {**real_config, "model": {"type": "random_forest", "params": {"n_estimators": 30, "random_state": 0}}}
    pipeline = train(config, X, y)

    tmp_dir = tmp_path_factory.mktemp("api_artifacts")
    model_path = save_model(pipeline, tmp_dir / "model.joblib")
    onnx_path = convert_to_onnx(pipeline, X, tmp_dir / "model.onnx")

    api_module.CONFIG["artifacts"]["model_path"] = str(model_path)
    api_module.CONFIG["artifacts"]["onnx_path"] = str(onnx_path)
    api_module._state.update({"sklearn_model": None, "onnx_session": None, "onnx_classes": None})

    with api_module.app.test_client() as test_client:
        yield test_client, X


@pytest.fixture
def sample_payload(client):
    _, X = client
    return X.iloc[0].to_dict()


def test_health_endpoint(client):
    test_client, _ = client
    resp = test_client.get("/health")
    assert resp.status_code == 200
    assert resp.get_json() == {"status": "ok"}


def test_features_endpoint_lists_required_fields(client):
    test_client, X = client
    resp = test_client.get("/features")
    assert resp.status_code == 200
    assert resp.get_json()["required_fields"] == list(X.columns)


def test_predict_with_sklearn_engine_returns_valid_grade(client, sample_payload):
    test_client, _ = client
    resp = test_client.post("/predict", json=sample_payload)
    body = resp.get_json()
    assert resp.status_code == 200
    assert body["engine"] == "sklearn"
    assert body["prediction"] in {"A", "B", "C", "D", "F"}
    assert pytest.approx(sum(body["probabilities"].values()), abs=1e-3) == 1.0


def test_predict_with_onnx_engine_returns_valid_grade(client, sample_payload):
    test_client, _ = client
    api_module.CONFIG["api"]["inference_engine"] = "onnx"
    try:
        resp = test_client.post("/predict", json=sample_payload)
    finally:
        api_module.CONFIG["api"]["inference_engine"] = "sklearn"
    body = resp.get_json()
    assert resp.status_code == 200
    assert body["engine"] == "onnx"
    assert body["prediction"] in {"A", "B", "C", "D", "F"}


def test_predict_missing_field_returns_400(client, sample_payload):
    test_client, _ = client
    incomplete = dict(sample_payload)
    incomplete.pop(next(iter(incomplete)))
    resp = test_client.post("/predict", json=incomplete)
    assert resp.status_code == 400
    assert "missing required fields" in resp.get_json()["error"]


def test_predict_non_json_body_returns_400(client):
    test_client, _ = client
    resp = test_client.post("/predict", data="not json", content_type="text/plain")
    assert resp.status_code == 400


def test_predict_non_object_json_returns_400(client):
    test_client, _ = client
    resp = test_client.post("/predict", json=[1, 2, 3])
    assert resp.status_code == 400
    assert "JSON object" in resp.get_json()["error"]
