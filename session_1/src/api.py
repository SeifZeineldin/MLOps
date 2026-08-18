"""Flask API serving grade predictions from the trained model."""
from __future__ import annotations

import logging

import pandas as pd
from flask import Flask, jsonify, request

from src.pipeline import dataframe_to_onnx_inputs, load_model
from src.utils import get_feature_target, load_config, load_data, resolve_path

logger = logging.getLogger(__name__)

CONFIG = load_config()
FEATURE_COLUMNS, _ = get_feature_target(CONFIG, load_data(CONFIG))
FEATURE_COLUMNS = FEATURE_COLUMNS.columns.tolist()

_state: dict = {"sklearn_model": None, "onnx_session": None, "onnx_classes": None}


def _get_sklearn_model():
    if _state["sklearn_model"] is None:
        _state["sklearn_model"] = load_model(CONFIG["artifacts"]["model_path"])
    return _state["sklearn_model"]


def _get_onnx_session():
    if _state["onnx_session"] is None:
        import onnxruntime as rt

        onnx_path = resolve_path(CONFIG["artifacts"]["onnx_path"])
        session = rt.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        _state["onnx_session"] = session
        # skl2onnx (zipmap disabled) emits "probabilities" columns ordered
        # per the label classes reported by the session's own metadata.
        _state["onnx_classes"] = session.get_modelmeta().custom_metadata_map.get("classes")
    return _state["onnx_session"]


def _validate_payload(payload: dict) -> tuple[dict | None, str | None]:
    if not isinstance(payload, dict):
        return None, "request body must be a JSON object"
    missing = [c for c in FEATURE_COLUMNS if c not in payload]
    if missing:
        return None, f"missing required fields: {missing}"
    return {c: payload[c] for c in FEATURE_COLUMNS}, None


def create_app() -> Flask:
    app = Flask(__name__)

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"})

    @app.get("/features")
    def features():
        return jsonify({"required_fields": FEATURE_COLUMNS})

    @app.post("/predict")
    def predict():
        payload = request.get_json(silent=True)
        if payload is None:
            return jsonify({"error": "request body must be JSON"}), 400

        row, error = _validate_payload(payload)
        if error:
            return jsonify({"error": error}), 400

        X = pd.DataFrame([row], columns=FEATURE_COLUMNS)
        engine = CONFIG["api"].get("inference_engine", "sklearn")

        try:
            if engine == "onnx":
                import json

                session = _get_onnx_session()
                feed = dataframe_to_onnx_inputs(X)
                label, probs = session.run(["label", "probabilities"], feed)
                prediction = str(label[0])
                classes = json.loads(_state["onnx_classes"])
                probabilities = {str(c): float(p) for c, p in zip(classes, probs[0])}
            else:
                model = _get_sklearn_model()
                prediction = str(model.predict(X)[0])
                classes = model.named_steps["classifier"].classes_
                proba = model.predict_proba(X)[0]
                probabilities = {str(c): float(p) for c, p in zip(classes, proba)}
        except Exception as exc:  # noqa: BLE001
            logger.exception("prediction failed")
            return jsonify({"error": f"invalid input: {exc}"}), 400

        response = {"prediction": prediction, "engine": engine}
        if probabilities is not None:
            response["probabilities"] = probabilities
        return jsonify(response)

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host=CONFIG["api"]["host"], port=CONFIG["api"]["port"])
