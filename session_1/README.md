# Student Grade Predictor

Predicts a student's final letter grade (`A`-`F`) from study habits, attendance,
sleep, and background features. Ships as a trained scikit-learn pipeline, an
equivalent ONNX model, a Flask prediction API, a pytest suite, and a Docker image.

## Project structure

```
configs/
  config.yaml         # data/model/artifact/api settings
data/
  raw/                # source CSV (student_performance_dataset.csv)
notebooks/
  01_eda.ipynb         # exploration only - not imported by src/
src/
  __init__.py
  model.py             # preprocessing + classifier definition
  pipeline.py           # train / evaluate / save / ONNX export
  utils.py               # config + data loading helpers
  api.py                   # Flask app
tests/                     # pytest suite
artifacts/                 # generated: model.joblib, model.onnx, metrics.json
Dockerfile
pyproject.toml
```

## The dataset and an important caveat

`data/raw/student_performance_dataset.csv` has 1000 students with `study_time_hours`,
`attendance_percent`, `sleep_hours`, `parental_education`, `internet_access`,
`extracurricular_activities`, `part_time_job`, `previous_grade`, `final_exam_score`
and the target `final_grade`.

**`final_grade` is a deterministic bucket of `final_exam_score`** (A: 90-100,
B: 80-89.9, C: 70-79.9, D: 60-69.9, F: <60). Training on `final_exam_score` gives a
trivially "perfect" model that has learned nothing but the bucketing rule, so both
`final_exam_score` and `student_id` (an identifier) are dropped before training —
see `data.drop_columns` in `configs/config.yaml`. See `notebooks/01_eda.ipynb` for
the full analysis. With that column removed, the real, non-leaky signal from study
habits/attendance/etc. is modest (~60% accuracy, 5-class) — that's an honest result
worth keeping in mind, not a bug to fix.

## Setup

Requires Python 3.10-3.12. This project uses [uv](https://docs.astral.sh/uv/):

```bash
uv venv .venv
uv pip install -p .venv/Scripts/python.exe -e ".[dev]"
```

(or with plain `pip`: `pip install -e ".[dev]"`)

## Train the model

```bash
python -m src.pipeline
```

Reads `configs/config.yaml`, trains a `RandomForestClassifier` on the CSV, evaluates
on a held-out split, and writes `artifacts/model.joblib`, `artifacts/model.onnx` and
`artifacts/metrics.json`.

## Run the API

```bash
python -m src.api
```

Endpoints:
- `GET /health` - liveness check
- `GET /features` - lists the fields `/predict` requires, in order
- `POST /predict` - body is a JSON object with all fields from `/features`

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
        "gender": "Male", "study_time_hours": 4.0, "attendance_percent": 98.0,
        "sleep_hours": 6.5, "parental_education": "Bachelors", "internet_access": "Yes",
        "extracurricular_activities": "Yes", "part_time_job": "No", "previous_grade": 76.9
      }'
```

`configs/config.yaml`'s `api.inference_engine` selects whether `/predict` runs the
scikit-learn pipeline (`sklearn`, default) or the exported ONNX graph via
onnxruntime (`onnx`) - both are validated to agree in `tests/test_pipeline.py` and
`tests/test_api.py`.

## Tests

```bash
pytest
```

Covers config/data loading, the preprocessing+model pipeline, the ONNX
export/parity, and the Flask API (validation, both inference engines).

## Docker

```bash
docker build -t student-grade-predictor .
docker run -p 5000:5000 student-grade-predictor
```

The image trains the model at build time (`RUN python -m src.pipeline`) so it
ships ready to serve; no volume or separate training step is required.

## Notebook

`notebooks/01_eda.ipynb` is exploration only - nothing in `src/` imports it. It
walks through the leakage finding above, class balance, and feature/grade
relationships, and ends with a short list of insights.
