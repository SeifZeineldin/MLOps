FROM python:3.10-slim AS base

WORKDIR /app

# System deps for scientific-python wheels that occasionally need compiling
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
COPY data ./data

RUN pip install --no-cache-dir .

# Train once at build time so the image ships with model.joblib + model.onnx baked in
RUN python -m src.pipeline

EXPOSE 5000

ENV FLASK_RUN_HOST=0.0.0.0

CMD ["python", "-m", "src.api"]
