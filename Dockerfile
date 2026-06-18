FROM python:3.11-slim AS base

# Image name: nemotron-asr:cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_ID=onnx-community/nemotron-3.5-asr-streaming-0.6b-onnx-int4 \
    MODEL_DIR=/models/nemotron \
    EXECUTION_PROVIDER=cpu \
    HF_HUB_DOWNLOAD_TIMEOUT=60

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libsndfile1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY config.yaml ./config.yaml
COPY tests ./tests
COPY examples ./examples

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
