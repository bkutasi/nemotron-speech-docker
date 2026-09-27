FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 curl bzip2 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Model fetch: baked into the image when the builder has net access; the
# entrypoint re-fetches at first runtime start if a bind mount hides it.
# (expects encoder.int8.onnx, decoder.int8.onnx, joiner.int8.onnx, tokens.txt).
ARG PARAKEET_URL=https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2
RUN mkdir -p /app/models && (curl -sSL -o /tmp/pk3.tar.bz2 "$PARAKEET_URL" \
    && tar xjf /tmp/pk3.tar.bz2 -C /app/models && rm -f /tmp/pk3.tar.bz2) || echo "model fetch deferred to entrypoint"

ENV MODEL_DIR=/app/models/parakeet-v3-int8 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY app /app/app
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh
EXPOSE 8000
CMD ["/app/entrypoint.sh"]
