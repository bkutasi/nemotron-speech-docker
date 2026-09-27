#!/bin/sh
# Fetch Parakeet-v3 int8 models on first run if the bind mount hides the
# baked-in image copy (empty ./parakeet-models on host), then start uvicorn.
set -e
MODEL_DIR="${MODEL_DIR:-/app/models/parakeet-v3-int8}"
URL="${PARAKEET_URL:-https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2}"
normalize() {
  if [ -f "$MODEL_DIR/encoder.int8.onnx" ]; then return 0; fi
  found=$(find "$(dirname "$MODEL_DIR")" -maxdepth 2 -name encoder.int8.onnx 2>/dev/null | head -1)
  if [ -n "$found" ]; then
    src=$(dirname "$found")
    if [ "$src" != "$MODEL_DIR" ]; then mv "$src" "$MODEL_DIR"; fi
  fi
}
normalize
if [ ! -f "$MODEL_DIR/encoder.int8.onnx" ]; then
  echo "Parakeet model missing in $MODEL_DIR; fetching..."
  mkdir -p "$(dirname "$MODEL_DIR")"
  curl -L -o /tmp/pk3.tar.bz2 "$URL"
  tar xjf /tmp/pk3.tar.bz2 -C "$(dirname "$MODEL_DIR")"
  rm -f /tmp/pk3.tar.bz2
  normalize
fi
ls "$MODEL_DIR"
exec uvicorn app.main:app --host 0.0.0.0 --port 8000
