# Nemotron ASR Docker Service

FastAPI wrapper for `onnx-community/nemotron-3.5-asr-streaming-0.6b-onnx-int4` using ONNX Runtime GenAI.

## Run

```bash
docker compose up --build
```

The first boot downloads the model into the Docker volume. Startup also runs the configured MP3 smoke tests and logs duration, wall time, realtime factor, and a transcript preview.

Health checks:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/ready
```

## Examples

File upload:

```bash
python examples/upload_file.py tests/audio/sample-0.mp3 --language en
```

Streaming:

```bash
pip install -r examples/requirements.txt
python examples/streaming.py tests/audio/sample-0.mp3 --language en
```

## Endpoints

- `POST /v1/transcriptions` accepts multipart audio upload with `language`, `use_vad`, and `chunk_ms`.
- `WS /v1/transcriptions/stream` accepts mono little-endian float32 PCM at the model sample rate.
- `GET /v1/languages` lists supported language codes.

## Config

Edit `config.yaml` for the model, provider, default chunk size, and startup self-test files. Environment variables still override model settings:

```text
MODEL_ID
MODEL_DIR
EXECUTION_PROVIDER
DEFAULT_CHUNK_MS
STARTUP_SELF_TEST
CONFIG_PATH
```
