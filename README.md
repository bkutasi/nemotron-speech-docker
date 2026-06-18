# Nemotron ASR Docker Service

FastAPI wrapper for `onnx-community/nemotron-3.5-asr-streaming-0.6b-onnx-int4` using ONNX Runtime GenAI.

## Run

```bash
docker compose up -d --build
```

Compose runs service/container/image `nemotron-asr:cpu` on `localhost:3003` by default.

HTTPS is enabled by default (`USE_SSL=true`) with a self-signed certificate. This is required for microphone access from LAN browsers. API calls use `curl -k` or `verify=False`.

The first boot downloads the model into the Docker volume. Startup also runs the configured MP3 smoke tests and logs duration, wall time, realtime factor, and a transcript preview.

Health checks:

```bash
curl -k https://localhost:3003/health
curl -k https://localhost:3003/ready
```

To disable HTTPS (HTTP only):

```yaml
environment:
  USE_SSL: "false"
```

## Live Transcription Test

Open `https://localhost:3003/` in a browser. Accept the self-signed certificate warning.

- Live Mic: Click Start, allow microphone access, and speak. Partial transcripts stream in real time.
- File Upload: Choose an audio file and click Transcribe.

From LAN, use `https://<host-ip>:3003/` -- you must accept the certificate warning in each browser.

## Examples

File upload:

```bash
python examples/upload_file.py tests/audio/sample-0.mp3 --language en --url https://localhost:3003/v1/transcriptions
```

Streaming:

```bash
pip install -r examples/requirements.txt
python examples/streaming.py tests/audio/sample-0.mp3 --language en --url wss://localhost:3003/v1/transcriptions/stream
```

## Endpoints

OpenAI-compatible endpoints:

- `POST /v1/audio/transcriptions` accepts multipart audio upload with `file`, `language`, `use_vad`, and `chunk_ms`; returns `{"text": ...}`.
- `GET /v1/models` returns the OpenAI-style model list.
- `GET /status` returns `{"status": "idle"}` or `{"status": "busy"}`.
- `WS /v1/audio/transcriptions/stream` OpenAI-compatible WebSocket streaming alias.

Nemotron-native endpoints:

- `POST /v1/transcriptions` accepts multipart audio upload with `language`, `use_vad`, and `chunk_ms` and returns text plus duration/RTF metrics.
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
USE_SSL
```
