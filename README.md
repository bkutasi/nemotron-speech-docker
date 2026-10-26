# Nemotron ASR Docker Service

FastAPI wrapper for `onnx-community/nemotron-3.5-asr-streaming-0.6b-onnx-int4` using ONNX Runtime GenAI.

## Run

```bash
docker compose up -d --build
```

Three ports are exposed:
- **HTTP 3003** -- API access (curl, scripts, file upload)
- **HTTPS 3004** -- Browser UI with microphone (self-signed cert)
- **HTTP 3005** -- Parakeet-v3 offline batch sidecar (`POST /transcribe`, plus `POST /audio/transcriptions` and `POST /v1/audio/transcriptions` aliases)

The first boot downloads the models into the host bind mounts (`./nemotron-models:/models`, `./parakeet-models:/app/models`). With `LAZY_LOAD=false` startup also loads the model and runs the configured MP3 smoke tests, logging duration, wall time, realtime factor, and a transcript preview; the default lazy mode skips that and loads on first use.

Health checks (compose gates both services on `/ready`, which reports the weights being on disk, not resident in RAM):

```bash
curl http://localhost:3003/health
curl -k https://localhost:3004/health
curl http://localhost:3003/ready
curl http://localhost:3005/ready
```

## Model residency (lazy load + idle unload)

Neither service keeps its weights in RAM when nothing is happening. `docker compose up` leaves both models on disk only; the first request loads them, and `IDLE_UNLOAD_SECONDS` (default 300 = 5 min) of no traffic drops them again. Predictions from `/transcriptions`, `/transcribe` and the websocket stream all load the model themselves, so a cold request just takes the load on the chin.

Reload speed comes from keeping the files hot rather than resident: startup reads the whole model directory once, so the weights live in the page cache and a reload is a memory copy instead of a disk read. Weights are never re-downloaded, the snapshot is only fetched when the bind mount is empty.

Unloading is not just dropping the Python reference. ONNX Runtime keeps freed buffers in its own arena, so the process held on to roughly 900 MiB after the first unload and more on every cycle. `unload()` therefore calls `malloc_trim(0)` after `gc.collect()`, which hands that heap back to the kernel. Measured on this host (CPU, int4, `IDLE_UNLOAD_SECONDS=20`, three load/unload cycles):

| Service | engine load | cold request | reload request | RSS loaded | RSS after unload | fresh start |
|---|---|---|---|---|---|---|
| nemotron-asr | 0.38s | 1.51s | 1.27s | 682 MiB | 166 MiB | 122 MiB |
| parakeet-asr | 0.75s | 1.27s | 0.95s | ~1.0 GiB | 101 MiB | 49 MiB |

Those idle figures were flat across all three cycles, so repeated load/unload does not creep upward. A stream held open for 24s with a 20s idle window stayed loaded throughout: nothing unloads under a stream or an upload.

`/health` carries the residency detail (`loaded`, `loads`, `load_seconds`, `idle_seconds`). Nothing unloads while a stream or upload is in flight.

Disable either half without editing the compose file:

```bash
LAZY_LOAD=false docker compose up -d         # load at startup, never unload
IDLE_UNLOAD_SECONDS=0 docker compose up -d   # once loaded, stay loaded
IDLE_UNLOAD_SECONDS=30 docker compose up -d  # aggressive unload, handy for testing
```

## Live Transcription Test

Open `https://localhost:3004/` in a browser. Accept the self-signed certificate warning.

- Live Mic: Click Start, allow microphone access, and speak. Partial transcripts accumulate in real time. An audio level bar shows mic input.
- File Upload: Choose an audio file and click Transcribe.

For LAN access, open `https://<host-ip>:3004/` from a local browser (not X11 forwarded -- mic must be on the machine running the browser).

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

OpenAI SDK compatible:

```bash
pip install openai
python examples/openai_client.py tests/audio/sample-0.mp3 --language en
```

Batch transcribe a directory:

```bash
python examples/batch_transcribe.py tests/audio/ --language en --chunk-ms 1120 --output results.jsonl
```

Curl reference:

```bash
bash examples/curl_examples.sh
```

With VAD and custom parameters:

```bash
# File upload with VAD
python examples/upload_file.py meeting.mp3 --language en --use-vad --vad-threshold 0.2 --vad-silence-ms 500

# Streaming with VAD
python examples/streaming.py meeting.mp3 --use-vad --vad-threshold 0.2 --vad-silence-ms 500

# Curl with VAD
curl -X POST http://localhost:3003/v1/transcriptions \
  -F "file=@meeting.mp3" \
  -F "language=en" \
  -F "use_vad=true" \
  -F "vad_threshold=0.3" \
  -F "vad_silence_duration_ms=1000"
```

### VAD Parameters

When `use_vad=true`, three parameters are configurable:

| Parameter | Default | Range | Description |
|---|---|---|---|
| `vad_threshold` | 0.3 | 0.1–0.9 | Speech detection sensitivity (lower = more sensitive) |
| `vad_silence_duration_ms` | 3360 | 100–10000 | Silence duration before speech segment ends |
| `vad_prefix_padding_ms` | 560 | — | Audio kept before speech starts as context |

For catching short utterances ("hmm", "okay"), use a smaller `vad_silence_duration_ms` (500–1000ms). For continuous transcription with long pauses, keep the default 3360ms.

### Chunk Size

Smaller chunks (80–160ms) reduce latency and catch short words better. Larger chunks (1120ms) give ~2x faster batch processing. Available: 80, 160, 320, 560, 1120.

**Short utterances ("hello"):** The INT4 model is optimized for 560ms chunks (the server default). Clips need roughly **560ms of continuous speech** — not silence-padded to 560ms. A 280ms "hello" often returns empty (`proc=0`). Speak slightly longer (~0.6–1.0s), or pass `chunk_ms=160`/`80` per request for lower latency. Batch uploads auto-retry alternate chunk sizes on empty results. Partial final chunks are zero-padded before inference.

### Parakeet-v3 batch sidecar (port 3005)

Offline accuracy path for noisy mic dictation; Nemotron WS stays the streaming path.

```bash
curl -X POST http://localhost:3005/transcribe -F file=@tests/audio/sample-0.mp3 -F language=en
# -> {"text": ..., "language": "en", "duration_s": 10.04, "rtf": 0.02, "engine": "parakeet-v3-int8"}
```

For local, tmux-scoped omp dictation on the host running this sidecar, the mic tunnel must already reach `tcp:localhost:4713`. Run from a **separate** tmux pane:

```bash
cd vad-dictate
./vad_omp.sh --list   # identify the omp pane, not this shell
./vad_omp.sh %2       # replace %2 with the live omp pane ID
```

The daemon inserts transcribed text without pressing Enter; Ctrl-C stops it. It exits when the pinned pane's omp process dies and never retargets. `touch ~/.vad-paused` pauses insertion; `rm ~/.vad-paused` resumes. `vad-dictate/*.log` holds transcribed speech and is gitignored. Pause the daemon before F1 recording or text interleaves.

If the default tunneled source is wrong, use `PULSE_SERVER=tcp:localhost:4713 pactl list sources short` to find the mic name, then override it:

```bash
VAD_DEVICE="<mic-source-name>" ./vad_omp.sh %2
```

## Endpoints

OpenAI-compatible endpoints:

- `POST /v1/audio/transcriptions` accepts multipart audio upload with `file`, `language`, `use_vad`, `chunk_ms`, `vad_threshold`, `vad_silence_duration_ms`, and `vad_prefix_padding_ms`; returns `{"text": ...}`.
- `GET /v1/models` returns the OpenAI-style model list.
- `GET /status` returns `{"status": "idle"}` or `{"status": "busy"}`.
- `GET /ready` returns `200 {"status":"ready","loaded":true/false,...}` while the weights are on disk and `503` when they are missing; compose healthcheck probes this.
- `GET /health` returns residency state (`loaded`, `loads`, `load_seconds`, `idle_seconds`, `idle_unload_seconds`).
- `WS /v1/audio/transcriptions/stream` alias for the custom streaming protocol below (JSON config + mono f32le PCM + `{"event":"end"}`), not an OpenAI protocol.

Nemotron-native endpoints:

- `POST /v1/transcriptions` accepts multipart audio upload with `language`, `use_vad`, `chunk_ms`, `vad_threshold`, `vad_silence_duration_ms`, and `vad_prefix_padding_ms`; returns text plus duration/RTF metrics.
- `WS /v1/transcriptions/stream` accepts JSON config, then mono little-endian float32 PCM at the model sample rate; client ends with `{"event":"end"}` and receives `partial`/`final` events.
- `GET /v1/languages` lists supported language codes.

Parakeet sidecar (port 3005):

- `GET /ready` returns `{"ready": true/false, "loaded": true/false, ...}`, 503 only when the model files are missing; compose healthcheck probes this.
- `GET /health` returns residency state (`loaded`, `loads`, `load_seconds`, `idle_seconds`).
- `POST /audio/transcriptions` and `POST /v1/audio/transcriptions` are OpenAI-compatible aliases for `POST /transcribe` (buffered whole-utterance POST).

## Config

Edit `config.yaml` for the model, provider, default chunk size, startup self-test files, and residency (`lazy_load`, `idle_unload_seconds`). Environment variables still override model settings:

```text
MODEL_ID
MODEL_DIR
EXECUTION_PROVIDER
DEFAULT_CHUNK_MS
STARTUP_SELF_TEST
CONFIG_PATH

# model residency (both services)
LAZY_LOAD              # default true: start with the model unloaded
IDLE_UNLOAD_SECONDS    # default 300: idle time before the model leaves RAM
```
