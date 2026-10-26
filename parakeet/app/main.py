import asyncio
import contextlib
import ctypes
import gc
import logging
import os
import subprocess
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

logger = logging.getLogger("uvicorn.error")

MODEL_DIR = Path(os.getenv("MODEL_DIR", "/app/models/parakeet-v3-int8"))
NUM_THREADS = int(os.getenv("OMP_NUM_THREADS", "4"))
SAMPLE_RATE = 16000
ENGINE = "parakeet-v3-int8"
# Below this duration the encoder conv stack gets 0 frames and throws
# ConvInteger {0,128} — Happens on Space-tap / instant-stop empty clips.
MIN_AUDIO_S = 0.25

try:
    _LIBC: ctypes.CDLL | None = ctypes.CDLL("libc.so.6")
except OSError:  # pragma: no cover - musl and friends
    _LIBC = None

def malloc_trim() -> None:
    """Return freed heap arenas to the OS; without it RSS climbs with every load cycle."""
    if _LIBC is not None and hasattr(_LIBC, "malloc_trim"):
        _LIBC.malloc_trim(0)

recognizer: Any = None

IDLE_UNLOAD_SECONDS = float(os.getenv("IDLE_UNLOAD_SECONDS", "300"))
LAZY_LOAD = os.getenv("LAZY_LOAD", "true").strip().lower() in {"1", "true", "yes", "on"}

_recognizer_lock = threading.Lock()
_last_activity = time.monotonic()
load_seconds: float | None = None
loads = 0

def _touch() -> None:
    global _last_activity
    _last_activity = time.monotonic()

def _idle_seconds() -> float:
    return time.monotonic() - _last_activity

def _model_files_present() -> bool:
    return all(
        (MODEL_DIR / name).exists()
        for name in ("encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt")
    )

def _ensure_loaded() -> Any:
    """Load on demand: only the first request after an unload pays for it."""
    global recognizer, load_seconds, loads
    with _recognizer_lock:
        if recognizer is not None:
            _touch()
            return recognizer
        started = time.perf_counter()
        recognizer = _load_recognizer()
        load_seconds = time.perf_counter() - started
        loads += 1
        _touch()
        logger.info("Parakeet model loaded in %.2fs (load #%s)", load_seconds, loads)
        return recognizer

def _unload() -> bool:
    """Drop the resident model and hand its heap back to the OS."""
    global recognizer
    with _recognizer_lock:
        if recognizer is None:
            return False
        recognizer = None
        gc.collect()
        malloc_trim()
        _touch()
        return True

def _maybe_unload() -> bool:
    if IDLE_UNLOAD_SECONDS <= 0 or recognizer is None:
        return False
    if _idle_seconds() < IDLE_UNLOAD_SECONDS:
        return False
    return _unload()

def _warm_page_cache() -> None:
    """Read the model files once so a reload copies from page cache, not disk."""
    for path in sorted(MODEL_DIR.rglob("*")):
        if path.is_file():
            with path.open("rb") as f:
                while f.read(8 << 20):
                    pass

async def _idle_unload_loop(interval_seconds: float = 15.0) -> None:
    while True:
        await asyncio.sleep(interval_seconds)
        try:
            if await asyncio.to_thread(_maybe_unload):
                logger.info(
                    "Parakeet model unloaded after %.0fs idle (load #%s took %.2fs)",
                    IDLE_UNLOAD_SECONDS,
                    loads,
                    load_seconds or 0.0,
                )
        except Exception as exc:
            logger.warning("Idle unload check failed: %s", exc)


def _load_recognizer() -> Any:
    import sherpa_onnx

    return sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=str(MODEL_DIR / "encoder.int8.onnx"),
        decoder=str(MODEL_DIR / "decoder.int8.onnx"),
        joiner=str(MODEL_DIR / "joiner.int8.onnx"),
        tokens=str(MODEL_DIR / "tokens.txt"),
        model_type="nemo_transducer",
        decoding_method="greedy_search",
        sample_rate=SAMPLE_RATE,
        feature_dim=80,
        num_threads=NUM_THREADS,
        provider="cpu",
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global recognizer
    warm_task: asyncio.Task | None = None
    if LAZY_LOAD:
        logger.info(
            "Parakeet startup: lazy mode, model stays unloaded until the first request "
            "(idle_unload_seconds=%s model_dir=%s)",
            IDLE_UNLOAD_SECONDS,
            MODEL_DIR,
        )
        # Read the weights now so the first load comes from page cache instead of disk.
        warm_task = asyncio.create_task(asyncio.to_thread(_warm_page_cache))
    else:
        try:
            recognizer = _ensure_loaded()
            logger.info("Parakeet ready: model_dir=%s threads=%s", MODEL_DIR, NUM_THREADS)
        except Exception as exc:
            recognizer = None
            logger.exception("Parakeet model load failed: %s", exc)
    watchdog = asyncio.create_task(_idle_unload_loop())
    try:
        yield
    finally:
        watchdog.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await watchdog
        if warm_task is not None and not warm_task.done():
            warm_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await warm_task


app = FastAPI(title="Parakeet-TDT-0.6B-v3 offline ASR", version="0.1.0", lifespan=lifespan)


def _decode_to_mono_16k(content: bytes, filename: str) -> tuple[np.ndarray, float]:
    suffix = Path(filename).suffix or ".audio"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)
    try:
        try:
            import soundfile as sf

            audio, sr = sf.read(str(tmp_path), dtype="float32")
        except Exception:
            audio, sr = _decode_with_ffmpeg(tmp_path)
        if getattr(audio, "ndim", 1) > 1:
            audio = audio.mean(axis=1)
        audio = np.ascontiguousarray(_resample(np.asarray(audio, dtype=np.float32), int(sr)), dtype=np.float32)
    finally:
        tmp_path.unlink(missing_ok=True)
    return audio, len(audio) / SAMPLE_RATE


def _decode_with_ffmpeg(path: Path) -> tuple[np.ndarray, int]:
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "-"],
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0 or not proc.stdout:
        raise RuntimeError(f"ffmpeg decode failed: {proc.stderr.decode()[:200]}")
    return np.frombuffer(proc.stdout, dtype=np.float32).copy(), SAMPLE_RATE


def _resample(audio: np.ndarray, sr: int) -> np.ndarray:
    if sr == SAMPLE_RATE:
        return audio
    import scipy.signal

    n = int(len(audio) * SAMPLE_RATE / sr)
    return scipy.signal.resample(audio, n).astype(np.float32)


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "loaded": recognizer is not None,
        "engine": ENGINE,
        "lazy_load": LAZY_LOAD,
        "idle_unload_seconds": IDLE_UNLOAD_SECONDS,
        "idle_seconds": round(_idle_seconds(), 1),
        "loads": loads,
        "load_seconds": round(load_seconds, 3) if load_seconds else None,
    }


@app.get("/ready")
async def ready() -> JSONResponse:
    payload = {
        "ready": _model_files_present(),
        "loaded": recognizer is not None,
        "model_dir": str(MODEL_DIR),
        "engine": ENGINE,
        "lazy_load": LAZY_LOAD,
        "idle_unload_seconds": IDLE_UNLOAD_SECONDS,
    }
    if payload["ready"]:
        return JSONResponse(payload)
    return JSONResponse(payload, status_code=503)


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...), language: str = Form("auto")) -> dict[str, Any]:
    try:
        rec = await asyncio.to_thread(_ensure_loaded)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Parakeet model load failed: {exc}") from exc
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty audio file")
    try:
        audio, duration_s = _decode_to_mono_16k(content, file.filename or "audio")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not decode audio: {exc}") from exc
    if len(audio) < int(MIN_AUDIO_S * SAMPLE_RATE):
        return {"text": "", "language": language or "auto", "duration_s": duration_s, "rtf": 0.0, "engine": ENGINE}
    try:
        t0 = time.perf_counter()
        stream = rec.create_stream()
        stream.accept_waveform(SAMPLE_RATE, audio)
        rec.decode_stream(stream)
        text = stream.result.text.strip()
        wall = time.perf_counter() - t0
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {exc}") from exc
    return {
        "text": text,
        "language": language or "auto",
        "duration_s": duration_s,
        "rtf": wall / duration_s if duration_s > 0 else None,
        "engine": ENGINE,
    }

@app.post("/v1/audio/transcriptions")
@app.post("/audio/transcriptions")
async def openai_transcriptions(
    file: UploadFile = File(...),
    model: str = Form("whisper-1"),
    response_format: str = Form("json"),
    language: str = Form("auto"),
) -> dict[str, Any]:
    # OpenAI-compatible alias for omp dictation (buffered whole-utterance POST).
    # `model`/`response_format` accepted for compat, ignored (cf. :3003 behavior).
    result = await transcribe(file=file, language=language)
    return {"text": result["text"]}
