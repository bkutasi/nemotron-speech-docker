import logging
import os
import subprocess
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

logger = logging.getLogger("uvicorn.error")

MODEL_DIR = Path(os.getenv("MODEL_DIR", "/app/models/parakeet-v3-int8"))
NUM_THREADS = int(os.getenv("OMP_NUM_THREADS", "4"))
SAMPLE_RATE = 16000
ENGINE = "parakeet-v3-int8"
# Below this duration the encoder conv stack gets 0 frames and throws
# ConvInteger {0,128} — Happens on Space-tap / instant-stop empty clips.
MIN_AUDIO_S = 0.25


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
    recognizer = _load_recognizer()
    logger.info("Parakeet ready: model_dir=%s threads=%s", MODEL_DIR, NUM_THREADS)
    yield


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
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
async def ready() -> dict[str, Any]:
    return {"ready": recognizer is not None, "model_dir": str(MODEL_DIR), "engine": ENGINE}


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...), language: str = Form("auto")) -> dict[str, Any]:
    if recognizer is None:
        raise HTTPException(status_code=503, detail="Parakeet model not loaded")
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
        stream = recognizer.create_stream()
        stream.accept_waveform(SAMPLE_RATE, audio)
        recognizer.decode_stream(stream)
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
