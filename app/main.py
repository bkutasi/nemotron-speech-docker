import asyncio
import contextlib
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.asr import (
    LANG_TO_ID,
    ModelNotReadyError,
    NemotronASR,
    SUPPORTED_CHUNK_MS,
    TranscriptionResult,
    UnsupportedChunkSizeError,
    UnsupportedLanguageError,
    build_engine,
)
from app.config import AppConfig, load_config

logger = logging.getLogger("uvicorn.error")
settings: AppConfig = load_config()
engine: NemotronASR = build_engine(
    model_id=settings.model_id,
    model_dir=settings.model_dir,
    execution_provider=settings.execution_provider,
    idle_unload_seconds=settings.idle_unload_seconds,
)
load_error: str | None = None
active_streams = 0
active_requests = 0

_STATIC_DIR = Path(__file__).resolve().parent.parent / "examples"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global load_error
    if settings.lazy_load or os.environ.get("NEMOTRON_RECYCLED") == "1":
        logger.info(
            "ASR startup: lazy mode, model stays unloaded until the first request "
            "(idle_unload_seconds=%s model_dir=%s)",
            settings.idle_unload_seconds,
            engine.model_dir,
        )
    else:
        try:
            logger.info("ASR startup: loading model_id=%s provider=%s", engine.model_id, engine.execution_provider)
            await asyncio.to_thread(engine.load)
            logger.info(
                "ASR ready: model_dir=%s sample_rate=%s chunk_samples=%s default_chunk_ms=%s",
                engine.model_dir,
                engine.sample_rate,
                engine.chunk_samples,
                settings.default_chunk_ms,
            )
            if settings.startup_self_test:
                await _run_startup_self_tests(settings)
        except Exception as exc:
            load_error = str(exc)
            logger.exception("ASR startup failed: %s", exc)
    watchdog = asyncio.create_task(_idle_unload_loop())
    try:
        yield
    finally:
        watchdog.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await watchdog

async def _idle_unload_loop(interval_seconds: float = 15.0) -> None:
    while True:
        await asyncio.sleep(interval_seconds)
        try:
            if os.environ.get("NEMOTRON_LISTEN_FD") is not None:
                if (
                    engine.idle_unload_seconds > 0
                    and engine.ready
                    and engine.idle_seconds >= engine.idle_unload_seconds
                    and active_requests == 0
                ):
                    logger.info("Recycling idle ASR process to release inference memory")
                    os.execve(
                        sys.executable,
                        [sys.executable, "-m", "app.server"],
                        {**os.environ, "NEMOTRON_RECYCLED": "1"},
                    )
                continue
            if await asyncio.to_thread(engine.maybe_unload):
                logger.info(
                    "ASR model unloaded after %.0fs idle (load #%s took %.2fs)",
                    engine.idle_unload_seconds,
                    engine.loads,
                    engine.load_seconds or 0.0,
                )
        except Exception as exc:
            logger.warning("Idle unload check failed: %s", exc)


class ActiveRequestsMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        global active_requests
        if scope["type"] not in {"http", "websocket"}:
            await self.app(scope, receive, send)
            return

        active_requests += 1

        async def tracked_receive() -> Message:
            message = await receive()
            # h11 needs the final body event delivered before a CPU-bound handler responds.
            if message["type"] == "http.request" and not message.get("more_body", False):
                await asyncio.sleep(0)
            return message

        try:
            await self.app(scope, tracked_receive if scope["type"] == "http" else receive, send)
        finally:
            if scope["path"] in {
                "/v1/transcriptions",
                "/v1/audio/transcriptions",
                "/v1/transcriptions/stream",
                "/v1/audio/transcriptions/stream",
            }:
                engine.touch()
            active_requests -= 1


app = FastAPI(title="Nemotron 3.5 ASR ONNX INT4", version="0.1.0", lifespan=lifespan)
app.add_middleware(ActiveRequestsMiddleware)


@app.get("/")
async def index():
    return FileResponse(_STATIC_DIR / "asr-test.html", media_type="text/html")


@app.get("/demo")
async def demo():
    return FileResponse(_STATIC_DIR / "win95-demo.html", media_type="text/html")


@app.get("/aero-theme.css")
async def aero_theme():
    return FileResponse(_STATIC_DIR / "aero-theme.css", media_type="text/css")


@app.get("/health")
async def health() -> dict[str, Any]:
    files_present = engine.model_files_present()
    return {
        "status": "healthy" if files_present else "unhealthy",
        "ok": files_present,
        "loaded": engine.ready,
        "default_model": engine.model_id,
        "models": [engine.model_id],
        "speedup": "n/a",
        "lazy_load": settings.lazy_load,
        "idle_unload_seconds": settings.idle_unload_seconds,
        "idle_seconds": round(engine.idle_seconds, 1),
        "loads": engine.loads,
        "load_seconds": round(engine.load_seconds, 3) if engine.load_seconds else None,
    }


@app.get("/status")
async def status() -> dict[str, str]:
    return {"status": "busy" if active_streams else "idle"}


@app.get("/v1/models")
async def models() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {
                "created": 1700000000,
                "id": engine.model_id,
                "object": "model",
                "owned_by": "onnx-community",
            }
        ],
    }


@app.get("/ready")
async def ready() -> JSONResponse:
    if engine.model_files_present():
        return JSONResponse(
            {
                "status": "ready",
                "loaded": engine.ready,
                "model_id": engine.model_id,
                "model_dir": str(engine.model_dir),
                "sample_rate": engine.sample_rate,
                "chunk_samples": engine.chunk_samples,
                "default_chunk_ms": settings.default_chunk_ms,
                "supported_chunk_ms": list(SUPPORTED_CHUNK_MS),
                "execution_provider": engine.execution_provider,
                "lazy_load": settings.lazy_load,
                "idle_unload_seconds": settings.idle_unload_seconds,
            }
        )
    return JSONResponse(
        {"status": "not_ready", "error": load_error or f"model files missing in {engine.model_dir}"},
        status_code=503,
    )


@app.get("/v1/languages")
async def languages() -> dict[str, Any]:
    return {
        "languages": [
            {"code": code, "id": lang_id, "name": name}
            for code, (lang_id, name) in sorted(LANG_TO_ID.items())
        ]
    }


@app.post("/v1/transcriptions")
async def transcriptions(
    file: Annotated[UploadFile, File()],
    language: Annotated[str, Form()] = "auto",
    use_vad: Annotated[bool, Form()] = False,
    chunk_ms: Annotated[int, Form()] = settings.default_chunk_ms,
    vad_threshold: Annotated[float | None, Form()] = None,
    vad_silence_duration_ms: Annotated[int | None, Form()] = None,
    vad_prefix_padding_ms: Annotated[int | None, Form()] = None,
) -> dict[str, Any]:
    result = await _transcribe_uploaded_file(
        file=file, language=language, use_vad=use_vad, chunk_ms=chunk_ms,
        vad_threshold=vad_threshold, vad_silence_duration_ms=vad_silence_duration_ms,
        vad_prefix_padding_ms=vad_prefix_padding_ms,
    )
    return {
        "text": result.text,
        "language": result.language,
        "language_name": result.language_name,
        "duration_seconds": result.duration_seconds,
        "wall_seconds": result.wall_seconds,
        "rtf": result.rtf,
        "sample_rate": result.sample_rate,
        "chunk_ms": result.chunk_ms,
        "chunk_samples": result.chunk_samples,
        "chunks_total": result.chunks_total,
        "chunks_processed": result.chunks_processed,
        "chunks_skipped": result.chunks_skipped,
        "vad_enabled": result.vad_enabled,
    }


@app.post("/v1/audio/transcriptions")
async def openai_transcriptions(
    file: Annotated[UploadFile, File()],
    language: Annotated[str, Form()] = "auto",
    use_vad: Annotated[bool, Form()] = False,
    chunk_ms: Annotated[int, Form()] = settings.default_chunk_ms,
    vad_threshold: Annotated[float | None, Form()] = None,
    vad_silence_duration_ms: Annotated[int | None, Form()] = None,
    vad_prefix_padding_ms: Annotated[int | None, Form()] = None,
) -> dict[str, str]:
    result = await _transcribe_uploaded_file(
        file=file, language=language, use_vad=use_vad, chunk_ms=chunk_ms,
        vad_threshold=vad_threshold, vad_silence_duration_ms=vad_silence_duration_ms,
        vad_prefix_padding_ms=vad_prefix_padding_ms,
    )
    return {"text": result.text}


async def _transcribe_uploaded_file(
    file: UploadFile,
    language: str = "auto",
    use_vad: bool = False,
    chunk_ms: int = settings.default_chunk_ms,
    vad_threshold: float | None = None,
    vad_silence_duration_ms: int | None = None,
    vad_prefix_padding_ms: int | None = None,
) -> TranscriptionResult:
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded audio file is empty")

    req_lang = language or "auto"
    req_chunk_ms = chunk_ms or settings.default_chunk_ms
    logger.info(
        "Transcribe request: filename=%s size=%d language=%s chunk_ms=%d use_vad=%s",
        file.filename, len(content), req_lang, req_chunk_ms, use_vad,
    )

    try:
        result = await engine.transcribe_bytes(
            content=content,
            filename=file.filename or "audio",
            language=language,
            use_vad=use_vad,
            chunk_ms=chunk_ms,
            vad_threshold=vad_threshold,
            vad_silence_duration_ms=vad_silence_duration_ms,
            vad_prefix_padding_ms=vad_prefix_padding_ms,
        )
        if not result.text.strip():
            logger.warning(
                "Transcribe empty result: filename=%s language=%s duration=%.2fs "
                "chunks_total=%d chunks_processed=%d chunks_skipped=%d "
                "chunk_ms=%d rtf=%s",
                file.filename, result.language, result.duration_seconds,
                result.chunks_total, result.chunks_processed, result.chunks_skipped,
                result.chunk_ms,
                f"{result.rtf:.2f}" if result.rtf is not None else "n/a",
            )
        else:
            preview = result.text[:80].replace("\n", " ")
            if len(result.text) > 80:
                preview += "..."
            logger.info(
                "Transcribe ok: filename=%s language=%s duration=%.2fs wall=%.2fs "
                "chunk_ms=%d rtf=%s text=%r",
                file.filename, result.language, result.duration_seconds,
                result.wall_seconds, result.chunk_ms,
                f"{result.rtf:.2f}" if result.rtf is not None else "n/a",
                preview,
            )
        return result
    except UnsupportedLanguageError as exc:
        raise HTTPException(
            status_code=400,
            detail={"message": str(exc), "supported": sorted(LANG_TO_ID)},
        ) from exc
    except ModelNotReadyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except UnsupportedChunkSizeError as exc:
        raise HTTPException(
            status_code=400,
            detail={"message": str(exc), "supported_chunk_ms": list(SUPPORTED_CHUNK_MS)},
        ) from exc
    except Exception as exc:
        logger.exception("Transcribe failed: filename=%s error=%s", file.filename, exc)
        raise HTTPException(status_code=400, detail=f"Could not transcribe audio: {exc}") from exc


@app.websocket("/v1/transcriptions/stream")
@app.websocket("/v1/audio/transcriptions/stream")
async def transcription_stream(websocket: WebSocket) -> None:
    global active_streams

    await websocket.accept()
    stream = None
    final_sent = False
    seg_idx = 0
    language = "auto"
    use_vad = False
    chunk_ms = settings.default_chunk_ms
    vad_threshold: float | None = None
    vad_silence_duration_ms: int | None = None
    vad_prefix_padding_ms: int | None = None

    try:
        # Lazy mode: a stream needs the model resident, so load it now (no-op when loaded).
        await engine.ensure_loaded()
        first = await websocket.receive()
        if "text" in first and first["text"]:
            config = _parse_control(first["text"])
            language = str(config.get("language", language))
            use_vad = bool(config.get("use_vad", use_vad))
            chunk_ms = int(config.get("chunk_ms", chunk_ms))
            vad_threshold = config.get("vad_threshold")
            if vad_threshold is not None:
                vad_threshold = float(vad_threshold)
            vad_silence_duration_ms = config.get("vad_silence_duration_ms")
            if vad_silence_duration_ms is not None:
                vad_silence_duration_ms = int(vad_silence_duration_ms)
            vad_prefix_padding_ms = config.get("vad_prefix_padding_ms")
            if vad_prefix_padding_ms is not None:
                vad_prefix_padding_ms = int(vad_prefix_padding_ms)
            requested_sample_rate = config.get("sample_rate", engine.sample_rate)
            if requested_sample_rate != engine.sample_rate:
                await websocket.send_json(
                    {
                        "event": "error",
                        "message": f"sample_rate must be {engine.sample_rate}; resampling is only supported by the upload endpoint",
                    }
                )
                return
        elif "bytes" in first and first["bytes"] is not None:
            pass
        else:
            await websocket.send_json({"event": "error", "message": "Expected config JSON or binary PCM audio"})
            return

        stream = await engine.create_stream(
            language=language, use_vad=use_vad, chunk_ms=chunk_ms,
            vad_threshold=vad_threshold,
            vad_silence_duration_ms=vad_silence_duration_ms,
            vad_prefix_padding_ms=vad_prefix_padding_ms,
        )
        active_streams += 1
        await websocket.send_json(
            {
                "event": "ready",
                "sample_rate": engine.sample_rate,
                "chunk_ms": stream.chunk_ms,
                "chunk_samples": stream.chunk_samples,
                "supported_chunk_ms": list(SUPPORTED_CHUNK_MS),
                "language": language,
                "language_name": stream.language_name,
                "format": "f32le",
            }
        )

        if "bytes" in first and first["bytes"] is not None:
            partial = stream.accept_pcm_f32le(first["bytes"])
            if partial:
                await websocket.send_json({"event": "partial", "text": partial, "cumulative": stream.text})
                await websocket.send_json({"event": "segment", "index": seg_idx, "text": partial})
                seg_idx += 1

        while True:
            message = await websocket.receive()
            if "bytes" in message and message["bytes"] is not None:
                partial = stream.accept_pcm_f32le(message["bytes"])
                if partial:
                    await websocket.send_json({"event": "partial", "text": partial, "cumulative": stream.text})
                    await websocket.send_json({"event": "segment", "index": seg_idx, "text": partial})
                    seg_idx += 1
            elif "text" in message and message["text"]:
                control = _parse_control(message["text"])
                if control.get("event") == "end":
                    final = stream.finish()
                    final_sent = True
                    await websocket.send_json(final)
                    return
            elif message.get("type") == "websocket.disconnect":
                if not final_sent and stream is not None:
                    try:
                        result = stream.finish()
                        logger.info("WS disconnect finalize chars=%s", len(result.get("text", "")))
                        try:
                            await websocket.send_json(result)
                        except Exception:
                            pass
                    except Exception as exc:
                        logger.warning("WS disconnect finalize failed: %s", exc)
                return
    except WebSocketDisconnect:
        if not final_sent and stream is not None:
            try:
                result = stream.finish()
                text = result.get("text", "")
                logger.info("WS disconnect finalize chars=%s text=%r", len(text), text[:120])
            except Exception as exc:
                logger.warning("WS disconnect finalize failed: %s", exc)
        return
    except UnsupportedLanguageError as exc:
        await websocket.send_json({"event": "error", "message": str(exc), "supported": sorted(LANG_TO_ID)})
    except ModelNotReadyError as exc:
        await websocket.send_json({"event": "error", "message": str(exc)})
    except UnsupportedChunkSizeError as exc:
        await websocket.send_json(
            {"event": "error", "message": str(exc), "supported_chunk_ms": list(SUPPORTED_CHUNK_MS)}
        )
    except Exception as exc:
        await websocket.send_json({"event": "error", "message": str(exc)})
    finally:
        if stream is not None:
            stream.close()
            active_streams = max(active_streams - 1, 0)


def _parse_control(text: str) -> dict[str, Any]:
    import json

    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("Text WebSocket messages must be JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("Text WebSocket messages must be JSON objects")
    return value


async def _run_startup_self_tests(config: AppConfig) -> None:
    if not config.self_test_audio:
        logger.info("ASR self-test: no startup audio configured")
        return

    logger.info("ASR self-test: running %s audio check(s)", len(config.self_test_audio))
    for test in config.self_test_audio:
        if not test.path.exists():
            logger.warning("ASR self-test skipped: language=%s file=%s missing", test.language, test.path)
            continue
        try:
            content = await asyncio.to_thread(test.path.read_bytes)
            result = await engine.transcribe_bytes(
                content=content,
                filename=test.path.name,
                language=test.language,
                use_vad=False,
                chunk_ms=config.default_chunk_ms,
            )
            preview = result.text[:120].replace("\n", " ")
            if len(result.text) > 120:
                preview += "..."
            logger.info(
                "ASR self-test ok: language=%s duration=%.2fs wall=%.2fs rtf=%s text=%r",
                result.language,
                result.duration_seconds,
                result.wall_seconds,
                f"{result.rtf:.2f}" if result.rtf is not None else "n/a",
                preview,
            )
        except Exception as exc:
            logger.exception("ASR self-test failed: language=%s file=%s error=%s", test.language, test.path, exc)
