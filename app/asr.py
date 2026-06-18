import asyncio
import json
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime_genai as og
from huggingface_hub import snapshot_download


LANG_TO_ID: dict[str, tuple[int, str]] = {
    "en": (0, "English (default / US)"),
    "en-US": (0, "English (United States)"),
    "en-GB": (1, "English (United Kingdom)"),
    "es-ES": (2, "Spanish (Spain)"),
    "es": (3, "Spanish (default / Latin America)"),
    "es-US": (3, "Spanish (US Latin American)"),
    "zh-CN": (4, "Chinese (Mandarin, Simplified)"),
    "hi": (6, "Hindi"),
    "hi-IN": (6, "Hindi (India)"),
    "ar": (7, "Arabic"),
    "ar-AR": (7, "Arabic"),
    "fr": (8, "French (default / France)"),
    "fr-FR": (8, "French (France)"),
    "de": (9, "German"),
    "de-DE": (9, "German (Germany)"),
    "ja": (10, "Japanese"),
    "ja-JP": (10, "Japanese"),
    "ru": (11, "Russian"),
    "ru-RU": (11, "Russian"),
    "pt-BR": (12, "Portuguese (Brazil)"),
    "pt": (13, "Portuguese (default / Portugal)"),
    "pt-PT": (13, "Portuguese (Portugal)"),
    "ko": (14, "Korean"),
    "ko-KR": (14, "Korean (South Korea)"),
    "it": (15, "Italian"),
    "it-IT": (15, "Italian"),
    "nl": (16, "Dutch"),
    "nl-NL": (16, "Dutch (Netherlands)"),
    "pl": (17, "Polish"),
    "pl-PL": (17, "Polish"),
    "tr": (18, "Turkish"),
    "tr-TR": (18, "Turkish"),
    "uk": (19, "Ukrainian"),
    "uk-UA": (19, "Ukrainian"),
    "ro": (20, "Romanian"),
    "ro-RO": (20, "Romanian"),
    "el": (21, "Greek"),
    "el-GR": (21, "Greek"),
    "cs": (22, "Czech"),
    "cs-CZ": (22, "Czech"),
    "hu": (23, "Hungarian"),
    "hu-HU": (23, "Hungarian"),
    "sv": (24, "Swedish"),
    "sv-SE": (24, "Swedish"),
    "da": (25, "Danish"),
    "da-DK": (25, "Danish"),
    "fi": (26, "Finnish"),
    "fi-FI": (26, "Finnish"),
    "sk": (28, "Slovak"),
    "sk-SK": (28, "Slovak"),
    "hr": (29, "Croatian"),
    "hr-HR": (29, "Croatian"),
    "bg": (30, "Bulgarian"),
    "bg-BG": (30, "Bulgarian"),
    "lt": (31, "Lithuanian"),
    "lt-LT": (31, "Lithuanian"),
    "th": (32, "Thai"),
    "th-TH": (32, "Thai"),
    "vi": (33, "Vietnamese"),
    "vi-VN": (33, "Vietnamese"),
    "et": (60, "Estonian"),
    "et-EE": (60, "Estonian"),
    "lv": (61, "Latvian"),
    "lv-LV": (61, "Latvian"),
    "sl": (62, "Slovenian"),
    "sl-SI": (62, "Slovenian"),
    "he": (64, "Hebrew"),
    "he-IL": (64, "Hebrew (Israel)"),
    "fr-CA": (100, "French (Canada)"),
    "auto": (101, "Auto-detect"),
    "mt": (102, "Maltese"),
    "mt-MT": (102, "Maltese"),
    "nb": (103, "Norwegian Bokmal"),
    "nb-NO": (103, "Norwegian Bokmal"),
    "nn": (104, "Norwegian Nynorsk"),
    "nn-NO": (104, "Norwegian Nynorsk"),
}

SUPPORTED_CHUNK_MS = (80, 160, 320, 560, 1120)
DEFAULT_CHUNK_MS = 560

# The Nemotron vocab includes language-tag tokens (e.g. <en-US>, <de-DE>) and
# other special tokens (<blank>, <unk>) that the model may emit during decoding.
# Strip them from the output text so they don't appear in transcripts.
_SPECIAL_TOKEN_RE = re.compile(r"<[a-zA-Z]{2}-[a-zA-Z]{2}>|<blank>|<unk>")


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    language: str
    language_name: str
    duration_seconds: float
    wall_seconds: float
    rtf: float | None
    sample_rate: int
    chunk_ms: int
    chunk_samples: int
    chunks_total: int
    chunks_processed: int
    chunks_skipped: int
    vad_enabled: bool


class ModelNotReadyError(RuntimeError):
    pass


class UnsupportedLanguageError(ValueError):
    pass


class UnsupportedChunkSizeError(ValueError):
    pass


class NemotronASR:
    def __init__(
        self,
        model_id: str,
        model_dir: str | Path,
        execution_provider: str = "cpu",
    ) -> None:
        self.model_id = model_id
        self.model_dir = Path(model_dir)
        self.execution_provider = execution_provider
        self.model: og.Model | None = None
        self.sample_rate: int | None = None
        self.chunk_samples: int | None = None

    @property
    def ready(self) -> bool:
        return self.model is not None and self.sample_rate is not None and self.chunk_samples is not None

    def download_model(self) -> None:
        self.model_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(repo_id=self.model_id, local_dir=str(self.model_dir))

    def load(self) -> None:
        self.download_model()
        self.sample_rate, self.chunk_samples = self._load_audio_config()
        config = self._get_config()
        self.model = og.Model(config)

    async def transcribe_bytes(
        self,
        content: bytes,
        filename: str,
        language: str = "auto",
        use_vad: bool = False,
        chunk_ms: int = DEFAULT_CHUNK_MS,
        vad_threshold: float | None = None,
        vad_silence_duration_ms: int | None = None,
        vad_prefix_padding_ms: int | None = None,
    ) -> TranscriptionResult:
        self._assert_ready()
        audio = await asyncio.to_thread(self._decode_audio_bytes, content, filename)
        assert self.sample_rate is not None
        duration = len(audio) / self.sample_rate
        return await asyncio.to_thread(
            self._transcribe_audio, audio, duration, language, use_vad, chunk_ms,
            vad_threshold, vad_silence_duration_ms, vad_prefix_padding_ms,
        )

    async def create_stream(
        self,
        language: str = "auto",
        use_vad: bool = False,
        chunk_ms: int = DEFAULT_CHUNK_MS,
        vad_threshold: float | None = None,
        vad_silence_duration_ms: int | None = None,
        vad_prefix_padding_ms: int | None = None,
    ) -> "ASRStream":
        self._assert_ready()
        return ASRStream(self, language=language, use_vad=use_vad, chunk_ms=chunk_ms,
                         vad_threshold=vad_threshold,
                         vad_silence_duration_ms=vad_silence_duration_ms,
                         vad_prefix_padding_ms=vad_prefix_padding_ms)

    def release_stream(self) -> None:
        pass

    def _assert_ready(self) -> None:
        if not self.ready:
            raise ModelNotReadyError("ASR model is not loaded")

    def _load_audio_config(self) -> tuple[int, int]:
        config_path = self.model_dir / "genai_config.json"
        with config_path.open("r", encoding="utf-8") as f:
            config = json.load(f)
        return int(config["model"]["sample_rate"]), int(config["model"]["chunk_samples"])

    def _get_config(self) -> og.Config:
        config = og.Config(str(self.model_dir))
        if self.execution_provider != "follow_config":
            config.clear_providers()
            if self.execution_provider != "cpu":
                config.append_provider(self.execution_provider)
        return config

    def _decode_audio_bytes(self, content: bytes, filename: str) -> np.ndarray:
        suffix = Path(filename).suffix or ".audio"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)
        try:
            return self._load_audio_file(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)

    def _load_audio_file(self, path: Path) -> np.ndarray:
        try:
            import soundfile as sf

            audio, sr = sf.read(path, dtype="float32")
        except Exception:
            audio, sr = self._load_audio_with_ffmpeg(path)

        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        assert self.sample_rate is not None
        if sr != self.sample_rate:
            import scipy.signal

            num_samples = int(len(audio) * self.sample_rate / sr)
            audio = scipy.signal.resample(audio, num_samples).astype(np.float32)
        return np.ascontiguousarray(audio, dtype=np.float32)

    def _load_audio_with_ffmpeg(self, path: Path) -> tuple[np.ndarray, int]:
        assert self.sample_rate is not None
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-ac",
            "1",
            "-ar",
            str(self.sample_rate),
            "-f",
            "f32le",
            "pipe:1",
        ]
        completed = subprocess.run(command, check=True, capture_output=True)
        return np.frombuffer(completed.stdout, dtype=np.float32).copy(), self.sample_rate

    def _transcribe_audio(
        self,
        audio: np.ndarray,
        duration: float,
        language: str,
        use_vad: bool,
        chunk_ms: int,
        vad_threshold: float | None = None,
        vad_silence_duration_ms: int | None = None,
        vad_prefix_padding_ms: int | None = None,
    ) -> TranscriptionResult:
        language_id, language_name = self._language(language)
        session = self._new_session(
            language_id=language_id, use_vad=use_vad,
            vad_threshold=vad_threshold,
            vad_silence_duration_ms=vad_silence_duration_ms,
            vad_prefix_padding_ms=vad_prefix_padding_ms,
        )
        chunk_samples = self._chunk_samples_for_ms(chunk_ms)

        stream_start = time.perf_counter()
        text = ""
        chunks_total = 0
        chunks_processed = 0
        chunks_skipped = 0

        for i in range(0, len(audio), chunk_samples):
            chunk = audio[i : i + chunk_samples].astype(np.float32)
            chunks_total += 1
            token_text, processed = session.process_chunk(chunk)
            text += token_text
            if processed:
                chunks_processed += 1
            else:
                chunks_skipped += 1

        text += session.flush()
        wall = time.perf_counter() - stream_start
        return TranscriptionResult(
            text=text.strip(),
            language=language,
            language_name=language_name,
            duration_seconds=duration,
            wall_seconds=wall,
            rtf=duration / wall if wall > 0 else None,
            sample_rate=self.sample_rate or 0,
            chunk_ms=chunk_ms,
            chunk_samples=chunk_samples,
            chunks_total=chunks_total,
            chunks_processed=chunks_processed,
            chunks_skipped=chunks_skipped,
            vad_enabled=session.vad_enabled,
        )

    def _language(self, language: str) -> tuple[int, str]:
        if language not in LANG_TO_ID:
            raise UnsupportedLanguageError(f"Unsupported language: {language}")
        return LANG_TO_ID[language]

    def _chunk_samples_for_ms(self, chunk_ms: int) -> int:
        if chunk_ms not in SUPPORTED_CHUNK_MS:
            supported = ", ".join(str(value) for value in SUPPORTED_CHUNK_MS)
            raise UnsupportedChunkSizeError(f"Unsupported chunk_ms: {chunk_ms}. Supported values: {supported}")
        assert self.sample_rate is not None
        return int(self.sample_rate * chunk_ms / 1000)

    def _new_session(
        self,
        language_id: int,
        use_vad: bool,
        vad_threshold: float | None = None,
        vad_silence_duration_ms: int | None = None,
        vad_prefix_padding_ms: int | None = None,
    ) -> "GenerationSession":
        assert self.model is not None
        processor = og.StreamingProcessor(self.model)
        processor.set_option("use_vad", "false")
        if use_vad:
            try:
                processor.set_option("use_vad", "true")
                if vad_threshold is not None:
                    processor.set_option("vad_threshold", str(vad_threshold))
                if vad_silence_duration_ms is not None:
                    processor.set_option("silence_duration_ms", str(vad_silence_duration_ms))
                if vad_prefix_padding_ms is not None:
                    processor.set_option("prefix_padding_ms", str(vad_prefix_padding_ms))
            except Exception:
                processor.set_option("use_vad", "false")

        tokenizer = og.Tokenizer(self.model)
        tokenizer_stream = tokenizer.create_stream()
        params = og.GeneratorParams(self.model)
        generator = og.Generator(self.model, params)
        generator.set_runtime_option("lang_id", str(int(language_id)))

        return GenerationSession(
            processor=processor,
            tokenizer_stream=tokenizer_stream,
            generator=generator,
            vad_enabled=processor.get_option("use_vad") == "true",
        )


class GenerationSession:
    def __init__(
        self,
        processor: Any,
        tokenizer_stream: Any,
        generator: Any,
        vad_enabled: bool,
    ) -> None:
        self.processor = processor
        self.tokenizer_stream = tokenizer_stream
        self.generator = generator
        self.vad_enabled = vad_enabled

    def process_chunk(self, chunk: np.ndarray) -> tuple[str, bool]:
        inputs = self.processor.process(chunk)
        if inputs is None:
            return "", False
        self.generator.set_inputs(inputs)
        return self._decode_tokens(), True

    def flush(self) -> str:
        inputs = self.processor.flush()
        if inputs is None:
            return ""
        self.generator.set_inputs(inputs)
        return self._decode_tokens()

    def _decode_tokens(self) -> str:
        text = ""
        while not self.generator.is_done():
            self.generator.generate_next_token()
            tokens = self.generator.get_next_tokens()
            if len(tokens) > 0:
                token_text = self.tokenizer_stream.decode(tokens[0])
                if token_text:
                    text += token_text
        return _SPECIAL_TOKEN_RE.sub("", text)


class ASRStream:
    def __init__(
        self,
        engine: NemotronASR,
        language: str,
        use_vad: bool,
        chunk_ms: int,
        vad_threshold: float | None = None,
        vad_silence_duration_ms: int | None = None,
        vad_prefix_padding_ms: int | None = None,
    ) -> None:
        self.engine = engine
        language_id, language_name = engine._language(language)
        self.language = language
        self.language_name = language_name
        self.chunk_ms = chunk_ms
        self.chunk_samples = engine._chunk_samples_for_ms(chunk_ms)
        self.session = engine._new_session(
            language_id=language_id, use_vad=use_vad,
            vad_threshold=vad_threshold,
            vad_silence_duration_ms=vad_silence_duration_ms,
            vad_prefix_padding_ms=vad_prefix_padding_ms,
        )
        self.started_at = time.perf_counter()
        self.samples_received = 0
        self.chunks_total = 0
        self.chunks_processed = 0
        self.chunks_skipped = 0
        self.text = ""
        self.buffer = np.empty(0, dtype=np.float32)
        self.closed = False

    def accept_pcm_f32le(self, payload: bytes) -> str:
        if len(payload) % 4 != 0:
            raise ValueError("Binary audio chunks must be little-endian float32 PCM")
        incoming = np.frombuffer(payload, dtype=np.float32).copy()
        self.samples_received += len(incoming)
        self.buffer = np.concatenate((self.buffer, incoming))
        text = ""
        while len(self.buffer) >= self.chunk_samples:
            chunk = self.buffer[: self.chunk_samples]
            self.buffer = self.buffer[self.chunk_samples :]
            text += self._process_chunk(chunk)
        self.text += text
        return text

    def finish(self) -> dict[str, Any]:
        if len(self.buffer) > 0:
            self.text += self._process_chunk(self.buffer)
            self.buffer = np.empty(0, dtype=np.float32)
        text = self.session.flush()
        self.text += text
        wall = time.perf_counter() - self.started_at
        sample_rate = self.engine.sample_rate or 0
        duration = self.samples_received / sample_rate if sample_rate else 0.0
        self.close()
        return {
            "event": "final",
            "text": self.text.strip(),
            "language": self.language,
            "language_name": self.language_name,
            "duration_seconds": duration,
            "wall_seconds": wall,
            "rtf": duration / wall if wall > 0 else None,
            "sample_rate": sample_rate,
            "chunk_ms": self.chunk_ms,
            "chunk_samples": self.chunk_samples,
            "chunks_total": self.chunks_total,
            "chunks_processed": self.chunks_processed,
            "chunks_skipped": self.chunks_skipped,
            "vad_enabled": self.session.vad_enabled,
        }

    def _process_chunk(self, chunk: np.ndarray) -> str:
        self.chunks_total += 1
        text, processed = self.session.process_chunk(chunk)
        if processed:
            self.chunks_processed += 1
        else:
            self.chunks_skipped += 1
        return text

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            self.engine.release_stream()


def build_engine_from_env() -> NemotronASR:
    return NemotronASR(
        model_id=os.getenv("MODEL_ID", "onnx-community/nemotron-3.5-asr-streaming-0.6b-onnx-int4"),
        model_dir=os.getenv("MODEL_DIR", "/models/nemotron"),
        execution_provider=os.getenv("EXECUTION_PROVIDER", "cpu"),
    )


def build_engine(model_id: str, model_dir: str | Path, execution_provider: str = "cpu") -> NemotronASR:
    return NemotronASR(model_id=model_id, model_dir=model_dir, execution_provider=execution_provider)
