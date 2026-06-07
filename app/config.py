import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

SUPPORTED_CONFIG_CHUNK_MS = (80, 160, 320, 560, 1120)
DEFAULT_CONFIG_CHUNK_MS = 560

@dataclass(frozen=True)
class SelfTestAudio:
    language: str
    path: Path


@dataclass(frozen=True)
class AppConfig:
    model_id: str
    model_dir: Path
    execution_provider: str
    default_chunk_ms: int
    startup_self_test: bool
    self_test_audio: tuple[SelfTestAudio, ...]


def load_config(path: str | Path | None = None) -> AppConfig:
    config_path = Path(path or os.getenv("CONFIG_PATH", "config.yaml"))
    data: dict[str, Any] = {}
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"{config_path} must contain a YAML object")
        data = loaded

    model_id = str(
        os.getenv(
            "MODEL_ID",
            data.get("model_id", "onnx-community/nemotron-3.5-asr-streaming-0.6b-onnx-int4"),
        )
    )
    model_dir = Path(os.getenv("MODEL_DIR", data.get("model_dir", "/models/nemotron")))
    execution_provider = str(os.getenv("EXECUTION_PROVIDER", data.get("execution_provider", "cpu")))
    default_chunk_ms = int(os.getenv("DEFAULT_CHUNK_MS", data.get("default_chunk_ms", DEFAULT_CONFIG_CHUNK_MS)))
    if default_chunk_ms not in SUPPORTED_CONFIG_CHUNK_MS:
        supported = ", ".join(str(value) for value in SUPPORTED_CONFIG_CHUNK_MS)
        raise ValueError(f"default_chunk_ms must be one of: {supported}")
    startup_self_test = _as_bool(os.getenv("STARTUP_SELF_TEST", data.get("startup_self_test", True)))

    raw_tests = data.get("self_test_audio", [])
    if not isinstance(raw_tests, list):
        raise ValueError("self_test_audio must be a list")

    self_test_audio: list[SelfTestAudio] = []
    for item in raw_tests:
        if not isinstance(item, dict):
            raise ValueError("self_test_audio entries must be objects")
        self_test_audio.append(SelfTestAudio(language=str(item["language"]), path=Path(str(item["path"]))))

    return AppConfig(
        model_id=model_id,
        model_dir=model_dir,
        execution_provider=execution_provider,
        default_chunk_ms=default_chunk_ms,
        startup_self_test=startup_self_test,
        self_test_audio=tuple(self_test_audio),
    )


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}
