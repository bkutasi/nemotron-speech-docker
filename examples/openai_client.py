#!/usr/bin/env python3
"""
Use the OpenAI Python SDK against the Nemotron ASR server.

The /v1/audio/transcriptions endpoint is OpenAI-compatible, so any
client that supports OpenAI's audio transcription API works here.

Install:  pip install openai
Usage:    python examples/openai_client.py tests/audio/sample-0.mp3
"""
import argparse
import json
from pathlib import Path

from openai import OpenAI


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe audio via OpenAI-compatible API.")
    parser.add_argument("audio", type=Path, help="Path to audio file")
    parser.add_argument("--base-url", default="http://localhost:3003/v1", help="API base URL")
    parser.add_argument("--language", default="en", help="Language code (auto, en, hi, ...)")
    args = parser.parse_args()

    # api_key is required by the SDK but the server doesn't check it.
    client = OpenAI(base_url=args.base_url, api_key="not-needed")

    result = client.audio.transcriptions.create(
        model="nemotron",
        file=args.audio,
        language=args.language,
    )

    print(result.text)


if __name__ == "__main__":
    main()
