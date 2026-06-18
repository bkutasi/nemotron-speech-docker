#!/usr/bin/env python3
import argparse
import asyncio
import json
import ssl
import subprocess
from pathlib import Path

import websockets


async def main() -> None:
    parser = argparse.ArgumentParser(description="Stream an audio file to the ASR WebSocket endpoint.")
    parser.add_argument("audio", type=Path)
    parser.add_argument("--url", default="ws://localhost:3003/v1/transcriptions/stream")
    parser.add_argument("--language", default="auto")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--chunk-ms", type=int, default=560)
    parser.add_argument("--use-vad", action="store_true")
    parser.add_argument("--vad-threshold", type=float, default=None, help="VAD sensitivity 0.1-0.9")
    parser.add_argument("--vad-silence-ms", type=int, default=None, help="Silence ms before VAD ends speech")
    args = parser.parse_args()

    pcm = decode_f32le(args.audio, args.sample_rate)
    bytes_per_chunk = int(args.sample_rate * args.chunk_ms / 1000) * 4

    ssl_ctx = None
    if args.url.startswith("wss"):
        ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ssl_ctx.check_hostname = False
        ssl_ctx.verify_mode = ssl.CERT_NONE

    config = {
        "language": args.language,
        "sample_rate": args.sample_rate,
        "chunk_ms": args.chunk_ms,
        "use_vad": args.use_vad,
        "format": "f32le",
    }
    if args.vad_threshold is not None:
        config["vad_threshold"] = args.vad_threshold
    if args.vad_silence_ms is not None:
        config["vad_silence_duration_ms"] = args.vad_silence_ms

    async with websockets.connect(args.url, max_size=None, ssl=ssl_ctx) as websocket:
        await websocket.send(json.dumps(config))
        print(await websocket.recv())

        for index in range(0, len(pcm), bytes_per_chunk):
            await websocket.send(pcm[index : index + bytes_per_chunk])
            await drain_available(websocket)

        await websocket.send(json.dumps({"event": "end"}))
        while True:
            message = await websocket.recv()
            print(message)
            event = json.loads(message).get("event")
            if event in {"final", "error"}:
                break


async def drain_available(websocket) -> None:
    while True:
        try:
            print(await asyncio.wait_for(websocket.recv(), timeout=0.01))
        except asyncio.TimeoutError:
            break


def decode_f32le(path: Path, sample_rate: int) -> bytes:
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
        str(sample_rate),
        "-f",
        "f32le",
        "pipe:1",
    ]
    completed = subprocess.run(command, check=True, capture_output=True)
    return completed.stdout


if __name__ == "__main__":
    asyncio.run(main())
