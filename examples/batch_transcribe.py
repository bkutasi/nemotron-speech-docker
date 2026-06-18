#!/usr/bin/env python3
"""
Batch transcribe all audio files in a directory.

Outputs one JSON line per file:
    {"file": "meeting1.mp3", "text": "...", "rtf": 14.2, "duration": 32.5}

Usage:
    python examples/batch_transcribe.py audio/ --language en --chunk-ms 1120
    python examples/batch_transcribe.py audio/ --output results.jsonl
"""
import argparse
import json
import sys
from pathlib import Path
from urllib import request

AUDIO_EXTS = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".webm", ".opus"}


def transcribe(url: str, file_path: Path, language: str, chunk_ms: int) -> dict:
    boundary = "----nemotron-batch"
    body = b""
    for name, value in [("language", language), ("chunk_ms", str(chunk_ms))]:
        body += f"--{boundary}\r\n".encode()
        body += f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode()
        body += value.encode() + b"\r\n"
    mime = "application/octet-stream"
    body += f"--{boundary}\r\n".encode()
    body += f'Content-Disposition: form-data; name="file"; filename="{file_path.name}"\r\n'.encode()
    body += f"Content-Type: {mime}\r\n\r\n".encode()
    body += file_path.read_bytes() + b"\r\n"
    body += f"--{boundary}--\r\n".encode()

    req = request.Request(
        url, data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch transcribe audio files in a directory.")
    parser.add_argument("directory", type=Path, help="Directory containing audio files")
    parser.add_argument("--url", default="http://localhost:3003/v1/transcriptions")
    parser.add_argument("--language", default="auto")
    parser.add_argument("--chunk-ms", type=int, default=1120, help="Chunk size in ms (1120 is fastest for batch)")
    parser.add_argument("--output", type=Path, default=None, help="Write JSONL to file instead of stdout")
    args = parser.parse_args()

    if not args.directory.is_dir():
        print(f"Error: {args.directory} is not a directory", file=sys.stderr)
        sys.exit(1)

    files = sorted(
        f for f in args.directory.iterdir()
        if f.suffix.lower() in AUDIO_EXTS
    )
    if not files:
        print(f"No audio files found in {args.directory}", file=sys.stderr)
        sys.exit(1)

    out = open(args.output, "w") if args.output else sys.stdout
    try:
        for f in files:
            try:
                result = transcribe(args.url, f, args.language, args.chunk_ms)
                line = json.dumps({
                    "file": f.name,
                    "text": result["text"],
                    "rtf": result.get("rtf"),
                    "duration": result.get("duration_seconds"),
                }, ensure_ascii=False)
                print(line, file=out, flush=True)
            except Exception as exc:
                print(json.dumps({"file": f.name, "error": str(exc)}, ensure_ascii=False), file=out, flush=True)
    finally:
        if args.output:
            out.close()
            print(f"Results written to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
