#!/usr/bin/env python3
import argparse
import json
import mimetypes
import ssl
import uuid
from pathlib import Path
from urllib import request


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload one audio file to the ASR endpoint.")
    parser.add_argument("audio", type=Path)
    parser.add_argument("--url", default="http://localhost:3003/v1/transcriptions")
    parser.add_argument("--language", default="auto")
    parser.add_argument("--chunk-ms", type=int, default=560)
    parser.add_argument("--use-vad", action="store_true")
    args = parser.parse_args()

    body, content_type = build_multipart(
        file_path=args.audio,
        fields={
            "language": args.language,
            "chunk_ms": str(args.chunk_ms),
            "use_vad": str(args.use_vad).lower(),
        },
    )
    req = request.Request(args.url, data=body, headers={"Content-Type": content_type}, method="POST")
    ctx = ssl._create_unverified_context() if args.url.startswith("https") else None
    with request.urlopen(req, timeout=300, context=ctx) as response:
        payload = json.loads(response.read().decode("utf-8"))

    print(json.dumps(payload, indent=2, ensure_ascii=False))


def build_multipart(file_path: Path, fields: dict[str, str]) -> tuple[bytes, str]:
    boundary = f"----nemotron-{uuid.uuid4().hex}"
    chunks: list[bytes] = []

    for name, value in fields.items():
        chunks.extend(
            [
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                value.encode(),
                b"\r\n",
            ]
        )

    mime = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    chunks.extend(
        [
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="file"; filename="{file_path.name}"\r\n'.encode(),
            f"Content-Type: {mime}\r\n\r\n".encode(),
            file_path.read_bytes(),
            b"\r\n",
            f"--{boundary}--\r\n".encode(),
        ]
    )
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


if __name__ == "__main__":
    main()
