#!/usr/bin/env bash
# Quick reference for all Nemotron ASR API endpoints.
# Usage: bash examples/curl_examples.sh
set -euo pipefail

BASE="${BASE_URL:-http://localhost:3003}"
# For HTTPS (browser/mic): https://localhost:3004
# Use -k to skip self-signed cert verification.

echo "=== Health ==="
curl -s "$BASE/health" | python3 -m json.tool

echo -e "\n=== Status ==="
curl -s "$BASE/status" | python3 -m json.tool

echo -e "\n=== Models (OpenAI-compatible) ==="
curl -s "$BASE/v1/models" | python3 -m json.tool

echo -e "\n=== Languages ==="
curl -s "$BASE/v1/languages" | python3 -m json.tool

echo -e "\n=== Transcribe (full metrics) ==="
curl -s -X POST "$BASE/v1/transcriptions" \
  -F "file=@tests/audio/sample-0.mp3" \
  -F "language=en" \
  -F "chunk_ms=1120" \
  | python3 -m json.tool

echo -e "\n=== Transcribe (OpenAI-compatible, text only) ==="
curl -s -X POST "$BASE/v1/audio/transcriptions" \
  -F "file=@tests/audio/sample-0.mp3" \
  -F "language=en" \
  | python3 -m json.tool

echo -e "\n=== Transcribe with VAD (custom params) ==="
curl -s -X POST "$BASE/v1/transcriptions" \
  -F "file=@tests/audio/sample-0.mp3" \
  -F "language=en" \
  -F "use_vad=true" \
  -F "vad_threshold=0.3" \
  -F "vad_silence_duration_ms=1000" \
  | python3 -m json.tool

echo -e "\n=== Batch transcribe (all mp3s in a folder) ==="
echo 'for f in audio/*.mp3; do'
echo '  echo -n "$f: "'
echo '  curl -s -X POST '"'"'$BASE/v1/audio/transcriptions'"'"' \'
echo '    -F "file=@$f" -F "language=en" \'
echo '    | python3 -c "import json,sys; print(json.load(sys.stdin)[\"text\"][:80])"'
echo 'done'
