#!/bin/sh
# Build a short spoken WAV and an MP3 of the same words, so both formats can be tested.
set -eu
cd "$(dirname "$0")/.."
mkdir -p samples

FFMPEG="${FFMPEG_BINARY:-ffmpeg}"
if ! command -v say >/dev/null 2>&1; then
  echo "macOS say is not available. Pass your own WAV or MP3 to transcribe." >&2
  exit 1
fi
if ! command -v "$FFMPEG" >/dev/null 2>&1; then
  echo "ffmpeg is required to build the mp3 sample." >&2
  exit 1
fi

say -o samples/sample.wav --data-format=LEI16@16000 \
  "This is a short sample for the transcription pipeline. The words should come back as text with timestamps."
"$FFMPEG" -y -hide_banner -loglevel error -i samples/sample.wav -c:a libmp3lame -q:a 4 samples/sample.mp3
echo "wrote samples/sample.wav and samples/sample.mp3"
