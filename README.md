# Transcription pipeline

A small service and a command-line script that accept an audio file, transcribe the speech, and return the text with a start and end time on every segment.

The speech engine is [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (Whisper running locally through CTranslate2). No model is trained. ffmpeg is the decoder, so the model only ever sees 16 kHz mono PCM.

## Try it

Open the page and drop in a recording, or press **Use the spoken sample**.

```bash
transcribe-api
```

Then open http://127.0.0.1:8000. When this process is running, the page calls the Python service. The public demo runs Whisper in the browser instead, so a reviewer can try it with no install. The Python package remains the production path: ffmpeg, long-file checkpoints, and the HTTP API.

Live demo: https://transcription-pipeline.vercel.app

Source: https://github.com/usmankicsit/transcription-pipeline

## What you need

- Python 3.11 or 3.12 (3.14 does not yet have a CTranslate2 wheel)
- ffmpeg and ffprobe on `PATH` (`brew install ffmpeg` on macOS, `sudo apt install ffmpeg` on Debian or Ubuntu)

## Install

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
pip install -e .
```

The first real transcription downloads the Whisper weights into the local Hugging Face cache. The default model is `base`. `tiny` is faster for a first check. `small`, `medium`, and `large-v3` are the same command with `--model`.

## How to test

Automated tests cover format handling, chunk planning, timestamp merging, the CLI, and the HTTP API. They use a deterministic stand-in engine, so they do not download model weights.

```bash
pytest -q
```

Run the pipeline on a real file:

```bash
# macOS can synthesize a spoken sample in both WAV and MP3
bash scripts/make_sample.sh

transcribe samples/sample.wav --model tiny
transcribe samples/sample.mp3 --model tiny -f srt -o sample.srt
transcribe samples/sample.mp3 --model tiny -f txt
```

A real run is already checked in. `samples/sample.wav` and `samples/sample.mp3` are the same spoken sentence. `examples/sample.transcript.json` and `examples/sample.srt` are what `tiny` produced from them.

JSON is the canonical result. It looks like this:

```json
{
  "language": "en",
  "duration": 6.2,
  "text": "This is a short sample for the transcription pipeline.",
  "segments": [
    {
      "start": 0.0,
      "end": 4.8,
      "text": "This is a short sample for the transcription pipeline.",
      "words": [
        {"start": 0.0, "end": 0.3, "text": "This"}
      ]
    }
  ],
  "source_format": "mp3",
  "source_codec": "mp3"
}
```

`--engine mock` runs the same path without model weights. It is what the tests use. It returns timed placeholder text so the decode, chunk, and merge steps can be checked on their own.

Start the service:

```bash
transcribe-api
```

Then open http://127.0.0.1:8000 and upload a file, or call it directly:

```bash
curl -s -F "file=@samples/sample.mp3" -F "language=en" http://127.0.0.1:8000/v1/transcriptions
curl -s -F "file=@samples/sample.wav" -F "response_format=srt" http://127.0.0.1:8000/v1/transcriptions
```

A file longer than 15 minutes, or a request with `async_job=true`, returns `202` and a job id:

```bash
curl -s -F "file=@samples/sample.mp3" -F "async_job=true" http://127.0.0.1:8000/v1/transcriptions
curl -s http://127.0.0.1:8000/v1/transcriptions/<job-id>
```

`GET /health` returns `{"status": "ok"}`.

To resume a long file after a stop, pass the same directory again. Finished chunks are skipped:

```bash
transcribe meeting.mp3 --job-dir ./work/meeting -o meeting.json
```

## Layout

| Path | Role |
| --- | --- |
| `src/transcription_pipeline/cli.py` | Script. Accepts a file path. |
| `src/transcription_pipeline/api.py` | Service. Accepts an upload. |
| `src/transcription_pipeline/audio.py` | Probe, silence scan, decode to 16 kHz mono. |
| `src/transcription_pipeline/chunking.py` | Cut long audio on silence, or overlap a hard cut. |
| `src/transcription_pipeline/transcriber.py` | faster-whisper, plus the mock engine used in tests. |
| `src/transcription_pipeline/merge.py` | Put chunk timestamps back on the original timeline. |
| `src/transcription_pipeline/export.py` | JSON, plain text, SRT, WebVTT. |
| `tests/` | What a reviewer can run with `pytest`. |

## Assessment answers

### Accept an audio file

The CLI takes a path: `transcribe recording.mp3`. The service takes `POST /v1/transcriptions` as a multipart upload. The upload is streamed to disk in 1 MB blocks, with a 500 MB cap and a rejection for an empty body, so the file is not held entirely in memory. The original filename is not used as the storage path. `ffprobe` then checks that the bytes contain an audio stream and records codec, sample rate, channels, and duration. A video file is accepted and only its audio track is used.

### Transcribe spoken language

faster-whisper runs Whisper locally. The default is the `base` model, on CPU with `int8`. `small`, `medium`, or `large-v3` are a config change when a GPU is available or accuracy matters more than speed. The caller can pass a language code such as `en`. Otherwise the language detected on the first chunk is locked for the rest of the file. A voice-activity pass skips silence. The model only receives 16 kHz mono PCM.

### Timestamps per segment

The JSON response above is the contract: every segment has `start`, `end`, and `text`, in seconds on the original timeline. Word times are included because captions and search need them. The same transcript is rendered as plain text, SRT, or WebVTT. Chunking is hidden from the caller. Timestamps are already shifted back onto the original file.

### Different audio formats

Format support is a decode step. `ffprobe` identifies the container and codec. `ffmpeg` decodes WAV, MP3, M4A/AAC, FLAC, OGG, Opus, WebM, and the audio track of a video into one representation: 16 kHz, mono, 16-bit PCM. Stereo is downmixed. A wrong extension still works, because ffmpeg identifies the container from the bytes. Anything corrupt, or without an audio stream, fails at this step and the model is never called. Adding a format later means ffmpeg can decode it. The transcription code does not change.

### Long audio files

Whisper already scores audio in windows of about 30 seconds. A long file is a problem of memory, of one failure throwing away the whole job, and of an HTTP call that would stay open for an hour.

The file is not loaded into one array. One streaming ffmpeg pass finds the silences. The timeline is cut into chunks of about 5 minutes. If a pause falls near a boundary, the cut is placed inside that pause. If that region has no pause, the cut is hard and the next chunk overlaps it by 2 seconds. Each word in the overlap is kept by exactly one chunk, the chunk that contains the word's midpoint.

Each chunk is transcribed on its own, its timestamps are shifted by the chunk start, and the result is checkpointed. The wav slice is deleted after the checkpoint is written. A retry with the same `--job-dir` skips finished chunks. Peak memory stays at one chunk plus the model.

Files under 15 minutes are answered inside the HTTP request. Longer files return `202` with a job id. In production that worker moves onto a queue and the audio moves to object storage. The job contract stays the same.
