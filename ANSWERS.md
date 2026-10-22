# What to submit

Live demo: https://transcription-pipeline.vercel.app

Source: https://github.com/usmankicsit/transcription-pipeline

The form has two kinds of questions.

- If the box says **Add file**, upload `transcription-pipeline.zip`.
- If the box is a text field, paste the matching section below. Each one already contains the links.

## Implement a service or script that accepts an audio file (WAV or MP3)

Live demo: https://transcription-pipeline.vercel.app
Source: https://github.com/usmankicsit/transcription-pipeline

The page accepts a dropped file. The service accepts `POST /v1/transcriptions` as a multipart upload, and the CLI accepts a path: `transcribe recording.mp3`. The upload is streamed to disk in 1 MB blocks, with a 500 MB cap and a rejection for an empty body. The original filename is not used as the storage path. `ffprobe` then checks that the bytes contain an audio stream and records the codec, sample rate, channels, and duration. A video file is accepted and only its audio track is used. WAV and MP3 are both covered by `tests/test_pipeline.py`.

## Implement a service or script that transcribes spoken language into text

Live demo: https://transcription-pipeline.vercel.app
Source: https://github.com/usmankicsit/transcription-pipeline

faster-whisper runs Whisper locally. Nothing is trained. The default is the `base` model, on CPU with `int8`. `small`, `medium`, or `large-v3` are a config change when a GPU is available or accuracy matters more than speed. The caller can pass a language code such as `en`. Otherwise the language detected on the first chunk is locked for the rest of the file. A voice-activity pass skips silence. The model only receives 16 kHz mono PCM. The checked-in result is in `examples/sample.transcript.json`. On the public page, press **Use the spoken sample**.

## Implement a service or script that returns the transcription with timestamps per segment

Live demo: https://transcription-pipeline.vercel.app
Source: https://github.com/usmankicsit/transcription-pipeline

The canonical response is JSON. Times are seconds on the original file. Every segment has `start`, `end`, and `text`. Word times are included because captions and search need them. The same transcript is rendered as plain text, SRT, or WebVTT from the page, from `-f txt|srt|vtt`, or from the `response_format` field. Chunking is hidden from the caller. Timestamps are already shifted back onto the original file.

```json
{
  "language": "en",
  "duration": 6.84,
  "text": "This is a short sample for the transcription pipeline. The words should come back as text with timestamps.",
  "segments": [
    {
      "start": 0.0,
      "end": 2.9,
      "text": "This is a short sample for the transcription pipeline.",
      "words": [{"start": 0.0, "end": 0.2, "text": "This"}]
    }
  ]
}
```

## How do you handle different audio formats?

Format support is a decode step. `ffprobe` identifies the container and codec. `ffmpeg` decodes WAV, MP3, M4A/AAC, FLAC, OGG, Opus, WebM, and the audio track of a video into one representation: 16 kHz, mono, 16-bit PCM. Stereo is downmixed. A wrong extension still works, because ffmpeg identifies the container from the bytes. Anything corrupt, or without an audio stream, fails at this step and the model is never called. Adding a format later means ffmpeg can decode it. The transcription code does not change. The public demo does the same normalization in the browser, then runs Whisper there.

## How do you deal with long audio files?

Whisper already scores audio in windows of about 30 seconds. A long file is a problem of memory, of one failure throwing away the whole job, and of an HTTP call that would stay open for an hour.

The file is not loaded into one array. One streaming ffmpeg pass finds the silences. The timeline is cut into chunks of about 5 minutes. If a pause falls near a boundary, the cut is placed inside that pause. If that region has no pause, the cut is hard and the next chunk overlaps it by 2 seconds. Each word in the overlap is kept by exactly one chunk, the chunk that contains the word's midpoint.

Each chunk is transcribed on its own, its timestamps are shifted by the chunk start, and the result is checkpointed. The wav slice is deleted after the checkpoint is written. A retry with the same `--job-dir` skips finished chunks. Peak memory stays at one chunk plus the model.

Files under 15 minutes are answered inside the HTTP request. Longer files return `202` with a job id. In production that worker moves onto a queue and the audio moves to object storage. The public page runs in the browser, so it keeps the same 30 second windows and asks for the Python service once a file is larger than 25 MB. A serverless host cannot hold the model and a multi-hour job, which is why the long-file path lives in the Python service.
