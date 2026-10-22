"""Command line entry: transcribe a file and print JSON, text, SRT, or VTT."""

import argparse
import logging
import sys
from pathlib import Path

from transcription_pipeline.errors import PipelineError
from transcription_pipeline.export import FORMATS, render
from transcription_pipeline.pipeline import Pipeline
from transcription_pipeline.settings import Settings
from transcription_pipeline.transcriber import build_transcriber


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="transcribe",
        description="Transcribe an audio file into timestamped text.",
    )
    parser.add_argument("audio", type=Path, help="WAV, MP3, M4A, FLAC, OGG, or any format ffmpeg can decode")
    parser.add_argument("-o", "--output", type=Path, help="Write the result to this path. Default: stdout")
    parser.add_argument("-f", "--format", choices=FORMATS, default="json")
    parser.add_argument("-l", "--language", default=None, help="Language code, for example en. Default: detect")
    parser.add_argument("--engine", choices=("faster-whisper", "mock"), default=None)
    parser.add_argument("--model", default="base", help="Whisper size: tiny, base, small, medium, large-v3")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--compute-type", default="default")
    parser.add_argument("--chunk-seconds", type=float, default=300.0)
    parser.add_argument("--overlap-seconds", type=float, default=2.0)
    parser.add_argument("--stream", type=int, default=0, help="Audio stream index when the file has more than one")
    parser.add_argument(
        "--job-dir",
        type=Path,
        default=None,
        help="Keep chunk checkpoints here and resume a long file",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    settings = Settings(
        model_size=args.model,
        device=args.device,
        compute_type=args.compute_type,
        chunk_seconds=args.chunk_seconds,
        overlap_seconds=args.overlap_seconds,
    )
    pipeline = Pipeline(build_transcriber(settings, args.engine), settings=settings)
    language = args.language.strip() if args.language else None
    try:
        transcript = pipeline.run(
            args.audio,
            language=language or None,
            job_dir=args.job_dir,
            stream_index=args.stream,
        )
    except PipelineError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    body = render(transcript, args.format)
    if not body.endswith("\n"):
        body += "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(body)
    else:
        sys.stdout.write(body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
