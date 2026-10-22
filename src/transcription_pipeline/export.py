"""Downstream shapes of one transcript: JSON, plain text, SRT, and WebVTT."""

import json
import re

from transcription_pipeline.schemas import Transcript

FORMATS = ("json", "txt", "srt", "vtt")


def render(transcript: Transcript, response_format: str) -> str:
    if response_format == "json":
        return json.dumps(transcript.model_dump(), indent=2)
    if response_format == "txt":
        return "\n".join(segment.text.strip() for segment in transcript.segments if segment.text.strip()) + "\n"
    if response_format == "srt":
        return _cues(transcript, decimal=",")
    if response_format == "vtt":
        body = _cues(transcript, decimal=".")
        return "WEBVTT\n\n" + body
    raise ValueError(f"unknown format: {response_format}")


def media_type(response_format: str) -> str:
    if response_format == "json":
        return "application/json"
    return "text/plain; charset=utf-8"


def format_timestamp(seconds: float, decimal: str = ",") -> str:
    if seconds < 0:
        seconds = 0.0
    total_ms = int(round(seconds * 1000))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{decimal}{millis:03d}"


def _cues(transcript: Transcript, decimal: str) -> str:
    blocks: list[str] = []
    for index, segment in enumerate(transcript.segments, start=1):
        text = re.sub(r"\s+", " ", segment.text).strip()
        if not text:
            continue
        start = format_timestamp(segment.start, decimal)
        end = format_timestamp(segment.end, decimal)
        blocks.append(f"{index}\n{start} --> {end}\n{text}\n")
    return "\n".join(blocks)
