"""Decode every container down to 16 kHz mono PCM.

The speech model never sees the original format. ffmpeg is the demuxer, so
WAV, MP3, M4A, FLAC, OGG, Opus, WebM, and the audio track of a video all
become the same kind of slice. A wrong file extension still works, because
ffmpeg identifies the container from the bytes.
"""

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from transcription_pipeline.errors import AudioError

_SILENCE_START = re.compile(r"silence_start:\s*([0-9.]+)")
_SILENCE_END = re.compile(r"silence_end:\s*([0-9.]+)")


@dataclass(frozen=True)
class AudioInfo:
    path: Path
    format_name: str
    codec: str
    sample_rate: int
    channels: int
    duration: float
    audio_streams: int


@dataclass(frozen=True)
class Binaries:
    ffmpeg: str
    ffprobe: str

    @classmethod
    def from_env(cls) -> "Binaries":
        ffmpeg = os.environ.get("FFMPEG_BINARY") or shutil.which("ffmpeg")
        ffprobe = os.environ.get("FFPROBE_BINARY") or shutil.which("ffprobe")
        if not ffmpeg or not ffprobe:
            raise AudioError(
                "ffmpeg and ffprobe are required on PATH. "
                "Install them with your package manager, or set FFMPEG_BINARY and FFPROBE_BINARY."
            )
        return cls(ffmpeg=ffmpeg, ffprobe=ffprobe)


def probe(path: Path, binaries: Binaries, stream_index: int = 0) -> AudioInfo:
    proc = _run(
        [
            binaries.ffprobe,
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ]
    )
    try:
        payload = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise AudioError("ffprobe returned unreadable output") from exc

    streams = [stream for stream in payload.get("streams", []) if stream.get("codec_type") == "audio"]
    if not streams:
        raise AudioError("no audio stream in this file")
    if stream_index < 0 or stream_index >= len(streams):
        raise AudioError(
            f"audio stream {stream_index} is not in this file ({len(streams)} audio stream(s))"
        )
    stream = streams[stream_index]
    fmt = payload.get("format") or {}
    return AudioInfo(
        path=path,
        format_name=str(fmt.get("format_name") or "unknown"),
        codec=str(stream.get("codec_name") or "unknown"),
        sample_rate=int(_number(stream.get("sample_rate"))),
        channels=int(_number(stream.get("channels"))),
        duration=_number(stream.get("duration") or fmt.get("duration")),
        audio_streams=len(streams),
    )


def detect_silences(
    path: Path,
    binaries: Binaries,
    *,
    noise_db: float,
    min_duration: float,
    stream_index: int = 0,
    file_duration: float | None = None,
) -> list[tuple[float, float]]:
    """One streaming pass. Nothing is written to disk."""
    proc = _run(
        [
            binaries.ffmpeg,
            "-hide_banner",
            "-i",
            str(path),
            "-map",
            f"0:a:{stream_index}",
            "-af",
            f"silencedetect=noise={noise_db}dB:d={min_duration}",
            "-f",
            "null",
            "-",
        ],
        ok_codes=(0,),
    )
    # silencedetect writes its log to stderr even on success.
    return parse_silences(proc.stderr or "", file_duration)


def parse_silences(log: str, duration: float | None = None) -> list[tuple[float, float]]:
    pending: list[float] = []
    regions: list[tuple[float, float]] = []
    for line in log.splitlines():
        start = _SILENCE_START.search(line)
        end = _SILENCE_END.search(line)
        if start:
            pending.append(float(start.group(1)))
        if end and pending:
            regions.append((pending.pop(0), float(end.group(1))))
    if duration is not None:
        for start in pending:
            if duration > start:
                regions.append((start, duration))
    return regions


def decode_wav(
    src: Path,
    dest: Path,
    binaries: Binaries,
    *,
    sample_rate: int,
    stream_index: int = 0,
) -> None:
    """Decode a whole file when the container does not advertise a duration."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            binaries.ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(src),
            "-map",
            f"0:a:{stream_index}",
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-c:a",
            "pcm_s16le",
            str(dest),
        ]
    )
    if not dest.is_file() or dest.stat().st_size == 0:
        raise AudioError("ffmpeg produced an empty wav")


def extract_wav(
    src: Path,
    dest: Path,
    start: float,
    end: float,
    binaries: Binaries,
    *,
    sample_rate: int,
    stream_index: int = 0,
) -> None:
    """Slice and normalize in one ffmpeg call.

    `-ss` is placed before `-i` so a late chunk of a long file does not
    decode from the beginning. That seek is keyframe-accurate on compressed
    audio, which is why hard cuts carry a short overlap.
    """
    duration = end - start
    if duration <= 0:
        raise AudioError("cannot extract a chunk with no duration")
    dest.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            binaries.ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{start:.3f}",
            "-i",
            str(src),
            "-t",
            f"{duration:.3f}",
            "-map",
            f"0:a:{stream_index}",
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-c:a",
            "pcm_s16le",
            str(dest),
        ]
    )
    if not dest.is_file() or dest.stat().st_size == 0:
        raise AudioError("ffmpeg produced an empty wav")


def _number(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


def _run(cmd: list[str], ok_codes: tuple[int, ...] = (0,)) -> subprocess.CompletedProcess[str]:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        raise AudioError(f"could not run {cmd[0]}") from exc
    if proc.returncode not in ok_codes:
        detail = (proc.stderr or proc.stdout or "").strip().splitlines()
        tail = detail[-1] if detail else f"{cmd[0]} failed"
        tail = re.sub(r"(?:^|\s)/\S+", " ", tail)
        tail = re.sub(r"\s+", " ", tail).strip(" :")
        raise AudioError(tail[-400:] or f"{cmd[0]} failed")
    return proc
