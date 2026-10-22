"""Speech engines.

`FasterWhisperTranscriber` is the real engine: CTranslate2 running Whisper
locally, with word timestamps and a VAD pass so silence is not transcribed.
`MockTranscriber` returns deterministic text from the wav duration so the
rest of the pipeline can be tested without model weights.
"""

import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from transcription_pipeline.errors import TranscriptionError
from transcription_pipeline.schemas import Segment, Word


@dataclass
class EngineResult:
    language: str | None
    segments: list[Segment]


class Transcriber(Protocol):
    def transcribe(self, wav_path: Path, *, language: str | None = None) -> EngineResult:
        """Return segments whose times start at zero for this wav."""


class MockTranscriber:
    def __init__(self) -> None:
        self.calls = 0

    def transcribe(self, wav_path: Path, *, language: str | None = None) -> EngineResult:
        self.calls += 1
        duration = _wav_duration(wav_path)
        text = f"speech {self.calls}"
        return EngineResult(
            language=language or "en",
            segments=[
                Segment(
                    start=0.0,
                    end=duration,
                    text=text,
                    words=[Word(start=0.0, end=duration, text=text)],
                )
            ],
        )


class FasterWhisperTranscriber:
    def __init__(
        self,
        model_size: str = "base",
        device: str = "auto",
        compute_type: str = "default",
        beam_size: int = 5,
    ) -> None:
        self.model_size = model_size
        self.device, self.compute_type = resolve_device(device, compute_type)
        self.beam_size = beam_size
        self._model = None

    def transcribe(self, wav_path: Path, *, language: str | None = None) -> EngineResult:
        try:
            model = self._load()
            # The pipeline has already decoded to 16 kHz mono PCM. Passing the
            # samples avoids a second decode through PyAV, whose API differs
            # across releases.
            segments_iter, info = model.transcribe(
                _pcm_samples(wav_path),
                language=language,
                word_timestamps=True,
                vad_filter=True,
                beam_size=self.beam_size,
                condition_on_previous_text=True,
            )
            segments: list[Segment] = []
            for segment in segments_iter:
                words = [
                    Word(start=float(word.start), end=float(word.end), text=word.word.strip())
                    for word in (segment.words or [])
                    if getattr(word, "word", "").strip()
                ]
                text = (segment.text or "").strip()
                if not text and not words:
                    continue
                segments.append(
                    Segment(
                        start=float(segment.start),
                        end=float(segment.end),
                        text=text,
                        words=words,
                    )
                )
        except TranscriptionError:
            raise
        except Exception as exc:
            raise TranscriptionError(str(exc)) from exc
        return EngineResult(language=getattr(info, "language", None), segments=segments)

    def _load(self):
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise TranscriptionError(
                    "faster-whisper is not installed. Install the project dependencies, "
                    "or run with --engine mock."
                ) from exc
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
        return self._model


def build_transcriber(settings, engine: str | None = None) -> Transcriber:
    import os

    chosen = engine or os.environ.get("TRANSCRIBE_ENGINE", "faster-whisper")
    if chosen == "mock":
        return MockTranscriber()
    if chosen != "faster-whisper":
        raise ValueError(f"unknown engine: {chosen}")
    return FasterWhisperTranscriber(
        model_size=settings.model_size,
        device=settings.device,
        compute_type=settings.compute_type,
        beam_size=settings.beam_size,
    )


def resolve_device(device: str, compute_type: str) -> tuple[str, str]:
    resolved = device
    if device == "auto":
        resolved = "cpu"
        try:
            import ctranslate2

            if ctranslate2.get_cuda_device_count() > 0:
                resolved = "cuda"
        except Exception:
            resolved = "cpu"
    resolved_compute = compute_type
    if compute_type == "default":
        resolved_compute = "float16" if resolved == "cuda" else "int8"
    return resolved, resolved_compute


def _pcm_samples(path: Path):
    import numpy as np

    try:
        with wave.open(str(path), "rb") as handle:
            if handle.getnchannels() != 1 or handle.getsampwidth() != 2:
                raise TranscriptionError(f"expected 16-bit mono wav, got {path.name}")
            rate = handle.getframerate()
            frames = handle.readframes(handle.getnframes())
    except wave.Error as exc:
        raise TranscriptionError(f"not a readable wav: {path.name}") from exc
    if rate != 16000:
        raise TranscriptionError(f"expected 16 kHz audio, got {rate} Hz")
    if not frames:
        raise TranscriptionError("audio chunk is empty")
    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0


def _wav_duration(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as handle:
            rate = handle.getframerate() or 1
            return handle.getnframes() / float(rate)
    except wave.Error as exc:
        raise TranscriptionError(f"not a readable wav: {path.name}") from exc
