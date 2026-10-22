"""Orchestrate probe, chunk, transcribe, merge.

Each chunk is checkpointed. A retry with the same `--job-dir` skips chunks
that already finished. Language detected on the first chunk is locked for
the rest of the file, so a long recording does not switch language mid-way.
"""

import hashlib
import json
import logging
import shutil
import tempfile
from pathlib import Path

from transcription_pipeline.audio import Binaries, decode_wav, detect_silences, extract_wav, probe
from transcription_pipeline.chunking import Span, plan_spans
from transcription_pipeline.errors import AudioError, TranscriptionError
from transcription_pipeline.merge import ChunkTranscript, merge_chunks
from transcription_pipeline.schemas import Segment, Transcript, Word
from transcription_pipeline.settings import Settings
from transcription_pipeline.transcriber import Transcriber

logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(
        self,
        transcriber: Transcriber,
        *,
        settings: Settings | None = None,
        binaries: Binaries | None = None,
    ) -> None:
        self.transcriber = transcriber
        self.settings = settings or Settings()
        self._binaries = binaries

    def run(
        self,
        source: Path,
        *,
        language: str | None = None,
        job_dir: Path | None = None,
        stream_index: int = 0,
    ) -> Transcript:
        source = Path(source)
        if not source.is_file():
            raise AudioError(f"file not found: {source}")

        binaries = self._binaries or Binaries.from_env()
        info = probe(source, binaries, stream_index=stream_index)
        cleanup = job_dir is None
        work = Path(job_dir) if job_dir is not None else Path(tempfile.mkdtemp(prefix="transcribe-"))
        work.mkdir(parents=True, exist_ok=True)
        try:
            media = source
            duration = info.duration
            if duration <= 0:
                media = work / "decoded.wav"
                if not media.is_file():
                    decode_wav(
                        source,
                        media,
                        binaries,
                        sample_rate=self.settings.sample_rate,
                        stream_index=stream_index,
                    )
                duration = probe(media, binaries).duration
                if duration <= 0:
                    raise AudioError("could not determine the audio duration")
            silences = detect_silences(
                media,
                binaries,
                noise_db=self.settings.silence_db,
                min_duration=self.settings.min_silence_seconds,
                stream_index=stream_index,
                file_duration=duration,
            )
            spans = plan_spans(
                duration,
                silences,
                target=self.settings.chunk_seconds,
                overlap=self.settings.overlap_seconds,
                search=self.settings.silence_search_seconds,
                min_tail=self.settings.min_tail_seconds,
            )
            if not spans:
                raise AudioError("audio duration is zero")
            self._sync_plan(work, source, spans)
            logger.info(
                "transcribing %s (%s, %.1fs) in %d chunk(s)",
                source.name,
                info.codec,
                duration,
                len(spans),
            )

            locked = language
            chunks: list[ChunkTranscript] = []
            for span in spans:
                checkpoint = work / f"chunk-{span.index:04d}.json"
                if checkpoint.is_file():
                    loaded_language, segments = _read_checkpoint(checkpoint)
                    if locked is None and loaded_language:
                        locked = loaded_language
                    chunks.append(ChunkTranscript(span.start, span.end, segments))
                    logger.info("chunk %d/%d resumed from checkpoint", span.index + 1, len(spans))
                    continue

                piece = work / f"chunk-{span.index:04d}.wav"
                extract_wav(
                    media,
                    piece,
                    span.start,
                    span.end,
                    binaries,
                    sample_rate=self.settings.sample_rate,
                    stream_index=stream_index,
                )
                try:
                    result = self.transcriber.transcribe(piece, language=locked)
                except TranscriptionError:
                    raise
                except Exception as exc:
                    raise TranscriptionError(f"chunk {span.index} failed: {exc}") from exc
                finally:
                    piece.unlink(missing_ok=True)

                if locked is None and result.language:
                    locked = result.language
                _write_checkpoint(checkpoint, span, result.language, result.segments)
                chunks.append(ChunkTranscript(span.start, span.end, result.segments))
                logger.info("chunk %d/%d done", span.index + 1, len(spans))

            segments = [_quantize_segment(segment) for segment in merge_chunks(chunks)]
            text = " ".join(segment.text.strip() for segment in segments if segment.text.strip())
            return Transcript(
                language=locked,
                duration=round(duration, 3),
                text=text,
                segments=segments,
                source_format=info.format_name,
                source_codec=info.codec,
            )
        finally:
            if cleanup:
                shutil.rmtree(work, ignore_errors=True)

    def _sync_plan(self, work: Path, source: Path, spans: list[Span]) -> None:
        stat = source.stat()
        payload = {
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "chunk_seconds": self.settings.chunk_seconds,
            "overlap_seconds": self.settings.overlap_seconds,
            "spans": [(span.start, span.end) for span in spans],
        }
        key = hashlib.sha256(json.dumps(payload).encode()).hexdigest()[:16]
        plan_path = work / "plan.json"
        if plan_path.is_file():
            try:
                existing = json.loads(plan_path.read_text())
            except json.JSONDecodeError:
                existing = {}
            if existing.get("key") != key:
                for old in work.glob("chunk-*.json"):
                    old.unlink()
        plan_path.write_text(json.dumps({"key": key}))


def _write_checkpoint(path: Path, span: Span, language: str | None, segments: list[Segment]) -> None:
    path.write_text(
        json.dumps(
            {
                "start": span.start,
                "end": span.end,
                "language": language,
                "segments": [segment.model_dump() for segment in segments],
            }
        )
    )


def _read_checkpoint(path: Path) -> tuple[str | None, list[Segment]]:
    data = json.loads(path.read_text())
    segments = [Segment.model_validate(segment) for segment in data.get("segments", [])]
    return data.get("language"), segments


def _quantize_segment(segment: Segment) -> Segment:
    return Segment(
        start=_q(segment.start),
        end=_q(segment.end),
        text=segment.text.strip(),
        words=[Word(start=_q(word.start), end=_q(word.end), text=word.text) for word in segment.words],
    )


def _q(value: float) -> float:
    return round(float(value), 3)
