import subprocess
from pathlib import Path

import pytest

from transcription_pipeline.audio import Binaries
from transcription_pipeline.errors import AudioError
from transcription_pipeline.pipeline import Pipeline
from transcription_pipeline.settings import Settings
from transcription_pipeline.transcriber import MockTranscriber
from tests.support import write_tone


def _ffmpeg() -> Binaries:
    try:
        return Binaries.from_env()
    except AudioError:
        pytest.skip("ffmpeg is not installed")
        raise


def test_missing_file_fails_before_ffmpeg():
    pipeline = Pipeline(MockTranscriber(), settings=Settings(min_tail_seconds=0))
    with pytest.raises(AudioError, match="file not found"):
        pipeline.run(Path("/tmp/does-not-exist-transcription.wav"))


def test_wav_and_mp3_share_one_pipeline(tmp_path: Path):
    binaries = _ffmpeg()
    wav = tmp_path / "tone.wav"
    mp3 = tmp_path / "tone.mp3"
    write_tone(wav, seconds=3.0)
    proc = subprocess.run(
        [binaries.ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", str(wav), "-c:a", "libmp3lame", "-q:a", "4", str(mp3)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr

    settings = Settings(chunk_seconds=300, overlap_seconds=2, min_tail_seconds=0, silence_search_seconds=1)
    wav_engine = MockTranscriber()
    mp3_engine = MockTranscriber()
    wav_result = Pipeline(wav_engine, settings=settings).run(wav, language="en")
    mp3_result = Pipeline(mp3_engine, settings=settings).run(mp3, language="en")

    assert wav_result.source_codec == "pcm_s16le"
    assert mp3_result.source_codec == "mp3"
    assert wav_result.language == "en"
    assert mp3_result.text == "speech 1"
    assert wav_result.segments[0].start == 0
    assert wav_result.segments[0].end == pytest.approx(3.0, abs=0.05)
    assert mp3_result.segments[0].end == pytest.approx(wav_result.duration, abs=0.15)


def test_long_file_is_chunked_and_can_resume(tmp_path: Path):
    _ffmpeg()
    wav = tmp_path / "long.wav"
    write_tone(wav, seconds=5.0)
    settings = Settings(
        chunk_seconds=1.5,
        overlap_seconds=0.3,
        silence_search_seconds=0.1,
        min_tail_seconds=0,
    )
    engine = MockTranscriber()
    job_dir = tmp_path / "job"
    pipeline = Pipeline(engine, settings=settings)
    first = pipeline.run(wav, job_dir=job_dir)
    calls_after_first = engine.calls
    assert calls_after_first > 1
    assert first.duration == pytest.approx(5.0, abs=0.05)
    assert first.segments
    assert all(segment.end >= segment.start for segment in first.segments)

    second = pipeline.run(wav, job_dir=job_dir)
    assert engine.calls == calls_after_first
    assert second.text == first.text
    assert [segment.start for segment in second.segments] == [segment.start for segment in first.segments]
