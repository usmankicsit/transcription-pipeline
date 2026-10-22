import json
from pathlib import Path

import pytest

from transcription_pipeline.audio import Binaries
from transcription_pipeline.cli import main
from transcription_pipeline.errors import AudioError
from tests.support import write_tone


def test_cli_mock_writes_timestamped_json(tmp_path: Path):
    try:
        Binaries.from_env()
    except AudioError:
        pytest.skip("ffmpeg is not installed")
    wav = tmp_path / "note.wav"
    write_tone(wav, seconds=1.5)
    out = tmp_path / "out.json"
    assert main(["--engine", "mock", "-l", "en", "-o", str(out), str(wav)]) == 0
    body = json.loads(out.read_text())
    assert body["language"] == "en"
    assert body["text"]
    assert body["segments"][0]["start"] == 0
    assert body["segments"][0]["end"] > 0
    assert "start" in body["segments"][0]["words"][0]


def test_cli_missing_file_returns_nonzero():
    assert main(["--engine", "mock", "/tmp/does-not-exist-transcription.wav"]) == 1
