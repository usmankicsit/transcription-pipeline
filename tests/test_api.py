import time

from fastapi.testclient import TestClient

from transcription_pipeline.api import create_app
from transcription_pipeline.errors import AudioError
from transcription_pipeline.schemas import Segment, Transcript, Word
from transcription_pipeline.settings import Settings


def _transcript() -> Transcript:
    return Transcript(
        language="en",
        duration=1.5,
        text="hello world",
        segments=[
            Segment(
                start=0.0,
                end=1.5,
                text="hello world",
                words=[Word(start=0.0, end=1.5, text="hello world")],
            )
        ],
        source_format="wav",
        source_codec="pcm_s16le",
    )


class FlakyPipeline:
    def __init__(self) -> None:
        self.calls = 0

    def run(self, source, *, language=None, job_dir=None, stream_index=0):
        self.calls += 1
        if self.calls == 1:
            raise AudioError("decode failed")
        return _transcript()


class StubPipeline:
    def __init__(self, delay: float = 0.0) -> None:
        self.delay = delay
        self.calls = 0

    def run(self, source, *, language=None, job_dir=None, stream_index=0):
        self.calls += 1
        if self.delay:
            time.sleep(self.delay)
        return _transcript()


def _client(tmp_path, pipeline=None, duration=1.0, **settings):
    app = create_app(
        pipeline=pipeline or StubPipeline(),
        settings=Settings(work_root=tmp_path, max_upload_bytes=1024, sync_max_seconds=30, **settings),
        duration_of=lambda path: duration,
    )
    return TestClient(app), app


def test_health_and_form(tmp_path):
    client, _app = _client(tmp_path)
    assert client.get("/health").json() == {"status": "ok"}
    page = client.get("/")
    assert page.status_code == 200
    assert 'name="file"' in page.text


def test_sync_upload_returns_segments(tmp_path):
    client, _app = _client(tmp_path)
    response = client.post(
        "/v1/transcriptions",
        files={"file": ("note.wav", b"RIFFdemo", "audio/wav")},
        data={"language": "en", "response_format": "json"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["text"] == "hello world"
    assert body["segments"][0]["start"] == 0.0
    assert body["segments"][0]["end"] == 1.5


def test_srt_response(tmp_path):
    client, _app = _client(tmp_path)
    response = client.post(
        "/v1/transcriptions",
        files={"file": ("note.wav", b"RIFFdemo", "audio/wav")},
        data={"response_format": "srt"},
    )
    assert response.status_code == 200
    assert "-->" in response.text
    assert "hello world" in response.text


def test_empty_file_is_rejected(tmp_path):
    client, _app = _client(tmp_path)
    response = client.post("/v1/transcriptions", files={"file": ("note.wav", b"", "audio/wav")})
    assert response.status_code == 400


def test_oversize_upload_is_rejected(tmp_path):
    client, _app = _client(tmp_path)
    response = client.post(
        "/v1/transcriptions",
        files={"file": ("note.wav", b"x" * 2048, "audio/wav")},
    )
    assert response.status_code == 413


def test_bad_language_is_rejected(tmp_path):
    client, _app = _client(tmp_path)
    response = client.post(
        "/v1/transcriptions",
        files={"file": ("note.wav", b"RIFFdemo", "audio/wav")},
        data={"language": "not a language"},
    )
    assert response.status_code == 400


def test_long_file_returns_a_job(tmp_path):
    client, _app = _client(tmp_path, pipeline=StubPipeline(delay=0.2), duration=3600)
    response = client.post(
        "/v1/transcriptions",
        files={"file": ("meeting.mp3", b"id3demo", "audio/mpeg")},
    )
    assert response.status_code == 202
    job_id = response.json()["id"]
    status = "queued"
    body = {}
    for _ in range(50):
        body = client.get(f"/v1/transcriptions/{job_id}").json()
        status = body["status"]
        if status in {"succeeded", "failed"}:
            break
        time.sleep(0.05)
    assert status == "succeeded"
    assert body["transcript"]["text"] == "hello world"


def test_unknown_job_is_not_found(tmp_path):
    client, _app = _client(tmp_path)
    assert client.get("/v1/transcriptions/" + ("a" * 32)).status_code == 404
    assert client.get("/v1/transcriptions/nope").status_code == 404


def _wait_for(client, job_id: str) -> dict:
    body = {}
    for _ in range(50):
        body = client.get(f"/v1/transcriptions/{job_id}").json()
        if body["status"] in {"succeeded", "failed"}:
            return body
        time.sleep(0.05)
    return body


def test_two_uploads_stay_isolated(tmp_path):
    client, _app = _client(tmp_path, pipeline=StubPipeline(delay=0.2), duration=3600)
    first = client.post("/v1/transcriptions", files={"file": ("a.mp3", b"one", "audio/mpeg")})
    second = client.post("/v1/transcriptions", files={"file": ("b.wav", b"two", "audio/wav")})
    assert first.status_code == 202
    assert second.status_code == 202
    assert first.json()["id"] != second.json()["id"]
    assert _wait_for(client, first.json()["id"])["status"] == "succeeded"
    assert _wait_for(client, second.json()["id"])["status"] == "succeeded"


def test_failed_job_retries_without_a_new_upload(tmp_path):
    pipeline = FlakyPipeline()
    client, _app = _client(tmp_path, pipeline=pipeline, duration=3600)
    created = client.post(
        "/v1/transcriptions",
        files={"file": ("note.wav", b"RIFFdemo", "audio/wav")},
        data={"async_job": "true"},
    )
    job_id = created.json()["id"]
    assert _wait_for(client, job_id)["status"] == "failed"
    retry = client.post(f"/v1/transcriptions/{job_id}/retry")
    assert retry.status_code == 202
    assert _wait_for(client, job_id)["status"] == "succeeded"
    assert pipeline.calls == 2


def test_explicit_async_flag(tmp_path):
    client, _app = _client(tmp_path, pipeline=StubPipeline())
    response = client.post(
        "/v1/transcriptions",
        files={"file": ("note.wav", b"RIFFdemo", "audio/wav")},
        data={"async_job": "true"},
    )
    assert response.status_code == 202
    job_id = response.json()["id"]
    for _ in range(50):
        body = client.get(f"/v1/transcriptions/{job_id}").json()
        if body["status"] == "succeeded":
            break
        time.sleep(0.05)
    assert body["status"] == "succeeded"
