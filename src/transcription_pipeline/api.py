"""HTTP API for the same pipeline the CLI uses.

Short files are transcribed inside the request. A long file, or a client
that asks for it, gets a job id and is polled. The worker is in-process so
the service runs with no extra infrastructure. A production deployment moves
that worker onto a queue without changing the job contract.
"""

import asyncio
import logging
import os
import re
import shutil
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from transcription_pipeline.errors import AudioError, PipelineError
from transcription_pipeline.export import FORMATS, media_type, render
from transcription_pipeline.pipeline import Pipeline
from transcription_pipeline.settings import Settings
from transcription_pipeline.transcriber import build_transcriber

logger = logging.getLogger(__name__)

_JOB_ID = re.compile(r"[0-9a-f]{32}")
_LANGUAGE = re.compile(r"[A-Za-z]{2,3}(-[A-Za-z0-9]{2,8})?")
_ALLOWED_SUFFIXES = {
    ".wav",
    ".mp3",
    ".m4a",
    ".flac",
    ".ogg",
    ".opus",
    ".webm",
    ".mp4",
    ".aac",
    ".wma",
}

_WEB_DIR = Path(__file__).resolve().parents[2] / "web"


class Job:
    def __init__(self, job_id: str, path: Path) -> None:
        self.id = job_id
        self.path = path
        self.status = "queued"
        self.transcript = None
        self.error: str | None = None


def create_app(
    pipeline: Pipeline | None = None,
    settings: Settings | None = None,
    duration_of=None,
) -> FastAPI:
    settings = settings or Settings()
    if pipeline is None:
        pipeline = Pipeline(build_transcriber(settings), settings=settings)

    app = FastAPI(title="Transcription pipeline", version="1.0.0")
    app.state.settings = settings
    app.state.pipeline = pipeline
    app.state.jobs = {}
    app.state.lock = threading.Lock()
    app.state.duration_of = duration_of

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/transcriptions")
    async def create(
        file: UploadFile = File(...),
        language: str | None = Form(default=None),
        response_format: str = Form(default="json"),
        async_job: str | None = Form(default=None),
    ):
        response_format = _check_format(response_format)
        language = _check_language(language)
        job_id = uuid.uuid4().hex
        job_dir = settings.work_root / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        dest = job_dir / f"source{_suffix(file.filename)}"
        try:
            size = await _save_upload(file, dest, settings.max_upload_bytes)
        except HTTPException:
            shutil.rmtree(job_dir, ignore_errors=True)
            raise
        if size == 0:
            shutil.rmtree(job_dir, ignore_errors=True)
            raise HTTPException(status_code=400, detail="empty file")

        run_async = _wants_async(async_job)
        if not run_async:
            try:
                duration = _duration(app, dest)
            except AudioError as exc:
                shutil.rmtree(job_dir, ignore_errors=True)
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            run_async = duration <= 0 or duration > settings.sync_max_seconds

        if run_async:
            job = Job(job_id, dest)
            with app.state.lock:
                app.state.jobs[job_id] = job
            _launch(app, job, language)
            return JSONResponse(
                {"id": job_id, "status": "queued", "status_url": f"/v1/transcriptions/{job_id}"},
                status_code=202,
            )

        try:
            transcript = await asyncio.to_thread(
                pipeline.run, dest, language=language, job_dir=job_dir
            )
        except AudioError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except PipelineError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        finally:
            shutil.rmtree(job_dir, ignore_errors=True)
        return _transcript_response(transcript, response_format)

    @app.get("/v1/transcriptions/{job_id}")
    def get_job(job_id: str, response_format: str = "json"):
        job = _require_job(app, job_id)
        response_format = _check_format(response_format)
        if job.status != "succeeded":
            return {"id": job.id, "status": job.status, "error": job.error}
        if response_format == "json":
            return {
                "id": job.id,
                "status": job.status,
                "transcript": job.transcript.model_dump(),
            }
        return Response(render(job.transcript, response_format), media_type=media_type(response_format))

    @app.post("/v1/transcriptions/{job_id}/retry")
    def retry(job_id: str):
        job = _require_job(app, job_id)
        if job.status != "failed":
            raise HTTPException(status_code=409, detail="only a failed job can be retried")
        if not job.path.is_file():
            raise HTTPException(status_code=409, detail="the uploaded audio is no longer on disk")
        with app.state.lock:
            job.status = "queued"
            job.error = None
        _launch(app, job, None)
        return JSONResponse(
            {"id": job.id, "status": "queued", "status_url": f"/v1/transcriptions/{job.id}"},
            status_code=202,
        )

    if _WEB_DIR.is_dir():
        app.mount("/", StaticFiles(directory=_WEB_DIR, html=True), name="web")

    return app


def main() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    host = os.environ.get("TRANSCRIBE_HOST", "127.0.0.1")
    port = int(os.environ.get("TRANSCRIBE_PORT", "8000"))
    uvicorn.run(create_app(), host=host, port=port)


def _launch(app: FastAPI, job: Job, language: str | None) -> None:
    pipeline: Pipeline = app.state.pipeline

    def work() -> None:
        with app.state.lock:
            job.status = "running"
        try:
            transcript = pipeline.run(job.path, language=language, job_dir=job.path.parent)
        except PipelineError as exc:
            logger.exception("job %s failed", job.id)
            with app.state.lock:
                job.status = "failed"
                job.error = str(exc)
            return
        except Exception as exc:
            logger.exception("job %s failed", job.id)
            with app.state.lock:
                job.status = "failed"
                job.error = str(exc)
            return
        with app.state.lock:
            job.status = "succeeded"
            job.transcript = transcript
            job.error = None

    threading.Thread(target=work, name=f"transcribe-{job.id}", daemon=True).start()


def _duration(app: FastAPI, path: Path) -> float:
    if app.state.duration_of is not None:
        return float(app.state.duration_of(path))
    from transcription_pipeline.audio import Binaries, probe

    return probe(path, Binaries.from_env()).duration


async def _save_upload(upload: UploadFile, dest: Path, limit: int) -> int:
    size = 0
    with dest.open("wb") as handle:
        while True:
            block = await upload.read(1024 * 1024)
            if not block:
                break
            size += len(block)
            if size > limit:
                raise HTTPException(status_code=413, detail="file is too large")
            handle.write(block)
    return size


def _suffix(filename: str | None) -> str:
    suffix = Path(filename or "audio").suffix.lower()
    if suffix not in _ALLOWED_SUFFIXES:
        return ".audio"
    return suffix


def _check_format(response_format: str) -> str:
    if response_format not in FORMATS:
        raise HTTPException(status_code=400, detail="response_format must be json, txt, srt, or vtt")
    return response_format


def _check_language(language: str | None) -> str | None:
    if language is None:
        return None
    language = language.strip()
    if not language:
        return None
    if not _LANGUAGE.fullmatch(language):
        raise HTTPException(status_code=400, detail="language must be a code such as en")
    return language


def _wants_async(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _require_job(app: FastAPI, job_id: str) -> Job:
    if not _JOB_ID.fullmatch(job_id):
        raise HTTPException(status_code=404, detail="job not found")
    with app.state.lock:
        job = app.state.jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job


def _transcript_response(transcript, response_format: str):
    if response_format == "json":
        return JSONResponse(transcript.model_dump())
    return Response(render(transcript, response_format), media_type=media_type(response_format))


app = create_app()
