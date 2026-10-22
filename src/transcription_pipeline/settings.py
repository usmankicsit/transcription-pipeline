import tempfile
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    """Knobs a deployer actually changes. Defaults target a CPU box."""

    model_size: str = "base"
    device: str = "auto"
    compute_type: str = "default"
    beam_size: int = 5
    # Five minutes is long enough that a meeting is a handful of model calls,
    # and short enough that one failed chunk is cheap to retry.
    chunk_seconds: float = 300.0
    overlap_seconds: float = 2.0
    silence_search_seconds: float = 20.0
    # A remainder under this length stays on the previous chunk.
    min_tail_seconds: float = 15.0
    silence_db: float = -35.0
    min_silence_seconds: float = 0.4
    sample_rate: int = 16000
    # Longer than this, the HTTP call returns a job id instead of blocking.
    sync_max_seconds: float = 15 * 60
    max_upload_bytes: int = 500 * 1024 * 1024
    work_root: Path = field(
        default_factory=lambda: Path(tempfile.gettempdir()) / "transcription-pipeline"
    )

    def __post_init__(self) -> None:
        if self.chunk_seconds <= 0:
            raise ValueError("chunk_seconds must be positive")
        if not 0 <= self.overlap_seconds < self.chunk_seconds:
            raise ValueError("overlap_seconds must be >= 0 and smaller than chunk_seconds")
        if self.silence_search_seconds < 0 or self.min_tail_seconds < 0:
            raise ValueError("silence_search_seconds and min_tail_seconds must be >= 0")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.max_upload_bytes <= 0 or self.sync_max_seconds <= 0:
            raise ValueError("upload and sync limits must be positive")
