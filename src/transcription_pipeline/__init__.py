"""Audio in, timestamped transcript out."""

from transcription_pipeline.pipeline import Pipeline
from transcription_pipeline.schemas import Segment, Transcript, Word

__all__ = ["Pipeline", "Segment", "Transcript", "Word"]
__version__ = "1.0.0"
