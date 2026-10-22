class PipelineError(Exception):
    """A failure the caller can show to a user."""


class AudioError(PipelineError):
    """The file could not be probed, decoded, or sliced."""


class TranscriptionError(PipelineError):
    """The speech engine failed on a chunk."""
