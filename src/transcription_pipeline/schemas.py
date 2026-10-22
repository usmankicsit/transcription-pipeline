from pydantic import BaseModel, Field


class Word(BaseModel):
    start: float = Field(..., description="Start time in seconds on the original timeline")
    end: float
    text: str


class Segment(BaseModel):
    start: float
    end: float
    text: str
    words: list[Word] = Field(default_factory=list)


class Transcript(BaseModel):
    language: str | None = None
    duration: float
    text: str
    segments: list[Segment]
    source_format: str | None = None
    source_codec: str | None = None
