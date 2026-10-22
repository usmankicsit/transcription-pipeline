"""Stitch chunk transcripts onto one timeline.

Chunks cut on silence do not overlap, so their segments are shifted and
concatenated. A hard cut rewinds the next chunk by the overlap. The overlap
is split at its midpoint: a word belongs to the chunk where its midpoint
falls, so each word is emitted once.
"""

from transcription_pipeline.schemas import Segment, Word


class ChunkTranscript:
    def __init__(self, start: float, end: float, segments: list[Segment]) -> None:
        self.start = start
        self.end = end
        self.segments = segments


def merge_chunks(chunks: list[ChunkTranscript]) -> list[Segment]:
    if not chunks:
        return []

    merged: list[Segment] = []
    for index, chunk in enumerate(chunks):
        shifted = _shift(chunk.segments, chunk.start)
        if index == 0:
            merged.extend(shifted)
            continue
        previous_end = chunks[index - 1].end
        if chunk.start >= previous_end - 1e-3:
            merged.extend(shifted)
            continue
        boundary = (chunk.start + previous_end) / 2.0
        merged = _trim_tail(merged, boundary)
        merged.extend(_trim_head(shifted, boundary))

    cleaned = [segment for segment in merged if segment.text.strip()]
    cleaned.sort(key=lambda segment: (segment.start, segment.end))
    return cleaned


def _shift(segments: list[Segment], offset: float) -> list[Segment]:
    if offset == 0:
        return list(segments)
    shifted: list[Segment] = []
    for segment in segments:
        words = [
            Word(start=word.start + offset, end=word.end + offset, text=word.text)
            for word in segment.words
        ]
        shifted.append(
            Segment(
                start=segment.start + offset,
                end=segment.end + offset,
                text=segment.text,
                words=words,
            )
        )
    return shifted


def _words_of(segment: Segment) -> list[Word]:
    if segment.words:
        return list(segment.words)
    return [Word(start=segment.start, end=segment.end, text=segment.text)]


def _rebuild(words: list[Word]) -> Segment | None:
    cleaned = [word for word in words if word.text.strip()]
    if not cleaned:
        return None
    text = " ".join(word.text.strip() for word in cleaned)
    return Segment(start=cleaned[0].start, end=cleaned[-1].end, text=text, words=cleaned)


def _midpoint(word: Word) -> float:
    return (word.start + word.end) / 2.0


def _trim_tail(segments: list[Segment], boundary: float) -> list[Segment]:
    kept: list[Segment] = []
    for segment in segments:
        words = _words_of(segment)
        if all(_midpoint(word) < boundary for word in words):
            kept.append(segment)
            continue
        if all(_midpoint(word) >= boundary for word in words):
            continue
        rebuilt = _rebuild([word for word in words if _midpoint(word) < boundary])
        if rebuilt is not None:
            kept.append(rebuilt)
    return kept


def _trim_head(segments: list[Segment], boundary: float) -> list[Segment]:
    kept: list[Segment] = []
    for segment in segments:
        words = _words_of(segment)
        if all(_midpoint(word) >= boundary for word in words):
            kept.append(segment)
            continue
        if all(_midpoint(word) < boundary for word in words):
            continue
        rebuilt = _rebuild([word for word in words if _midpoint(word) >= boundary])
        if rebuilt is not None:
            kept.append(rebuilt)
    return kept
