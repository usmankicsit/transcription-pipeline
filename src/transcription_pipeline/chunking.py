"""Turn a duration and a silence map into chunk spans.

Whisper-style models already score audio in short windows. Chunking here is
about memory, retries, and not cutting a word in half. A pause near the
target boundary wins. If that region has no pause, the cut is hard and the
next chunk rewinds by `overlap` so the merge step can give the boundary to
exactly one chunk.
"""

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Span:
    index: int
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


def plan_spans(
    duration: float,
    silences: Sequence[tuple[float, float]],
    *,
    target: float,
    overlap: float,
    search: float,
    min_tail: float = 0.0,
) -> list[Span]:
    if target <= 0:
        raise ValueError("target must be positive")
    if overlap < 0 or overlap >= target:
        raise ValueError("overlap must be >= 0 and smaller than the chunk length")
    if search < 0 or min_tail < 0:
        raise ValueError("search and min_tail must be >= 0")
    if duration <= 0:
        return []

    raw: list[tuple[float, float]] = []
    cursor = 0.0
    for _ in range(100_000):
        if cursor >= duration:
            break
        remaining = duration - cursor
        if remaining <= target + min_tail:
            raw.append((cursor, duration))
            break

        ideal = cursor + target
        window_lo = max(cursor + (target * 0.5), ideal - search)
        window_hi = min(duration, ideal + search)
        cut = _silence_cut(silences, ideal, window_lo, window_hi)
        if cut is not None and duration - cut <= min_tail:
            raw.append((cursor, duration))
            break
        if cut is None:
            raw.append((cursor, ideal))
            nxt = ideal - overlap
            if nxt <= cursor:
                raise ValueError("overlap is too large to advance through the file")
            cursor = nxt
            continue
        raw.append((cursor, cut))
        cursor = cut
    else:
        raise ValueError("could not plan chunks")

    return [Span(index, start, end) for index, (start, end) in enumerate(raw)]


def _silence_cut(
    silences: Sequence[tuple[float, float]],
    ideal: float,
    window_lo: float,
    window_hi: float,
) -> float | None:
    """Point inside a silence, inside the window, closest to `ideal`."""
    if window_hi <= window_lo:
        return None
    best: float | None = None
    best_dist: float | None = None
    for start, end in silences:
        if end < start:
            continue
        lo = max(start, window_lo)
        hi = min(end, window_hi)
        if hi <= lo:
            continue
        if ideal < lo:
            candidate = lo
        elif ideal > hi:
            candidate = hi
        else:
            candidate = ideal
        dist = abs(candidate - ideal)
        if best_dist is None or dist < best_dist:
            best = candidate
            best_dist = dist
    return best
