import pytest

from transcription_pipeline.chunking import plan_spans


def _covers(spans, duration: float, step: float = 0.05) -> None:
    point = 0.0
    while point < duration:
        assert any(span.start <= point <= span.end for span in spans), point
        point += step
    assert spans[0].start == pytest.approx(0.0)
    assert spans[-1].end == pytest.approx(duration)


def test_short_file_is_one_chunk():
    spans = plan_spans(12.0, [], target=300, overlap=2, search=20)
    assert [(span.start, span.end) for span in spans] == [(0.0, 12.0)]


def test_zero_duration_has_no_chunks():
    assert plan_spans(0.0, [], target=300, overlap=2, search=20) == []


def test_hard_cuts_overlap_and_cover_the_file():
    spans = plan_spans(10.0, [], target=4, overlap=1, search=0.5, min_tail=0)
    assert [(span.start, span.end) for span in spans] == [(0.0, 4.0), (3.0, 7.0), (6.0, 10.0)]
    _covers(spans, 10.0)


def test_silence_cut_lands_inside_the_pause():
    silences = [(299.2, 300.8), (599.0, 601.0), (899.5, 900.2)]
    spans = plan_spans(1000.0, silences, target=300, overlap=2, search=20, min_tail=0)
    assert [(span.start, span.end) for span in spans] == [
        (0.0, 300.0),
        (300.0, 600.0),
        (600.0, 900.0),
        (900.0, 1000.0),
    ]
    for left, right in zip(spans, spans[1:]):
        assert right.start == pytest.approx(left.end)


def test_long_silence_is_cut_at_the_target_not_its_middle():
    spans = plan_spans(1000.0, [(250.0, 700.0)], target=300, overlap=2, search=20, min_tail=0)
    assert spans[0].end == pytest.approx(300.0)


def test_closer_pause_wins():
    silences = [(291.8, 292.2), (308.8, 309.2)]
    spans = plan_spans(1000.0, silences, target=300, overlap=2, search=20, min_tail=0)
    assert spans[0].end == pytest.approx(292.2)


def test_small_remainder_stays_on_the_last_chunk():
    spans = plan_spans(10.0, [], target=8, overlap=1, search=1, min_tail=5)
    assert len(spans) == 1
    assert spans[0].end == pytest.approx(10.0)


def test_overlap_must_be_smaller_than_the_chunk():
    with pytest.raises(ValueError):
        plan_spans(100.0, [], target=10, overlap=10, search=1)
