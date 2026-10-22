from transcription_pipeline.export import format_timestamp, render
from transcription_pipeline.schemas import Segment, Transcript, Word


def _transcript() -> Transcript:
    return Transcript(
        language="en",
        duration=2.5,
        text="hello there",
        segments=[
            Segment(
                start=0,
                end=1.25,
                text="hello there",
                words=[Word(start=0, end=0.5, text="hello"), Word(start=0.5, end=1.25, text="there")],
            )
        ],
        source_format="wav",
        source_codec="pcm_s16le",
    )


def test_timestamp_rolls_hours():
    assert format_timestamp(3661.007) == "01:01:01,007"
    assert format_timestamp(3661.007, ".") == "01:01:01.007"


def test_json_contains_segment_times():
    body = render(_transcript(), "json")
    assert '"start": 0.0' in body
    assert '"end": 1.25' in body
    assert "hello there" in body


def test_srt_and_vtt_cues():
    srt = render(_transcript(), "srt")
    assert "00:00:00,000 --> 00:00:01,250" in srt
    vtt = render(_transcript(), "vtt")
    assert vtt.startswith("WEBVTT")
    assert "00:00:00.000 --> 00:00:01.250" in vtt


def test_plain_text_is_the_words_only():
    assert render(_transcript(), "txt") == "hello there\n"
