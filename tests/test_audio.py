from transcription_pipeline.audio import parse_silences


def test_parse_pairs_silence_regions():
    log = "\n".join(
        [
            "silence_start: 1.5",
            "silence_end: 2.25 | silence_duration: 0.75",
            "silence_start: 8",
            "silence_end: 8.4",
        ]
    )
    assert parse_silences(log) == [(1.5, 2.25), (8.0, 8.4)]


def test_unclosed_silence_is_closed_at_the_file_end():
    assert parse_silences("silence_start: 4.0", duration=5.5) == [(4.0, 5.5)]
