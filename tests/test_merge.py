from transcription_pipeline.merge import ChunkTranscript, merge_chunks
from transcription_pipeline.schemas import Segment, Word


def _word(start: float, end: float, text: str) -> Word:
    return Word(start=start, end=end, text=text)


def test_silence_cuts_are_shifted_and_concatenated():
    chunks = [
        ChunkTranscript(0, 5, [Segment(start=0, end=2, text="one", words=[_word(0, 2, "one")])]),
        ChunkTranscript(5, 9, [Segment(start=0, end=1.5, text="two", words=[_word(0, 1.5, "two")])]),
    ]
    merged = merge_chunks(chunks)
    assert [(segment.start, segment.end, segment.text) for segment in merged] == [
        (0, 2, "one"),
        (5, 6.5, "two"),
    ]


def test_overlap_keeps_each_word_once():
    first = Segment(
        start=8,
        end=10,
        text="alpha beta",
        words=[_word(8, 9, "alpha"), _word(9, 10, "beta")],
    )
    second_overlap = Segment(
        start=0,
        end=2,
        text="alpha beta",
        words=[_word(0, 1, "alpha"), _word(1, 2, "beta")],
    )
    second_rest = Segment(start=2, end=4, text="gamma", words=[_word(2, 4, "gamma")])
    chunks = [
        ChunkTranscript(0, 10, [first]),
        ChunkTranscript(8, 14, [second_overlap, second_rest]),
    ]
    text = " ".join(segment.text for segment in merge_chunks(chunks))
    assert text.count("alpha") == 1
    assert text.count("beta") == 1
    assert text.count("gamma") == 1
    assert "alpha" in text and "beta" in text and "gamma" in text
