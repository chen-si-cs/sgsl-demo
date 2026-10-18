import numpy as np

from sgsl_demo.decoder import decode_letters


def test_single_frame_flicker_does_not_emit_a_second_letter():
    probabilities = np.asarray([[.99, .01]] * 4 + [[.01, .99]] + [[.99, .01]] * 4)
    tokens, _ = decode_letters(probabilities, np.arange(9), ("A", "B"), 9)
    assert [token["label"] for token in tokens] == ["A"]


def test_missing_hand_allows_a_repeated_letter():
    probabilities = np.asarray([[.99, .01]] * 6)
    tokens, _ = decode_letters(probabilities, np.asarray([0, 1, 2, 5, 6, 7]), ("A", "B"), 8)
    assert [token["label"] for token in tokens] == ["A", "A"]
