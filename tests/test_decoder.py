import numpy as np

from sgsl_demo.decoder import decode_letters, decode_consistent_letters


def test_single_frame_flicker_does_not_emit_a_second_letter():
    probabilities = np.asarray([[.99, .01]] * 4 + [[.01, .99]] + [[.99, .01]] * 4)
    tokens, _ = decode_letters(probabilities, np.arange(9), ("A", "B"), 9)
    assert [token["label"] for token in tokens] == ["A"]


def test_missing_hand_allows_a_repeated_letter():
    probabilities = np.asarray([[.99, .01]] * 6)
    tokens, _ = decode_letters(probabilities, np.asarray([0, 1, 2, 5, 6, 7]), ("A", "B"), 8)
    assert [token["label"] for token in tokens] == ["A", "A"]


def test_consensus_rejects_one_high_confidence_spike():
    scores = np.asarray([[.99, .01]] * 4 + [[.001, .999]] + [[.99, .01]] * 4)
    tokens, _ = decode_consistent_letters(scores, np.arange(9), ("A", "B"), np.arange(9) / 10)
    assert [token["label"] for token in tokens] == ["A"]


def test_two_strong_frames_can_recover_a_brief_new_letter():
    scores = np.asarray([[.99, .01]] * 3 + [[.01, .99]] * 2)
    tokens, _ = decode_consistent_letters(scores, np.arange(5), ("A", "B"), np.arange(5) / 10)
    assert [token["label"] for token in tokens] == ["A", "B"]
    assert tokens[-1]["sample_index"] == 4
    assert tokens[-1]["support_frames"] == 2
    assert tokens[-1]["peak_sample_index"] <= tokens[-1]["sample_index"]


def test_confirmation_is_causal_prefix_invariant():
    scores = np.asarray([[.99, .01]] * 3 + [[.01, .99]] * 3)
    times = np.arange(6) / 10
    full, _ = decode_consistent_letters(scores, np.arange(6), ("A", "B"), times)
    for length in range(1, 7):
        prefix, _ = decode_consistent_letters(scores[:length], np.arange(length), ("A", "B"), times[:length])
        assert prefix == [token for token in full if token["sample_index"] < length]


def test_missing_hand_gap_resets_repeat_suppression():
    valid = np.asarray([0, 1, 2, 6, 7, 8])
    tokens, _ = decode_consistent_letters(np.asarray([[.99, .01]] * 6), valid, ("A", "B"), np.arange(9) / 10)
    assert [token["label"] for token in tokens] == ["A", "A"]


def test_consistency_does_not_override_a_confident_wrong_class():
    tokens, _ = decode_consistent_letters(np.asarray([[.99, .01]] * 6), np.arange(6), ("N", "P"), np.arange(6) / 10)
    assert [token["label"] for token in tokens] == ["N"]
