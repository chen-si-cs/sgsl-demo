import numpy as np

from sgsl_demo.features import FEATURE_CHANNELS, LANDMARK_COUNT, resample_sequence, temporal_descriptor


def test_resample_and_descriptor_are_fixed_size():
    sequence = np.zeros((7, LANDMARK_COUNT, FEATURE_CHANNELS), dtype=np.float32)
    sequence[:, 0, 0] = np.linspace(-1, 1, 7)
    sequence[:, 0, 3] = 1.0
    sampled = resample_sequence(sequence)
    descriptor = temporal_descriptor(sequence)
    assert sampled.shape == (24, LANDMARK_COUNT, FEATURE_CHANNELS)
    assert descriptor.shape == (24 * LANDMARK_COUNT * (FEATURE_CHANNELS + 3 + 3),)
    assert np.isfinite(descriptor).all()

