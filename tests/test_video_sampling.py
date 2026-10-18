import numpy as np
import pytest

from sgsl_demo.landmarks import video_sample_indices


def test_exact_10hz_six_second_grid():
    indices = video_sample_indices(180, 30, sample_hz=10)
    assert indices == list(range(0, 180, 3))
    assert len(indices) == 60


def test_grid_does_not_silently_degrade_on_long_clips():
    with pytest.raises(ValueError, match="shorter clip"):
        video_sample_indices(3000, 30, sample_hz=10, max_frames=600)


def test_non_integer_fps_and_low_fps_are_quantised_without_duplicates():
    indices = video_sample_indices(180, 29.97, sample_hz=10)
    assert len(indices) == len(set(indices))
    times = np.arange(len(indices)) / 10
    assert np.max(np.abs(np.asarray(indices) / 29.97 - times)) <= .5 / 29.97
    assert video_sample_indices(30, 5, sample_hz=10) == list(range(30))
