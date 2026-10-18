from __future__ import annotations

import numpy as np


HAND_LANDMARKS = 21
POSE_INDICES = (0, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24)
LANDMARK_COUNT = HAND_LANDMARKS * 2 + len(POSE_INDICES)
FEATURE_CHANNELS = 4  # body-relative x, y, z, and a presence mask
TARGET_FRAMES = 24


def validate_sequence(sequence: np.ndarray) -> np.ndarray:
    array = np.asarray(sequence, dtype=np.float32)
    expected = (LANDMARK_COUNT, FEATURE_CHANNELS)
    if array.ndim != 3 or array.shape[1:] != expected:
        raise ValueError(f"Expected [frames, {expected[0]}, {expected[1]}], got {array.shape}")
    if len(array) == 0:
        raise ValueError("A landmark sequence must contain at least one frame")
    if not np.isfinite(array).all():
        raise ValueError("Landmark sequence contains NaN or infinity")
    return array


def resample_sequence(sequence: np.ndarray, target_frames: int = TARGET_FRAMES) -> np.ndarray:
    sequence = validate_sequence(sequence)
    if target_frames < 2:
        raise ValueError("target_frames must be at least 2")
    if len(sequence) == target_frames:
        return sequence.copy()
    old = np.linspace(0.0, 1.0, len(sequence), dtype=np.float32)
    new = np.linspace(0.0, 1.0, target_frames, dtype=np.float32)
    flat = sequence.reshape(len(sequence), -1)
    sampled = np.stack([np.interp(new, old, flat[:, index]) for index in range(flat.shape[1])], axis=1)
    return sampled.reshape(target_frames, LANDMARK_COUNT, FEATURE_CHANNELS).astype(np.float32)


def temporal_descriptor(sequence: np.ndarray, target_frames: int = TARGET_FRAMES) -> np.ndarray:
    """Encode shape, body location, and motion in a small fixed-size vector."""
    sampled = resample_sequence(sequence, target_frames)
    coordinates = sampled[..., :3]
    mask = sampled[..., 3:4]
    velocity = np.diff(coordinates, axis=0, prepend=coordinates[:1]) * mask
    acceleration = np.diff(velocity, axis=0, prepend=velocity[:1]) * mask
    descriptor = np.concatenate(
        [sampled.reshape(-1), velocity.reshape(-1), acceleration.reshape(-1)],
        axis=0,
    )
    return descriptor.astype(np.float32)


def visible_fraction(sequence: np.ndarray) -> float:
    sequence = validate_sequence(sequence)
    hand_mask = sequence[:, : HAND_LANDMARKS * 2, 3]
    return float(np.mean(hand_mask > 0.5))

