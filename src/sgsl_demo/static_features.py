"""Compact hand geometry shared by GPU training and NumPy CPU inference."""
from __future__ import annotations

import numpy as np

from .features import validate_sequence

PAIRS = np.triu_indices(21, 1)
FINGER_CHAINS = ((0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (0, 9, 10, 11, 12),
                 (0, 13, 14, 15, 16), (0, 17, 18, 19, 20))
STATIC_FEATURE_DIM = 63 + 210 + 15


def hand_features(sequence: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sequence = validate_sequence(sequence)
    rows = []
    indices = []
    for index, frame in enumerate(sequence):
        left_score = float(frame[:21, 3].mean())
        right_score = float(frame[21:42, 3].mean())
        if max(left_score, right_score) < 0.25:
            continue
        offset = 0 if left_score >= right_score else 21
        coordinates = frame[offset:offset + 21, :3].copy()
        coordinates -= coordinates[:1]
        palm = float(np.linalg.norm(coordinates[9, :2]))
        if palm < 1e-5:
            continue
        coordinates /= palm
        # Canonicalise anatomical hand side while retaining signing orientation.
        if offset == 0:
            coordinates[:, 0] *= -1.0
        distances = np.linalg.norm(coordinates[:, None] - coordinates[None, :], axis=-1)[PAIRS]
        angles = []
        for chain in FINGER_CHAINS:
            points = coordinates[list(chain)]
            bones = np.diff(points, axis=0)
            bones /= np.maximum(np.linalg.norm(bones, axis=1, keepdims=True), 1e-6)
            angles.extend(np.sum(bones[:-1] * bones[1:], axis=1).tolist())
        row = np.concatenate([coordinates.ravel(), distances, angles]).astype(np.float32)
        if np.isfinite(row).all():
            rows.append(row)
            indices.append(index)
    if not rows:
        return np.zeros((0, STATIC_FEATURE_DIM), dtype=np.float32), np.asarray([], dtype=int)
    return np.stack(rows), np.asarray(indices, dtype=int)
