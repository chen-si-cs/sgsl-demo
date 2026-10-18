"""Confidence gating and dwell-time decoding for fingerspelled letter streams."""
from __future__ import annotations

import numpy as np


def decode_letters(probabilities: np.ndarray, valid_indices: np.ndarray, classes,
                   total_frames: int, threshold: float = 0.65, dwell_frames: int = 3):
    frame_scores = {int(index): row for index, row in zip(valid_indices, probabilities)}
    smoothed = None
    candidate = None
    dwell = 0
    last_emitted = None
    missing = 0
    tokens = []
    display = []
    for index in range(total_frames):
        if index not in frame_scores:
            missing += 1
            if missing >= 2:
                smoothed = None
                candidate = None
                dwell = 0
                last_emitted = None
            display.append(("NO_HAND", 0.0))
            continue
        missing = 0
        scores = frame_scores[index]
        smoothed = scores.copy() if smoothed is None else 0.65 * scores + 0.35 * smoothed
        best = int(np.argmax(smoothed))
        confidence = float(smoothed[best])
        label = classes[best] if confidence >= threshold else "UNCERTAIN"
        display.append((label, confidence))
        if label == "UNCERTAIN":
            candidate = None
            dwell = 0
            continue
        if label == candidate:
            dwell += 1
        else:
            candidate = label
            dwell = 1
        if dwell >= dwell_frames and label != last_emitted:
            tokens.append({"label": label, "sample_index": index, "confidence": confidence})
            last_emitted = label
    return tokens, display
