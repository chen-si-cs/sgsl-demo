"""Confidence gating and dwell-time decoding for fingerspelled letter streams."""
from __future__ import annotations

import numpy as np


def decode_consistent_letters(probabilities, valid_indices, classes, sample_times,
                              threshold=.65, window_seconds=.30, min_support=2,
                              consensus=2 / 3, min_margin=.10, blank_reset_seconds=.30):
    """Causal, timestamp-aware majority vote; a lone confidence spike cannot emit.

    Pick each frame's highest-probability class. Confirm only a majority supported
    by at least two confident frames in the trailing source-time window. Store a
    peak evidence frame for inspection, but never substitute it for consensus.
    """
    times = np.asarray(sample_times, dtype=float)
    if not len(times) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Sample times must be finite and strictly increasing")
    if not 0 <= threshold <= 1 or min_support < 2 or not .5 < consensus <= 1:
        raise ValueError("Invalid confidence/consensus parameters")
    if window_seconds <= 0 or blank_reset_seconds <= 0 or not 0 <= min_margin <= 1:
        raise ValueError("Invalid temporal window or margin")
    probabilities = np.asarray(probabilities, dtype=float)
    valid = np.asarray(valid_indices, dtype=int)
    if probabilities.shape != (len(valid), len(classes)) or len(set(valid)) != len(valid):
        raise ValueError("Scores must have one row per unique valid sample")
    if np.any(valid < 0) or np.any(valid >= len(times)) or not np.isfinite(probabilities).all():
        raise ValueError("Invalid frame probabilities or indices")
    if np.any(probabilities < 0) or (len(probabilities) and not np.allclose(probabilities.sum(axis=1), 1, atol=1e-4)):
        raise ValueError("Frame scores must be probability distributions")
    scores_by_index = dict(zip(valid.tolist(), probabilities))
    history = []
    tokens, display = [], []
    last_emitted = None
    last_seen = None
    for index, now in enumerate(times):
        if index and now - times[index - 1] > blank_reset_seconds + 1e-6:
            history.clear()
            last_emitted = None
        history = [item for item in history if now - times[item[0]] <= window_seconds - 1e-6]
        scores = scores_by_index.get(index)
        if scores is None:
            display.append(("NO_HAND", 0.0))
            history.append((index, None, None))
            if last_seen is not None and now - last_seen >= blank_reset_seconds - 1e-6:
                last_emitted = None
                history.clear()
            continue
        last_seen = now
        best = int(np.argmax(scores))
        confidence = float(scores[best])
        confident = confidence >= threshold
        display.append((classes[best] if confident else "UNCERTAIN", confidence))
        history.append((index, best if confident else None, scores))
        if not confident:
            continue
        supporters = [item for item in history if item[1] == best]
        if len(supporters) < min_support or len(supporters) / len(history) + 1e-9 < consensus:
            continue
        observed = [item[2] for item in history if item[2] is not None]
        pooled = np.mean(observed, axis=0)
        pooled_order = np.argsort(pooled)[::-1]
        if int(pooled_order[0]) != best or pooled[best] < threshold:
            continue
        margin = float(pooled[best] - pooled[pooled_order[1]]) if len(classes) > 1 else 1.0
        if margin < min_margin or classes[best] == last_emitted:
            continue
        peak = max(supporters, key=lambda item: item[2][best])
        tokens.append({"label": classes[best], "sample_index": index,
                       "time_seconds": float(now), "confidence": float(pooled[best]),
                       "support_frames": len(supporters), "window_frames": len(history),
                       "peak_sample_index": peak[0], "peak_confidence": float(peak[2][best]),
                       "runner_up_margin": margin})
        last_emitted = classes[best]
    return tokens, display


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
