from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .classifier import Prediction
from .static_features import hand_features


class LandmarkMLPClassifier:
    """GPU-trained small MLP, exported as arrays for CPU-only serving."""

    def __init__(self, path: str | Path):
        with np.load(path, allow_pickle=False) as data:
            self.classes = tuple(data["classes"].tolist())
            self.mean = data["mean"].copy()
            self.scale = data["scale"].copy()
            self.weights = [data[f"weight_{i}"].copy() for i in range(3)]
            self.biases = [data[f"bias_{i}"].copy() for i in range(3)]
            self.temperature = float(data["temperature"])
            self.metadata = json.loads(str(data["metadata"].item()))

    def frame_probabilities(self, sequence: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        features, indices = hand_features(sequence)
        if not len(features):
            return np.zeros((0, len(self.classes)), dtype=np.float32), indices
        x = np.clip((features - self.mean) / self.scale, -8, 8)
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            x = x @ weight.T + bias
            if index < len(self.weights) - 1:
                x = np.maximum(x, 0)
        x /= self.temperature
        x -= np.max(x, axis=1, keepdims=True)
        probabilities = np.exp(x)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        return probabilities, indices

    def predict(self, sequence: np.ndarray, threshold: float = 0.65) -> Prediction:
        probabilities, _ = self.frame_probabilities(sequence)
        if not len(probabilities):
            return Prediction("NO_HAND", 0.0, {}, False)
        # Average evidence over frames; stable held letters survive frame jitter.
        pooled = probabilities.mean(axis=0)
        best = int(np.argmax(pooled))
        confidence = float(pooled[best])
        scores = {self.classes[i]: float(pooled[i]) for i in np.argsort(-pooled)}
        accepted = confidence >= threshold
        return Prediction(self.classes[best] if accepted else "UNCERTAIN", confidence, scores, accepted)
