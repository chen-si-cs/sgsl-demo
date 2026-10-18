from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .features import temporal_descriptor


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float
    probabilities: dict[str, float]
    accepted: bool


class TinySequenceClassifier:
    """A standardized linear softmax classifier for short landmark sequences."""

    def __init__(
        self,
        classes: Sequence[str],
        weights: np.ndarray,
        bias: np.ndarray,
        mean: np.ndarray,
        scale: np.ndarray,
        metadata: dict | None = None,
    ) -> None:
        self.classes = tuple(str(label) for label in classes)
        self.weights = np.asarray(weights, dtype=np.float32)
        self.bias = np.asarray(bias, dtype=np.float32)
        self.mean = np.asarray(mean, dtype=np.float32)
        self.scale = np.asarray(scale, dtype=np.float32)
        self.metadata = dict(metadata or {})

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        exp = np.exp(shifted)
        return exp / np.sum(exp, axis=-1, keepdims=True)

    @classmethod
    def fit(
        cls,
        sequences: Iterable[np.ndarray],
        labels: Sequence[str],
        *,
        epochs: int = 500,
        learning_rate: float = 0.25,
        l2: float = 1e-4,
        seed: int = 7,
        metadata: dict | None = None,
    ) -> "TinySequenceClassifier":
        descriptors = np.stack([temporal_descriptor(sequence) for sequence in sequences])
        class_names = tuple(sorted(set(labels)))
        if len(class_names) < 2:
            raise ValueError("Training requires at least two labels")
        lookup = {label: index for index, label in enumerate(class_names)}
        targets = np.asarray([lookup[label] for label in labels], dtype=np.int64)
        mean = descriptors.mean(axis=0)
        scale = descriptors.std(axis=0)
        scale[scale < 1e-5] = 1.0
        x = np.clip((descriptors - mean) / scale, -8.0, 8.0).astype(np.float32)

        rng = np.random.default_rng(seed)
        weights = rng.normal(0.0, 0.002, size=(x.shape[1], len(class_names))).astype(np.float32)
        bias = np.zeros(len(class_names), dtype=np.float32)
        one_hot = np.eye(len(class_names), dtype=np.float32)[targets]
        for epoch in range(epochs):
            probabilities = cls._softmax(x @ weights + bias)
            error = (probabilities - one_hot) / len(x)
            rate = learning_rate * (0.2 + 0.8 * (1.0 - epoch / max(epochs, 1)))
            weights -= rate * (x.T @ error + l2 * weights)
            bias -= rate * error.sum(axis=0)

        return cls(class_names, weights, bias, mean, scale, metadata)

    def predict(self, sequence: np.ndarray, threshold: float = 0.65) -> Prediction:
        descriptor = temporal_descriptor(sequence)
        x = np.clip((descriptor - self.mean) / self.scale, -8.0, 8.0)
        probabilities = self._softmax(x @ self.weights + self.bias)
        index = int(np.argmax(probabilities))
        confidence = float(probabilities[index])
        scores = {
            label: float(score)
            for label, score in sorted(
                zip(self.classes, probabilities), key=lambda item: item[1], reverse=True
            )
        }
        return Prediction(
            label=self.classes[index] if confidence >= threshold else "UNCERTAIN",
            confidence=confidence,
            probabilities=scores,
            accepted=confidence >= threshold,
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            classes=np.asarray(self.classes),
            weights=self.weights,
            bias=self.bias,
            mean=self.mean,
            scale=self.scale,
            metadata=np.asarray(json.dumps(self.metadata, sort_keys=True)),
        )

    @classmethod
    def load(cls, path: str | Path) -> "TinySequenceClassifier":
        with np.load(Path(path), allow_pickle=False) as data:
            if "kind" in data and str(data["kind"].item()) == "landmark_mlp_v1":
                from .mlp_classifier import LandmarkMLPClassifier

                return LandmarkMLPClassifier(path)
            metadata = json.loads(str(data["metadata"].item()))
            return cls(
                classes=data["classes"].tolist(),
                weights=data["weights"],
                bias=data["bias"],
                mean=data["mean"],
                scale=data["scale"],
                metadata=metadata,
            )
