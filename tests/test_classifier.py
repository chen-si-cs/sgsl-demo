import numpy as np

from sgsl_demo.classifier import TinySequenceClassifier
from sgsl_demo.features import FEATURE_CHANNELS, LANDMARK_COUNT


def _sequence(value: float) -> np.ndarray:
    sequence = np.zeros((8, LANDMARK_COUNT, FEATURE_CHANNELS), dtype=np.float32)
    sequence[:, 0, 0] = value
    sequence[:, 0, 3] = 1.0
    return sequence


def test_classifier_learns_and_round_trips(tmp_path):
    sequences = [_sequence(-1.0), _sequence(-0.9), _sequence(1.0), _sequence(0.9)]
    labels = ["LEFT", "LEFT", "RIGHT", "RIGHT"]
    model = TinySequenceClassifier.fit(sequences, labels, epochs=200)
    assert model.predict(_sequence(-0.95), threshold=0.0).label == "LEFT"
    assert model.predict(_sequence(0.95), threshold=0.0).label == "RIGHT"
    path = tmp_path / "model.npz"
    model.save(path)
    loaded = TinySequenceClassifier.load(path)
    assert loaded.predict(_sequence(0.95), threshold=0.0).label == "RIGHT"

