from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from sgsl_demo.classifier import TinySequenceClassifier


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a tiny temporal classifier from landmark files")
    parser.add_argument("landmarks", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--epochs", type=int, default=600)
    args = parser.parse_args()
    paths = sorted(args.landmarks.glob("*/*.npz"))
    if not paths:
        raise SystemExit(f"No <label>/*.npz files found under {args.landmarks}")
    sequences: list[np.ndarray] = []
    labels: list[str] = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            sequences.append(data["landmarks"].astype(np.float32))
        labels.append(path.parent.name)
    counts = {label: labels.count(label) for label in sorted(set(labels))}
    if min(counts.values()) < 2:
        raise SystemExit(f"Each class needs at least two clips; counts={counts}")
    model = TinySequenceClassifier.fit(
        sequences,
        labels,
        epochs=args.epochs,
        metadata={
            "demo_only": False,
            "task": "user-provided isolated-sign clips",
            "class_counts": counts,
            "limitations": "Validate with signer-held-out data before making performance claims.",
        },
    )
    model.save(args.output)
    predictions = [model.predict(sequence, threshold=0.0).label for sequence in sequences]
    print(f"saved={args.output}")
    print(f"training_accuracy={np.mean(np.asarray(predictions) == np.asarray(labels)):.4f}")
    print(f"counts={counts}")


if __name__ == "__main__":
    main()

