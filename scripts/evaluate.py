from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sgsl_demo.classifier import TinySequenceClassifier


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on labeled landmark files")
    parser.add_argument("model", type=Path)
    parser.add_argument("landmarks", type=Path)
    parser.add_argument("--threshold", type=float, default=0.65)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    paths = sorted(args.landmarks.glob("*/*.npz"))
    if not paths:
        raise SystemExit(f"No <label>/*.npz files found under {args.landmarks}")
    model = TinySequenceClassifier.load(args.model)
    records = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            sequence = data["landmarks"].astype(np.float32)
        prediction = model.predict(sequence, threshold=args.threshold)
        records.append((path.parent.name, prediction.label, prediction.confidence))

    classes = sorted(set(actual for actual, _, _ in records))
    accepted = [(actual, predicted) for actual, predicted, _ in records if predicted != "UNCERTAIN"]
    coverage = len(accepted) / len(records)
    accuracy = sum(actual == predicted for actual, predicted, _ in records) / len(records)
    recalls = {}
    f1_scores = {}
    confusion = {actual: {predicted: 0 for predicted in classes + ["UNCERTAIN"]} for actual in classes}
    for actual, predicted, _ in records:
        confusion[actual].setdefault(predicted, 0)
        confusion[actual][predicted] += 1
    for label in classes:
        true_positive = sum(a == label and p == label for a, p, _ in records)
        false_positive = sum(a != label and p == label for a, p, _ in records)
        false_negative = sum(a == label and p != label for a, p, _ in records)
        recall = true_positive / max(1, true_positive + false_negative)
        precision = true_positive / max(1, true_positive + false_positive)
        recalls[label] = recall
        f1_scores[label] = 2 * precision * recall / max(1e-12, precision + recall)
    report = {
        "samples": len(records),
        "threshold": args.threshold,
        "accuracy_with_abstentions_as_errors": accuracy,
        "coverage": coverage,
        "macro_f1": float(np.mean(list(f1_scores.values()))),
        "per_class_recall": recalls,
        "confusion": confusion,
        "warning": "These metrics are meaningful only for a genuinely held-out, preferably signer-held-out set.",
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
