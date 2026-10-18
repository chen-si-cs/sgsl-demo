from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2
import numpy as np

from sgsl_demo.classifier import TinySequenceClassifier
from sgsl_demo.features import HAND_LANDMARKS
from sgsl_demo.landmarks import MediaPipeLandmarkExtractor


LABEL_PATTERN = re.compile(r"^([A-Z])-")


def augment(frame: np.ndarray, rng: np.random.Generator, count: int) -> list[np.ndarray]:
    samples: list[np.ndarray] = [np.repeat(frame[None, ...], 24, axis=0).astype(np.float32)]
    for _ in range(count):
        length = int(rng.integers(12, 33))
        sequence = np.repeat(frame[None, ...], length, axis=0)
        mask = sequence[..., 3:4] > 0.0
        coordinate_noise = rng.normal(0.0, 0.018, size=sequence[..., :3].shape).astype(np.float32)
        sequence[..., :3] += coordinate_noise * mask
        sequence[..., :3] *= float(rng.uniform(0.94, 1.06))
        drift = np.linspace(0.0, float(rng.normal(0.0, 0.035)), length, dtype=np.float32)
        sequence[..., 0] += drift[:, None] * sequence[..., 3]
        sequence[..., 1] += drift[::-1, None] * 0.35 * sequence[..., 3]
        if rng.random() < 0.25:
            drop_index = int(rng.integers(0, length))
            sequence[drop_index, : HAND_LANDMARKS * 2] = 0.0
        samples.append(sequence.astype(np.float32))
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the bundled overfit alphabet smoke model")
    parser.add_argument("reference_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--augmentations", type=int, default=20)
    args = parser.parse_args()

    image_dir = args.reference_root / "output" / "yolov8_output" / "predict_valid"
    paths = sorted(image_dir.glob("*.jpg"))
    sequences: list[np.ndarray] = []
    labels: list[str] = []
    skipped: list[str] = []
    rng = np.random.default_rng(17)
    with MediaPipeLandmarkExtractor() as extractor:
        for path in paths:
            match = LABEL_PATTERN.match(path.name)
            if not match:
                continue
            image = cv2.imread(str(path))
            if image is None:
                skipped.append(path.name)
                continue
            extracted = extractor.extract_frame(image)
            if extracted.hands_found == 0:
                skipped.append(path.name)
                continue
            generated = augment(extracted.landmarks, rng, args.augmentations)
            sequences.extend(generated)
            labels.extend([match.group(1)] * len(generated))

    metadata = {
        "demo_only": True,
        "task": "static fingerspelling smoke test",
        "source": "GloriaNeo/SgSL-Sign-Language-Recognition validation images",
        "license": "MIT repository; see THIRD_PARTY_NOTICES.md",
        "classes": sorted(set(labels)),
        "training_sequences": len(sequences),
        "skipped_images": skipped,
        "limitations": "Overfit technical checkpoint; G and J absent; not signer-independent SgSL evidence.",
    }
    model = TinySequenceClassifier.fit(
        sequences,
        labels,
        epochs=650,
        learning_rate=0.18,
        l2=2e-4,
        metadata=metadata,
    )
    model.save(args.output)
    predictions = [model.predict(sequence, threshold=0.0).label for sequence in sequences]
    accuracy = float(np.mean(np.asarray(predictions) == np.asarray(labels)))
    print(f"saved={args.output}")
    print(f"classes={','.join(model.classes)}")
    print(f"augmented_training_accuracy={accuracy:.4f}")
    print(f"skipped={len(skipped)}")


if __name__ == "__main__":
    main()

