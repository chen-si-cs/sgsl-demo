"""Cache every source frame's landmarks/scores for a label-independent audit."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from sgsl_demo.classifier import TinySequenceClassifier
from sgsl_demo.landmarks import MediaPipeLandmarkExtractor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(args.video))
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    with MediaPipeLandmarkExtractor() as extractor:
        extracted = extractor.extract_video(args.video, max_frames=count)
    model = TinySequenceClassifier.load(root / "models" / "alphabet_mlp.npz")
    probabilities, valid = model.frame_probabilities(extracted.landmarks)
    dense = np.zeros((len(extracted.landmarks), len(model.classes)), np.float32)
    dense[valid] = probabilities
    np.savez_compressed(args.output / "dense_frame_cache.npz", landmarks=extracted.landmarks,
                        probabilities=dense, valid_indices=valid, classes=np.asarray(model.classes),
                        source_frame_indices=extracted.sampled_frame_indices, fps=extracted.input_fps)
    rows = []
    for index, scores in enumerate(dense):
        order = np.argsort(scores)[::-1][:5]
        rows.append({"frame": extracted.sampled_frame_indices[index],
                     "time_seconds": extracted.sampled_frame_indices[index] / extracted.input_fps,
                     "top5": {model.classes[i]: float(scores[i]) for i in order} if scores.sum() else {}})
    report = {"source_sha256": hashlib.sha256(args.video.read_bytes()).hexdigest(),
              "fps": extracted.input_fps, "frames": count, "rows": rows}
    (args.output / "dense_frame_scores.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    previous = None
    for row in rows:
        label = next(iter(row["top5"]), "NO_HAND")
        if label != previous:
            print(f"{row['time_seconds']:4.2f}s {row['top5'] or 'NO_HAND'}", flush=True)
            previous = label


if __name__ == "__main__":
    main()
