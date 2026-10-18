"""Verify exported public examples through the full deployed CPU pipeline."""
from pathlib import Path
import json
import time

from PIL import Image
import numpy as np

from sgsl_demo.pipeline import RecognitionPipeline


def main():
    root = Path(__file__).resolve().parents[1]
    pipeline = RecognitionPipeline(root / "models" / "alphabet_mlp.npz")
    records = []
    for path in sorted((root / "examples" / "heldout").glob("correct_*.jpg")):
        label = path.stem.split("_")[1]
        image = np.asarray(Image.open(path).convert("RGB"))
        start = time.monotonic()
        overlay, result = pipeline.predict_image(image)
        row = {"file": path.name, "true_label": label, "prediction": result.prediction.label,
               "confidence": result.prediction.confidence, "detected": result.detection_rate,
               "seconds": time.monotonic() - start}
        records.append(row)
        if label in {"C", "A", "T"}:
            Image.fromarray(overlay).save(root / "training_results" / "ws3_alphabet_v1" / f"overlay_{label}.jpg")
        print(json.dumps(row), flush=True)
    report = {"images": records, "correct": sum(row["true_label"] == row["prediction"] for row in records),
              "total": len(records)}
    video = root / "examples" / "heldout" / "heldout_CAB_montage.mp4"
    result = pipeline.predict_video(video)
    report["video"] = {"sequence": result.sequence_text, "detection_coverage": result.detection_rate,
                       "preview": result.preview_path, "timeline": result.timeline}
    output = root / "training_results" / "ws3_alphabet_v1" / "interface_check.json"
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["video"]), flush=True)


if __name__ == "__main__":
    main()
