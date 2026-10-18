"""Record current predictions without using them to change the random pack."""
import json
from pathlib import Path

import numpy as np
from PIL import Image

from sgsl_demo.pipeline import RecognitionPipeline


def main():
    root = Path(__file__).resolve().parents[1]
    folder = root / "examples" / "random_demo"
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    pipeline = RecognitionPipeline(root / "models" / "alphabet_mlp.npz")
    images, videos = [], []
    for entry in manifest["images"]:
        _, result = pipeline.predict_image(np.asarray(Image.open(folder / entry["file"]).convert("RGB")))
        images.append({"file": entry["file"], "expected": entry["label"],
                       "prediction": result.prediction.label, "confidence": result.prediction.confidence})
    for entry in manifest["videos"]:
        result = pipeline.predict_video(folder / entry["file"])
        videos.append({"file": entry["file"], "expected": entry["expected"],
                       "sequence": result.sequence_text, "coverage": result.detection_rate})
        print(json.dumps(videos[-1]), flush=True)
    report = {"selection_changed_after_prediction": False, "images": images, "videos": videos}
    output = root / "training_results" / "ws3_alphabet_v1" / "random_pack_check.json"
    output.write_text(json.dumps(report, indent=2))
    print(f"Image matches: {sum(row['expected'] == row['prediction'] for row in images)}/{len(images)}", flush=True)


if __name__ == "__main__":
    main()
