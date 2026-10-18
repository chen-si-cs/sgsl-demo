from pathlib import Path

import cv2
import numpy as np

from sgsl_demo.pipeline import RecognitionPipeline


ROOT = Path(__file__).resolve().parents[1]


def test_bundled_image_and_video_examples():
    pipeline = RecognitionPipeline(ROOT / "models" / "demo_alphabet_classifier.npz")
    encoded = np.fromfile(ROOT / "examples" / "example_C.jpg", dtype=np.uint8)
    bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    _, image_result = pipeline.predict_image(rgb, threshold=0.65)
    assert image_result.prediction.label == "C"
    video_result = pipeline.predict_video(ROOT / "examples" / "example_C.mp4", threshold=0.65)
    assert video_result.prediction.label == "C"
    assert video_result.detection_rate > 0.9
    assert Path(video_result.preview_path).exists()
