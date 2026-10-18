from __future__ import annotations

import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .classifier import Prediction, TinySequenceClassifier
from .features import visible_fraction
from .decoder import decode_letters
from .landmarks import MediaPipeLandmarkExtractor, save_landmark_preview


@dataclass(frozen=True)
class InferenceResult:
    prediction: Prediction
    detection_rate: float
    preview_path: str | None
    note: str
    sequence_text: str = ""
    timeline: tuple = ()


class RecognitionPipeline:
    def __init__(
        self,
        classifier_path: str | Path,
        model_dir: str | Path | None = None,
    ) -> None:
        self.classifier_path = Path(classifier_path)
        self.classifier = TinySequenceClassifier.load(self.classifier_path)
        self.model_dir = model_dir

    def _predict_sequence(self, sequence: np.ndarray, threshold: float) -> Prediction:
        if visible_fraction(sequence) < 0.02:
            return Prediction("NO_HAND", 0.0, {}, False)
        return self.classifier.predict(sequence, threshold=threshold)

    def predict_image(self, image: np.ndarray, threshold: float = 0.65) -> tuple[np.ndarray, InferenceResult]:
        if image is None:
            raise ValueError("Please upload an image")
        bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        with MediaPipeLandmarkExtractor(self.model_dir) as extractor:
            extracted = extractor.extract_frame(bgr)
        sequence = extracted.landmarks[None, ...]
        prediction = self._predict_sequence(sequence, threshold)
        overlay = cv2.cvtColor(extracted.overlay_bgr, cv2.COLOR_BGR2RGB)
        note = self._model_note()
        return overlay, InferenceResult(
            prediction=prediction,
            detection_rate=float(extracted.hands_found > 0),
            preview_path=None,
            note=note,
        )

    def predict_video(self, video_path: str | Path, threshold: float = 0.65) -> InferenceResult:
        if not video_path:
            raise ValueError("Please upload or record a video")
        with MediaPipeLandmarkExtractor(self.model_dir) as extractor:
            extracted = extractor.extract_video(video_path)
        prediction = self._predict_sequence(extracted.landmarks, threshold)
        tokens = []
        if hasattr(self.classifier, "frame_probabilities"):
            probabilities, indices = self.classifier.frame_probabilities(extracted.landmarks)
            tokens, display = decode_letters(probabilities, indices, self.classifier.classes,
                                               len(extracted.landmarks), threshold)
            for frame, (label, confidence) in zip(extracted.overlays_bgr, display):
                cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (55, 35, 20), -1)
                cv2.putText(frame, f"{label}  {confidence:.0%}", (12, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (240, 245, 250), 2, cv2.LINE_AA)
            for token in tokens:
                token["time_seconds"] = extracted.sampled_frame_indices[token["sample_index"]] / extracted.input_fps
        preview_path = Path(tempfile.gettempdir()) / f"sgsl_landmarks_{uuid.uuid4().hex}.mp4"
        preview_fps = min(15.0, max(6.0, extracted.input_fps * len(extracted.overlays_bgr) / max(1, extracted.sampled_frame_indices[-1] + 1)))
        save_landmark_preview(extracted.overlays_bgr, preview_path, preview_fps)
        note = self._model_note()
        return InferenceResult(
            prediction=prediction,
            detection_rate=extracted.detection_rate,
            preview_path=str(preview_path),
            note=note,
            sequence_text="".join(token["label"] for token in tokens),
            timeline=tuple(tokens),
        )

    def _model_note(self) -> str:
        metadata = self.classifier.metadata
        if metadata.get("demo_only"):
            return "Overfit alphabet smoke checkpoint"
        if metadata.get("kind") == "landmark_mlp_v1":
            return "GPU-trained alphabet MLP; ASL image source, 24 static letters"
        return "Custom isolated-sign model"


def format_result(result: InferenceResult) -> tuple[str, dict[str, float], str]:
    prediction = result.prediction
    if result.sequence_text and len(result.sequence_text) > 1:
        headline = f"Stable letter sequence: {result.sequence_text}"
    elif prediction.label == "NO_HAND":
        headline = "No hand detected"
    elif not prediction.accepted:
        headline = f"Uncertain ({prediction.confidence:.0%}) - retry or type"
    else:
        headline = f"{prediction.label}  ·  {prediction.confidence:.0%}"
    top = dict(list(prediction.probabilities.items())[:5])
    details = (
        f"{result.note}. Hand detection coverage: {result.detection_rate:.0%}. "
        "Review the output before adding it to your message."
    )
    return headline, top, details
