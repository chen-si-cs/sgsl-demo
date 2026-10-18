from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import subprocess
from typing import Iterable

import cv2
import imageio_ffmpeg
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from .features import FEATURE_CHANNELS, HAND_LANDMARKS, LANDMARK_COUNT, POSE_INDICES


HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)
POSE_CONNECTIONS = ((3, 4), (4, 5), (5, 7), (3, 6), (6, 8), (3, 9), (4, 10))


@dataclass
class ExtractedFrame:
    landmarks: np.ndarray
    overlay_bgr: np.ndarray
    hands_found: int
    pose_found: bool


@dataclass
class ExtractedSequence:
    landmarks: np.ndarray
    overlays_bgr: list[np.ndarray]
    sampled_frame_indices: list[int]
    input_fps: float
    detection_rate: float


def _default_model_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "models" / "mediapipe"


def video_sample_indices(frame_count: int, fps: float, *, sample_hz: float | None = None,
                         max_frames: int | None = 72) -> list[int]:
    if frame_count <= 0 or not np.isfinite(fps) or fps <= 0:
        raise ValueError("Video must have a positive frame count and frame rate")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be positive")
    if sample_hz is None:
        count = min(frame_count, max_frames or frame_count)
        return sorted(set(np.linspace(0, frame_count - 1, count).astype(int).tolist()))
    if not np.isfinite(sample_hz) or sample_hz <= 0:
        raise ValueError("sample_hz must be positive and finite")
    # Quantise the fixed source-time grid to the closest available source frame.
    grid = np.arange(0.0, frame_count / fps, 1.0 / sample_hz)
    indices = sorted(set(np.rint(grid * fps).astype(int).tolist()))
    indices = [index for index in indices if index < frame_count]
    if max_frames is not None and len(indices) > max_frames:
        raise ValueError(f"Clip requires {len(indices)} samples at {sample_hz:g} Hz; use a shorter clip (limit {max_frames})")
    return indices


class MediaPipeLandmarkExtractor:
    """Extract two hands plus an upper-body subset with official MediaPipe Tasks."""

    def __init__(self, model_dir: str | Path | None = None) -> None:
        model_dir = Path(model_dir) if model_dir else _default_model_dir()
        hand_path = model_dir / "hand_landmarker.task"
        pose_path = model_dir / "pose_landmarker_lite.task"
        if not hand_path.exists() or not pose_path.exists():
            raise FileNotFoundError(
                "MediaPipe task files are missing. Run scripts/download_mediapipe_models.py."
            )
        hand_options = vision.HandLandmarkerOptions(
            # Use buffers because some Windows MediaPipe wheels cannot open model
            # paths containing non-ASCII user names.
            base_options=python.BaseOptions(model_asset_buffer=hand_path.read_bytes()),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.35,
            min_hand_presence_confidence=0.35,
        )
        pose_options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_buffer=pose_path.read_bytes()),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.35,
            min_pose_presence_confidence=0.35,
        )
        self.hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)
        self.pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)

    def close(self) -> None:
        self.hand_landmarker.close()
        self.pose_landmarker.close()

    def __enter__(self) -> "MediaPipeLandmarkExtractor":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @staticmethod
    def _resize(frame_bgr: np.ndarray, max_side: int = 720) -> np.ndarray:
        height, width = frame_bgr.shape[:2]
        scale = min(1.0, max_side / max(height, width))
        if scale == 1.0:
            return frame_bgr
        return cv2.resize(frame_bgr, (round(width * scale), round(height * scale)))

    @staticmethod
    def _mp_image(frame_bgr: np.ndarray) -> mp.Image:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))

    @staticmethod
    def _normalization(hand_result: object, pose_result: object) -> tuple[np.ndarray, float]:
        if getattr(pose_result, "pose_landmarks", None):
            pose = pose_result.pose_landmarks[0]
            left = np.asarray([pose[11].x, pose[11].y, pose[11].z], dtype=np.float32)
            right = np.asarray([pose[12].x, pose[12].y, pose[12].z], dtype=np.float32)
            center = (left + right) / 2.0
            scale = float(np.linalg.norm(left[:2] - right[:2]))
            if scale > 1e-3:
                return center, scale
        if getattr(hand_result, "hand_landmarks", None):
            hand = hand_result.hand_landmarks[0]
            wrist = np.asarray([hand[0].x, hand[0].y, hand[0].z], dtype=np.float32)
            middle = np.asarray([hand[9].x, hand[9].y, hand[9].z], dtype=np.float32)
            scale = float(np.linalg.norm(wrist[:2] - middle[:2]))
            return wrist, max(scale, 0.04)
        return np.asarray([0.5, 0.5, 0.0], dtype=np.float32), 1.0

    @staticmethod
    def _draw_landmarks(
        frame_bgr: np.ndarray, hand_result: object, pose_result: object
    ) -> np.ndarray:
        overlay = frame_bgr.copy()
        height, width = overlay.shape[:2]
        for hand in getattr(hand_result, "hand_landmarks", []):
            points = [(round(point.x * width), round(point.y * height)) for point in hand]
            for start, end in HAND_CONNECTIONS:
                cv2.line(overlay, points[start], points[end], (242, 157, 56), 2, cv2.LINE_AA)
            for point in points:
                cv2.circle(overlay, point, 3, (42, 225, 170), -1, cv2.LINE_AA)
        if getattr(pose_result, "pose_landmarks", None):
            pose = pose_result.pose_landmarks[0]
            points = [(round(pose[index].x * width), round(pose[index].y * height)) for index in POSE_INDICES]
            for start, end in POSE_CONNECTIONS:
                cv2.line(overlay, points[start], points[end], (64, 100, 255), 2, cv2.LINE_AA)
            for point in points:
                cv2.circle(overlay, point, 3, (255, 210, 82), -1, cv2.LINE_AA)
        return overlay

    def extract_frame(self, frame_bgr: np.ndarray) -> ExtractedFrame:
        frame_bgr = self._resize(np.asarray(frame_bgr))
        image = self._mp_image(frame_bgr)
        hand_result = self.hand_landmarker.detect(image)
        pose_result = self.pose_landmarker.detect(image)
        center, scale = self._normalization(hand_result, pose_result)
        packed = np.zeros((LANDMARK_COUNT, FEATURE_CHANNELS), dtype=np.float32)

        occupied: set[int] = set()
        for index, hand in enumerate(hand_result.hand_landmarks):
            label = ""
            score = 0.0
            if index < len(hand_result.handedness) and hand_result.handedness[index]:
                category = hand_result.handedness[index][0]
                label = (category.category_name or category.display_name or "").lower()
                score = float(category.score or 0.0)
            slot = 0 if label == "left" else HAND_LANDMARKS
            if slot in occupied:
                slot = HAND_LANDMARKS if slot == 0 else 0
            occupied.add(slot)
            for point_index, point in enumerate(hand):
                vector = np.asarray([point.x, point.y, point.z], dtype=np.float32)
                packed[slot + point_index, :3] = (vector - center) / scale
                packed[slot + point_index, 3] = max(score, 0.5)

        pose_found = bool(pose_result.pose_landmarks)
        if pose_found:
            pose = pose_result.pose_landmarks[0]
            offset = HAND_LANDMARKS * 2
            for output_index, pose_index in enumerate(POSE_INDICES):
                point = pose[pose_index]
                vector = np.asarray([point.x, point.y, point.z], dtype=np.float32)
                packed[offset + output_index, :3] = (vector - center) / scale
                packed[offset + output_index, 3] = float(
                    min(getattr(point, "visibility", 1.0), getattr(point, "presence", 1.0))
                )

        return ExtractedFrame(
            landmarks=packed,
            overlay_bgr=self._draw_landmarks(frame_bgr, hand_result, pose_result),
            hands_found=len(hand_result.hand_landmarks),
            pose_found=pose_found,
        )

    def extract_images(self, frames_bgr: Iterable[np.ndarray]) -> list[ExtractedFrame]:
        return [self.extract_frame(frame) for frame in frames_bgr]

    def extract_video(self, path: str | Path, max_frames: int | None = 72,
                      *, sample_hz: float | None = None) -> ExtractedSequence:
        path = Path(path)
        capture_path = str(path)
        if os.name == "nt" and not capture_path.isascii():
            try:
                capture_path = str(path.resolve().relative_to(Path.cwd().resolve()))
            except ValueError:
                capture_path = str(path)
        capture = cv2.VideoCapture(capture_path)
        if not capture.isOpened():
            raise ValueError(f"Could not open video: {path}")
        frame_count = max(1, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 24.0)
        try:
            indices = video_sample_indices(frame_count, fps, sample_hz=sample_hz, max_frames=max_frames)
        except Exception:
            capture.release()
            raise
        extracted: list[ExtractedFrame] = []
        used: list[int] = []
        for frame_index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                continue
            extracted.append(self.extract_frame(frame))
            used.append(frame_index)
        capture.release()
        if not extracted:
            raise ValueError("No readable frames were found in the video")
        sequence = np.stack([frame.landmarks for frame in extracted])
        rate = float(np.mean([frame.hands_found > 0 for frame in extracted]))
        return ExtractedSequence(
            landmarks=sequence,
            overlays_bgr=[frame.overlay_bgr for frame in extracted],
            sampled_frame_indices=used,
            input_fps=fps,
            detection_rate=rate,
        )


def save_landmark_preview(
    frames_bgr: list[np.ndarray], output_path: str | Path, fps: float = 12.0
) -> Path:
    if not frames_bgr:
        raise ValueError("No preview frames")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames_bgr[0].shape[:2]
    raw_path = Path(f"sgsl_raw_{output_path.stem}.mp4")
    encoded_path = Path(f"sgsl_h264_{output_path.stem}.mp4")
    writer = cv2.VideoWriter(
        str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"), max(1.0, fps), (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not create the MP4 preview")
    for frame in frames_bgr:
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height))
        writer.write(frame)
    writer.release()
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-loglevel", "error",
        "-y",
        "-i", str(raw_path),
        "-an",
        "-c:v", "libx264",
        "-crf", "12",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(encoded_path),
    ]
    try:
        subprocess.run(command, check=True)
        encoded_path.replace(output_path)
    finally:
        raw_path.unlink(missing_ok=True)
        encoded_path.unlink(missing_ok=True)
    return output_path

