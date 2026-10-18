"""Time-synchronised alphabet inference visualisation, shared by HF and exports."""
from __future__ import annotations

from bisect import bisect_right
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import cv2
import imageio_ffmpeg
import numpy as np

from .features import POSE_INDICES, validate_sequence
from .landmarks import HAND_CONNECTIONS


def composed_at(tokens, sample_index: int) -> str:
    return "".join(token["label"] for token in tokens if token["sample_index"] <= sample_index)


def _text(image, text, origin, size=0.6, color=(245, 247, 250), thickness=1):
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, size, color,
                thickness, cv2.LINE_AA)


def _center_text(image, text, y, size, color, thickness=1):
    width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, size, thickness)[0][0]
    _text(image, text, ((image.shape[1] - width) // 2, y), size, color, thickness)


def skeleton_bounds(sequence):
    points = sequence[..., :2][sequence[..., 3] > 0.25]
    if not len(points):
        return (-1.6, -1.3, 1.6, 2.3)
    lo, hi = np.percentile(points, [1, 99], axis=0)
    # One common body-relative viewport avoids skeleton rescaling every frame.
    return (float(min(lo[0] - .2, -1.2)), float(min(lo[1] - .3, -1.1)),
            float(max(hi[0] + .2, 1.2)), float(max(hi[1] + .2, 1.8)))


def draw_skeleton(landmarks, width, height, bounds):
    panel = np.full((height, width, 3), (247, 245, 241), np.uint8)
    blue, cyan = (165, 85, 0), (180, 145, 5)
    _text(panel, "LANDMARK VIEW", (12, 22), .42, blue)
    hand_found = bool(np.any(landmarks[:42, 3] > .25))
    cv2.circle(panel, (width - 17, 16), 4, (45, 55, 225) if hand_found else (170, 170, 170), -1)
    x0, y0, x1, y1 = bounds
    scale = min((width - 30) / (x1 - x0), (height - 43) / (y1 - y0))
    center = np.asarray([(x0 + x1) / 2, (y0 + y1) / 2])

    def point(index):
        xy = (landmarks[index, :2] - center) * scale + [width / 2, 30 + (height - 35) / 2]
        return tuple(np.rint(xy).astype(int))

    def connection(a, b, color=blue, thickness=2):
        if min(landmarks[a, 3], landmarks[b, 3]) > .25:
            cv2.line(panel, point(a), point(b), color, thickness, cv2.LINE_AA)

    pose = {original: 42 + index for index, original in enumerate(POSE_INDICES)}
    for a, b in ((11, 12), (11, 23), (12, 24), (23, 24),
                 (11, 13), (13, 15), (12, 14), (14, 16)):
        connection(pose[a], pose[b])
    if landmarks[pose[0], 3] > .25:
        cv2.circle(panel, point(pose[0]), max(6, round(scale * .17)), blue, 2, cv2.LINE_AA)
    for offset in (0, 21):
        for a, b in HAND_CONNECTIONS:
            connection(offset + a, offset + b, cyan, 1)
    if not np.any(landmarks[..., 3] > .25):
        _text(panel, "No landmarks detected", (12, height - 12), .36, (100, 100, 100))
    cv2.rectangle(panel, (0, 0), (width - 1, height - 1), (200, 180, 150), 1)
    return panel


def draw_hud(frame, landmarks, label, confidence, composed, bounds, seconds, duration, credit="", playback_speed=1):
    source_height, source_width = frame.shape[:2]
    width = max(960, min(1280, source_width))
    height = round(source_height * width / source_width)
    width, height = width // 2 * 2, height // 2 * 2
    canvas = np.full((height + 144, width, 3), (36, 25, 15), np.uint8)
    canvas[:height] = cv2.resize(frame, (width, height))
    margin = 16
    inset_width = round(width * .235)
    inset_height = round(inset_width * .68)
    canvas[margin:margin + inset_height, margin:margin + inset_width] = draw_skeleton(
        landmarks, inset_width, inset_height, bounds)
    card_width, card_height = 190, 156
    x = width - margin - card_width
    region = canvas[margin:margin + card_height, x:x + card_width]
    region[:] = (region.astype(np.float32) * .1 + np.asarray([36, 25, 15]) * .9).astype(np.uint8)
    _text(canvas, "CURRENT LETTER", (x + 16, margin + 25), .46, (220, 220, 220))
    accepted = label not in {"NO_HAND", "UNCERTAIN"}
    glyph = label if accepted else "-" if label == "NO_HAND" else "?"
    _text(canvas, glyph, (x + 68, margin + 99), 2.1, (195, 240, 100), 3)
    status = f"Confidence {confidence:.0%}" if accepted else (
        "No hand detected" if label == "NO_HAND" else f"Uncertain {confidence:.0%}")
    _text(canvas, status, (x + 16, margin + 133), .43)
    _text(canvas, "COMPOSED CHARACTERS", (24, height + 27), .47, (220, 206, 160))
    _text(canvas, f"{seconds:04.1f} / {duration:04.1f}s", (width - 160, height + 27), .48)
    message = composed or "Waiting for a stable letter..."
    if len(message) > 40:
        message = "..." + message[-37:]
    _center_text(canvas, message, height + 81, 1.1 if composed else .75, (250, 250, 250), 2)
    footer = "Alphabet-model output | Not SgSL sentence translation | Offline, time-synchronised"
    if playback_speed != 1:
        footer = f"{playback_speed:g}x playback | Source-time clock | Alphabet-model output, not sentence translation"
    _center_text(canvas, footer, height + 112, .39, (185, 185, 185))
    if credit:
        _center_text(canvas, credit, height + 132, .34, (185, 185, 185))
    return canvas


def render_prediction_video(source, output, sequence, sampled_indices, fps, display, tokens, credit="", *, playback_speed=1):
    if not np.isfinite(playback_speed) or not .25 <= playback_speed <= 1:
        raise ValueError("Playback speed must be between 0.25 and 1")
    sequence = validate_sequence(sequence)
    indices = [int(index) for index in sampled_indices]
    if len(sequence) != len(indices) or len(display) != len(indices):
        raise ValueError("Each sampled frame needs landmarks and a display prediction")
    if not indices or indices[0] < 0 or indices != sorted(set(indices)):
        raise ValueError("Sample indices must be ordered, unique and non-negative")
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or output.exists():
        raise ValueError("Choose a new output path; existing files are never overwritten")
    capture = cv2.VideoCapture(os.path.relpath(source))
    process = None
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot read video: {source}")
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        source_fps = float(capture.get(cv2.CAP_PROP_FPS))
        if count <= 0 or not np.isfinite(source_fps) or source_fps <= 0:
            raise ValueError("Video has no usable timing")
        if abs(source_fps - fps) > .01 or indices[-1] >= count:
            raise ValueError("Landmark timing does not match this source video")
        duration = count / fps
        bounds = skeleton_bounds(sequence)
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="sgsl_hud_") as temp:
            encoded = Path(temp) / "hud.mp4"
            with (Path(temp) / "ffmpeg.log").open("w+b") as errors:
                written = 0
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    sample = bisect_right(indices, written) - 1
                    landmarks = sequence[sample] if sample >= 0 else np.zeros_like(sequence[0])
                    label, confidence = display[sample] if sample >= 0 else ("NO_HAND", 0)
                    rendered = draw_hud(frame, landmarks, label, confidence, composed_at(tokens, sample),
                                        bounds, written / fps, duration, credit, playback_speed)
                    if process is None:
                        command = [imageio_ffmpeg.get_ffmpeg_exe(), "-loglevel", "error", "-f", "rawvideo",
                                   "-pix_fmt", "bgr24", "-s", f"{rendered.shape[1]}x{rendered.shape[0]}",
                                   "-r", str(fps * playback_speed), "-i", "pipe:0", "-i", str(source), "-map", "0:v:0",
                                   "-map", "1:a?", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
                                   "-c:a", "aac", "-t", str(duration / playback_speed),
                                   "-movflags", "+faststart", str(encoded)]
                        if playback_speed != 1:
                            # Chaining supports 0.25x while keeping source audio pitch.
                            remaining = playback_speed
                            tempos = []
                            while remaining < .5:
                                tempos.append("atempo=0.5")
                                remaining /= .5
                            tempos.append(f"atempo={remaining:g}")
                            command[-1:-1] = ["-af", ",".join(tempos)]
                        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=errors)
                    process.stdin.write(rendered.tobytes())
                    written += 1
                if process is None:
                    raise ValueError("No readable video frames")
                process.stdin.close()
                if process.wait() != 0:
                    errors.seek(0)
                    raise RuntimeError(errors.read().decode(errors="replace"))
            if written != count:
                raise ValueError(f"Incomplete decode: {written} / {count} frames")
            shutil.copyfile(encoded, output)
        report = {"kind": "model_prediction", "sentence_translation": False,
                  "fps": fps * playback_speed, "source_fps": fps, "frames": written,
                  "duration_seconds": duration / playback_speed, "source_duration_seconds": duration,
                  "playback_speed": playback_speed,
                  "sampled_frames": len(sequence), "composition": composed_at(tokens, len(sequence) - 1),
                  "timeline": list(tokens), "sampled_frame_indices": indices,
                  "frame_predictions": [{"sample_index": i, "time_seconds": indices[i] / fps,
                                         "label": label, "confidence": float(confidence)}
                                        for i, (label, confidence) in enumerate(display)],
                  "rendering": "offline; sample-held character HUD; not a live speed measurement"}
        for token in report["timeline"]:
            token["playback_time_seconds"] = indices[token["sample_index"]] / fps / playback_speed
        output.with_suffix(".predictions.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return output
    finally:
        capture.release()
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
