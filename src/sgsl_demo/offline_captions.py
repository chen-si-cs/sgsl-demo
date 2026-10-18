"""Render timed captions; this module does not recognise or translate signs."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile

import cv2
import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont


def validate_captions(document: dict, duration: float | None = None) -> list[dict]:
    if document.get("kind") not in {"prediction", "reference"}:
        raise ValueError("Caption kind must be prediction or reference")
    if not str(document.get("source", "")).strip():
        raise ValueError("Caption provenance source is required")
    if document["kind"] == "prediction" and not str(document.get("model", "")).strip():
        raise ValueError("Predicted captions must identify the model")
    segments = document.get("segments")
    if not isinstance(segments, list) or not segments:
        raise ValueError("A non-empty segments list is required")
    previous_end = 0.0
    for item in segments:
        start, end = float(item["start"]), float(item["end"])
        if not all(math.isfinite(x) for x in (start, end)):
            raise ValueError("Caption times must be finite")
        if start < previous_end or end <= start:
            raise ValueError("Captions must be ordered, non-overlapping positive intervals")
        if duration is not None and end > duration + 0.05:
            raise ValueError("Caption extends beyond the video")
        if not isinstance(item.get("text"), str) or not item["text"].strip():
            raise ValueError("Each caption requires text")
        item["start"], item["end"] = start, end
        previous_end = end
    return segments


def caption_at(segments: list[dict], time: float) -> str:
    return next((s["text"] for s in segments if s["start"] <= time < s["end"]), "")


def load_font(size: int):
    for name in ("C:/Windows/Fonts/arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def draw_caption(frame_rgb: Image.Image, text: str, kind: str, attribution: str) -> Image.Image:
    """Reserve a bottom caption strip so captions do not obscure the hands."""
    width, height = frame_rgb.size
    band_height = max(100, round(height * 0.20))
    output = Image.new("RGB", (width, (height + band_height + 1) // 2 * 2), "#101b2a")
    output.paste(frame_rgb, (0, 0))
    draw = ImageDraw.Draw(output)
    label = "MODEL PREDICTION" if kind == "prediction" else "REFERENCE CAPTIONS - NOT MODEL PREDICTIONS"
    draw.text((width / 2, height + 8), label, anchor="mt", fill="#7dd3fc",
              font=load_font(max(10, round(width / 95))))
    font = load_font(max(14, round(width / 38)))
    lines: list[str] = []
    for paragraph in text.splitlines():
        line = ""
        for word in paragraph.split():
            candidate = f"{line} {word}".strip()
            if draw.textlength(candidate, font=font) > width * 0.92 and line:
                lines.append(line)
                line = word
            else:
                line = candidate
        if line:
            lines.append(line)
    line_height = font.size + 5
    if len(lines) > 2 or any(draw.textlength(line, font=font) > width * 0.94 for line in lines):
        raise ValueError("Caption is too long; split it into shorter timed segments")
    center_y = height + band_height * 0.52
    for index, line in enumerate(lines):
        y = center_y + (index - (len(lines) - 1) / 2) * line_height
        draw.text((width / 2, y), line, font=font, fill="white", anchor="mm")
    credit_font = load_font(max(9, round(width / 110)))
    if draw.textlength(attribution, font=credit_font) > width * 0.94:
        raise ValueError("Attribution is too long for the caption strip")
    draw.text((width / 2, output.height - 8), attribution, font=credit_font,
              fill="#b6c5d8", anchor="mb")
    return output


def render_video(source: Path, captions: Path, output: Path) -> dict:
    source, captions, output = (p.resolve() for p in (source, captions, output))
    if output in {source, captions} or output.exists():
        raise ValueError("Choose a new output path; inputs and existing files are never overwritten")
    document = json.loads(captions.read_text(encoding="utf-8"))
    attribution = str(document.get("attribution", document["source"]))
    capture = cv2.VideoCapture(os.path.relpath(source))
    if not capture.isOpened():
        raise ValueError(f"Cannot open {source}")
    process = None
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not math.isfinite(fps) or fps <= 0 or frame_count <= 0:
            raise ValueError("Video has no usable frame timing")
        segments = validate_captions(document, frame_count / fps)
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="sgsl_captions_") as temporary:
            temporary_output = Path(temporary) / "captioned.mp4"
            error_log = Path(temporary) / "ffmpeg.log"
            with error_log.open("wb") as errors:
                written = 0
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    rgb = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    rendered = draw_caption(rgb, caption_at(segments, written / fps),
                                            document["kind"], attribution)
                    if process is None:
                        command = [imageio_ffmpeg.get_ffmpeg_exe(), "-loglevel", "error",
                                   "-f", "rawvideo", "-pix_fmt", "rgb24", "-s",
                                   f"{rendered.width}x{rendered.height}", "-r", str(fps),
                                   "-i", "pipe:0", "-i", str(source), "-map", "0:v:0",
                                   "-map", "1:a?", "-c:v", "libx264", "-crf", "18",
                                   "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
                                   "-movflags", "+faststart", str(temporary_output)]
                        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=errors)
                    process.stdin.write(rendered.tobytes())
                    written += 1
                if process is None:
                    raise ValueError("No readable video frames")
                process.stdin.close()
                if process.wait() != 0:
                    raise RuntimeError(error_log.read_text(errors="replace"))
            if written != frame_count:
                raise ValueError(f"Incomplete decode: {written} / {frame_count} frames")
            import shutil
            shutil.copyfile(temporary_output, output)
        source_hash = hashlib.sha256()
        with source.open("rb") as source_file:
            for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
                source_hash.update(chunk)
        receipt = {"kind": document["kind"], "caption_source": document["source"],
                   "model": document.get("model"), "frames": written, "fps": fps,
                   "duration_seconds": written / fps, "audio": "preserved if present",
                   "rendering": "offline; not a live-latency measurement",
                   "source_sha256": source_hash.hexdigest(),
                   "captions_sha256": hashlib.sha256(captions.read_bytes()).hexdigest()}
        output.with_suffix(".receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        return receipt
    finally:
        capture.release()
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("captions", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(render_video(args.video, args.captions, args.output), indent=2))


if __name__ == "__main__":
    main()
