import pytest
import json
import subprocess

import cv2
import imageio_ffmpeg
from PIL import Image

from sgsl_demo.offline_captions import caption_at, draw_caption, render_video, validate_captions


def reference():
    return {"kind": "reference", "source": "Manual test annotation",
            "segments": [{"start": 0, "end": 1, "text": "Test caption"}]}


def test_explicit_provenance_and_model_required():
    document = reference()
    validate_captions(document)
    document["kind"] = "prediction"
    with pytest.raises(ValueError, match="model"):
        validate_captions(document)


def test_invalid_timing_rejected():
    document = reference()
    document["segments"].append({"start": 0.5, "end": 2, "text": "Overlap"})
    with pytest.raises(ValueError, match="non-overlapping"):
        validate_captions(document)
    with pytest.raises(ValueError, match="beyond"):
        validate_captions(reference(), duration=0.5)


def test_half_open_caption_intervals():
    segments = validate_captions(reference())
    assert caption_at(segments, 0) == "Test caption"
    assert caption_at(segments, 1) == ""


def test_caption_strip_preserves_signing_pixels():
    frame = Image.new("RGB", (640, 360), "red")
    rendered = draw_caption(frame, "Test caption", "reference", "Test source")
    assert rendered.size == (640, 460)
    assert rendered.crop((0, 0, 640, 360)).tobytes() == frame.tobytes()
    with pytest.raises(ValueError, match="too long"):
        draw_caption(frame, "long " * 200, "reference", "Test source")


def test_render_mp4_preserves_timing_and_audio(tmp_path):
    source = tmp_path / "source_测试.mp4"
    output = tmp_path / "output.mp4"
    captions = tmp_path / "captions.json"
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([ffmpeg, "-loglevel", "error", "-f", "lavfi", "-i",
                    "color=c=red:s=320x240:r=10:d=1", "-f", "lavfi", "-i",
                    "sine=frequency=440:duration=1", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(source)], check=True)
    captions.write_text(json.dumps(reference()), encoding="utf-8")
    receipt = render_video(source, captions, output)
    capture = cv2.VideoCapture(str(output))
    assert capture.isOpened()
    assert capture.get(cv2.CAP_PROP_FRAME_COUNT) == 10
    assert capture.get(cv2.CAP_PROP_FPS) == 10
    assert capture.get(cv2.CAP_PROP_FRAME_HEIGHT) == 340
    capture.release()
    assert receipt["kind"] == "reference"
    assert output.with_suffix(".receipt.json").exists()
    subprocess.run([ffmpeg, "-loglevel", "error", "-i", str(output),
                    "-map", "0:a:0", "-f", "null", "-"], check=True)
    with pytest.raises(ValueError, match="never overwritten"):
        render_video(source, captions, output)
