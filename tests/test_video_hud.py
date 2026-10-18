import json
from pathlib import Path
import subprocess

import cv2
import imageio_ffmpeg
import numpy as np
import pytest

from sgsl_demo.features import FEATURE_CHANNELS, LANDMARK_COUNT
from sgsl_demo.video_hud import composed_at, draw_skeleton, render_prediction_video
from sgsl_demo.classifier import Prediction
from sgsl_demo.pipeline import InferenceResult, format_result


def test_composition_never_leaks_future_tokens():
    tokens = [{"sample_index": 2, "label": "A"}, {"sample_index": 6, "label": "B"}]
    assert composed_at(tokens, -1) == ""
    assert composed_at(tokens, 1) == ""
    assert composed_at(tokens, 2) == "A"
    assert composed_at(tokens, 5) == "A"
    assert composed_at(tokens, 6) == "AB"


def test_one_stable_token_is_not_hidden_by_uncertain_video_pool():
    result = InferenceResult(Prediction("UNCERTAIN", .4, {}, False), .8, "preview.mp4", "test",
                             sequence_text="A")
    assert format_result(result)[0] == "Stable letter sequence: A"


def test_skeleton_does_not_invent_missing_body():
    frame = np.zeros((LANDMARK_COUNT, FEATURE_CHANNELS), np.float32)
    panel = draw_skeleton(frame, 280, 180, (-2, -1, 2, 2))
    # With no detections there is no synthetic torso or arm in the middle.
    assert np.all(panel[45:130, 70:210] == [247, 245, 241])


@pytest.mark.parametrize("speed", [1, .5])
def test_hud_output_keeps_source_timing(tmp_path, speed):
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    source, output = tmp_path / "source.mp4", tmp_path / "hud.mp4"
    subprocess.run([ffmpeg, "-loglevel", "error", "-f", "lavfi", "-i",
                    "color=c=red:s=320x240:r=10:d=1", "-f", "lavfi", "-i",
                    "sine=frequency=440:duration=0.94", "-c:v", "libx264", "-pix_fmt",
                    "yuv420p", "-c:a", "aac", str(source)], check=True)
    sequence = np.zeros((2, LANDMARK_COUNT, FEATURE_CHANNELS), np.float32)
    tokens = [{"sample_index": 1, "label": "A", "confidence": .9, "time_seconds": .5}]
    render_prediction_video(source, output, sequence, [0, 5], 10,
                            [("UNCERTAIN", .3), ("A", .9)], tokens, playback_speed=speed)
    capture = cv2.VideoCapture(str(output))
    assert capture.get(cv2.CAP_PROP_FRAME_COUNT) == 10
    assert capture.get(cv2.CAP_PROP_FPS) == 10 * speed
    assert capture.get(cv2.CAP_PROP_FRAME_WIDTH) == 960
    assert capture.get(cv2.CAP_PROP_FRAME_HEIGHT) == 864
    capture.release()
    report = json.loads(output.with_suffix(".predictions.json").read_text())
    assert report["composition"] == "A"
    assert report["sentence_translation"] is False
    assert report["timeline"][0]["time_seconds"] == .5
    assert report["timeline"][0]["playback_time_seconds"] == .5 / speed
    assert report["duration_seconds"] == 1 / speed
    subprocess.run([ffmpeg, "-loglevel", "error", "-i", str(output),
                    "-map", "0:a:0", "-f", "null", "-"], check=True)
    with pytest.raises(ValueError, match="never overwritten"):
        render_prediction_video(source, output, sequence, [0, 5], 10, [("A", .9)] * 2, tokens)


def test_featured_example_is_raw_and_not_a_label_input():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "examples" / "featured" / "manifest.json").read_text())
    for entry in manifest["videos"]:
        assert entry["expected"] is None
        assert entry["file"] != entry["processed_file"]
        assert (root / "examples" / "featured" / entry["file"]).is_file()
        assert (root / "examples" / "featured" / entry["processed_file"]).is_file()


def test_replacement_example_uses_requested_source_and_trim():
    root = Path(__file__).resolve().parents[1] / "examples" / "featured"
    entry = json.loads((root / "manifest.json").read_text())["videos"][0]
    assert entry["source"] == "https://www.youtube.com/watch?v=LxdjT1g3REc"
    assert entry["source_start_seconds"] == 4
    assert entry["duration_seconds"] == 6
    for name in (entry["file"], entry["processed_file"]):
        capture = cv2.VideoCapture(str(root / name))
        assert capture.isOpened()
        assert capture.get(cv2.CAP_PROP_FRAME_COUNT) == 180
        expected_fps = 30 if name == entry["file"] else 30 * entry.get("processed_playback_speed", 1)
        assert capture.get(cv2.CAP_PROP_FPS) == expected_fps
        capture.release()


@pytest.mark.parametrize("prefix,frames,source_seconds,composition", [
    ("deafsg_abcd_10hz", 195, 6.5, "ABCD"),
    ("deafsg_alphabet_10hz", 1532, 1532 / 30, None),
])
def test_alphabet_lesson_exports_and_frozen_results(prefix, frames, source_seconds, composition):
    root = Path(__file__).resolve().parents[1] / "examples" / "featured"
    entries = json.loads((root / "manifest.json").read_text())["videos"]
    entry = next(row for row in entries if row["file"] == f"{prefix}_excerpt.mp4")
    assert entry["source"] == "https://www.youtube.com/watch?v=6EW4PVT_d_4"
    assert entry["source_start_seconds"] == 0
    assert entry["duration_seconds"] == pytest.approx(source_seconds)
    report = json.loads((root / entry["report_file"]).read_text())
    assert report["sampling_hz"] == 10
    assert report["threshold"] == .65
    if composition is not None:
        assert report["composition"] == composition
    for key, fps in (("file", 30), ("processed_file", 15)):
        capture = cv2.VideoCapture(str(root / entry[key]))
        assert capture.isOpened()
        assert capture.get(cv2.CAP_PROP_FRAME_COUNT) == frames
        assert capture.get(cv2.CAP_PROP_FPS) == fps
        capture.release()
