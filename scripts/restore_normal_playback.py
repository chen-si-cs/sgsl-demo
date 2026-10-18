"""Restore source-speed playback of an existing HUD export without re-inference."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import cv2
import imageio_ffmpeg
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("processed", type=Path)
    parser.add_argument("original", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Choose a new output path; existing files are not overwritten")
    report = json.loads(args.processed.with_suffix(".predictions.json").read_text(encoding="utf-8"))
    if hashlib.sha256(args.original.read_bytes()).hexdigest() != report["source_sha256"]:
        raise ValueError("Original source does not match the prediction report")
    capture = cv2.VideoCapture(str(args.processed))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if frames != report["frames"] or width <= 0 or height < 144:
        raise ValueError("Processed video does not match its report")
    # Replace only the old speed disclaimer; preserve every predicted HUD frame.
    footer = np.full((28, width, 3), (36, 25, 15), np.uint8)
    text = "1x playback | Original timing and audio | Alphabet-model output, not sentence translation"
    text_width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, .39, 1)[0][0]
    cv2.putText(footer, text, ((width - text_width) // 2, 20), cv2.FONT_HERSHEY_SIMPLEX,
                .39, (185, 185, 185), 1, cv2.LINE_AA)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="sgsl_retime_") as temporary:
        footer_path = Path(temporary) / "footer.png"
        footer_path.write_bytes(cv2.imencode(".png", footer)[1].tobytes())
        filters = (f"[0:v]setpts={report['playback_speed']:g}*(PTS-STARTPTS)[normal];"
                   f"[normal][2:v]overlay=0:{height - 52}[video]")
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-loglevel", "error",
                        "-i", str(args.processed), "-i", str(args.original), "-i", str(footer_path),
                        "-filter_complex", filters, "-map", "[video]", "-map", "1:a?",
                        "-r", str(report["source_fps"]), "-c:v", "libx264", "-crf", "18",
                        "-pix_fmt", "yuv420p", "-c:a", "copy", "-t", str(report["source_duration_seconds"]),
                        "-movflags", "+faststart", str(args.output)], check=True)
    output_capture = cv2.VideoCapture(str(args.output))
    output_frames = int(output_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    output_fps = output_capture.get(cv2.CAP_PROP_FPS)
    output_capture.release()
    if output_frames != frames or abs(output_fps - report["source_fps"]) > .001:
        raise ValueError("Retimed video failed the source timing check")
    report.update({"fps": output_fps, "duration_seconds": report["source_duration_seconds"],
                   "playback_speed": 1, "retimed_from": args.processed.name,
                   "retiming": "Predictions unchanged; original source audio restored without time stretching",
                   "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest()})
    for token in report["timeline"]:
        token["playback_time_seconds"] = token["time_seconds"]
    args.output.with_suffix(".predictions.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "frames": frames, "fps": output_fps,
                      "duration_seconds": report["duration_seconds"], "composition": report["composition"]}, indent=2))


if __name__ == "__main__":
    main()
