"""Package a source clip and genuine CPU predictions for HF."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
import cv2

from sgsl_demo.pipeline import RecognitionPipeline


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, default=root / "examples" / "featured")
    parser.add_argument("--threshold", type=float, default=.65)
    parser.add_argument("--slug", default="sgsl_my_country")
    parser.add_argument("--title", default="My Country | 6-second SgSL excerpt")
    parser.add_argument("--creator", default="Jie Ying")
    parser.add_argument("--source-url", default="https://www.youtube.com/watch?v=Vko4It1y7rY")
    parser.add_argument("--source-start", type=float, default=20)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--append", action="store_true", help="Keep existing featured entries")
    parser.add_argument("--kind", default="public_sgsl_demo_excerpt")
    parser.add_argument("--sample-hz", type=float, default=10)
    parser.add_argument("--playback-speed", type=float, default=.5)
    parser.add_argument("--description", default="Natural word-sign footage, not fingerspelling. The alphabet model may be wrong or abstain; source subtitles are not predictions.")
    args = parser.parse_args()
    capture = cv2.VideoCapture(str(args.source))
    if not capture.isOpened() or capture.get(cv2.CAP_PROP_FPS) <= 0:
        capture.release()
        raise ValueError("Cannot read the source clip timing")
    actual_duration = capture.get(cv2.CAP_PROP_FRAME_COUNT) / capture.get(cv2.CAP_PROP_FPS)
    capture.release()
    if args.duration is not None and abs(args.duration - actual_duration) > .1:
        raise ValueError("Supplied duration does not match the actual source clip")
    args.duration = actual_duration
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.slug or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_" for character in args.slug):
        raise ValueError("Slug must contain lowercase ASCII letters, digits or underscores")
    raw = args.output / f"{args.slug}_excerpt.mp4"
    if raw.exists() and raw.read_bytes() != args.source.read_bytes():
        raise ValueError("Existing source differs; choose a new output directory")
    if not raw.exists():
        shutil.copyfile(args.source, raw)
    processed = args.output / f"{args.slug}_processed.mp4"
    model = root / "models" / "alphabet_mlp.npz"
    start = time.perf_counter()
    result = RecognitionPipeline(model).predict_video(
        raw, args.threshold, output_path=processed, sample_hz=args.sample_hz, playback_speed=args.playback_speed,
        credit=f"Source: {args.creator} | {args.source_start:g}-{args.source_start + args.duration:g}s")
    report_path = processed.with_suffix(".predictions.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report.update({"model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
                   "source_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
                   "threshold": args.threshold,
                   "hand_detection_coverage": result.detection_rate,
                   "processing_wall_seconds": time.perf_counter() - start,
                   "source_url": args.source_url,
                   "source_start_seconds": args.source_start,
                   "source_end_seconds": args.source_start + args.duration})
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    manifest = {"videos": [{"file": raw.name, "processed_file": processed.name,
                           "report_file": report_path.name,
                           "title": args.title,
                           "description": args.description,
                           "kind": args.kind, "expected": None,
                           "source": report["source_url"], "creator": args.creator,
                           "source_start_seconds": args.source_start, "duration_seconds": args.duration,
                           "sampling_hz": args.sample_hz, "processed_playback_speed": args.playback_speed,
                           "processed_duration_seconds": report["duration_seconds"],
                           "license": "Original creator rights retained; no explicit reuse license verified"}]}
    manifest_path = args.output / "manifest.json"
    if args.append and manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        previous["videos"] = [entry for entry in previous.get("videos", []) if entry["file"] != raw.name] + manifest["videos"]
        manifest = previous
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"raw": str(raw), "processed": str(processed),
                      "composition": result.sequence_text, "timeline": result.timeline,
                      "detection_coverage": result.detection_rate,
                      "wall_seconds": report["processing_wall_seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
