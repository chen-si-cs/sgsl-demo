from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from sgsl_demo.landmarks import MediaPipeLandmarkExtractor


VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert data/clips/<label>/*.mp4 into normalized landmark .npz files"
    )
    parser.add_argument("clips", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--max-frames", type=int, default=72)
    args = parser.parse_args()
    videos = [path for path in args.clips.rglob("*") if path.suffix.lower() in VIDEO_SUFFIXES]
    if not videos:
        raise SystemExit(f"No videos found under {args.clips}")
    args.output.mkdir(parents=True, exist_ok=True)
    with MediaPipeLandmarkExtractor() as extractor:
        for index, path in enumerate(videos, 1):
            label = path.parent.name
            result = extractor.extract_video(path, max_frames=args.max_frames)
            destination = args.output / label / f"{path.stem}.npz"
            destination.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                destination,
                landmarks=result.landmarks,
                source=str(path),
                detection_rate=result.detection_rate,
            )
            print(f"[{index}/{len(videos)}] {label}: {path.name} -> {destination.name}")


if __name__ == "__main__":
    main()

