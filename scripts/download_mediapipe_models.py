from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path


FILES = {
    "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    "pose_landmarker_lite.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
}


def main() -> None:
    destination = Path(__file__).resolve().parents[1] / "models" / "mediapipe"
    destination.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        path = destination / name
        if not path.exists():
            urllib.request.urlretrieve(url, path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        print(f"{name} {path.stat().st_size} bytes sha256={digest}")


if __name__ == "__main__":
    main()

