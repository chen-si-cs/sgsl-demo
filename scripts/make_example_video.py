from __future__ import annotations

from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
import subprocess


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    source_path = root / "examples" / "example_C.jpg"
    source = cv2.imdecode(np.fromfile(source_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if source is None:
        raise SystemExit("examples/example_C.jpg is missing")
    height, width = source.shape[:2]
    output = root / "examples" / "example_C.mp4"
    temporary = Path("example_C_raw.mp4")
    encoded = Path("example_C_h264.mp4")
    writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (width, height))
    for _ in range(45):
        writer.write(source)
    writer.release()
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(), "-loglevel", "error", "-y",
            "-i", str(temporary), "-an", "-c:v", "libx264",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(encoded),
        ],
        check=True,
    )
    encoded.replace(output)
    temporary.unlink(missing_ok=True)
    print(output)


if __name__ == "__main__":
    main()
