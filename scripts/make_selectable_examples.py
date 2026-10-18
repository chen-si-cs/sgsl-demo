"""Create a reproducible public-media example pack for the Gradio picker."""
from __future__ import annotations

import json
import os
from pathlib import Path
import random
import subprocess

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from sgsl_demo.landmarks import save_landmark_preview


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "examples" / "random_demo"
    manifest = json.loads((output / "random_source_manifest.json").read_text())
    rng = random.Random(manifest["seed"])
    by_label = {entry["label"]: entry for entry in manifest["images"]}
    videos = []
    letters = list(by_label)
    single_letters = rng.sample(letters, 2)
    for word in ["HI", "YES", "HELLO", *single_letters]:
        frames = []
        for label in word:
            image = Image.open(output / by_label[label]["file"]).convert("RGB")
            image = ImageOps.contain(image, (200, 200))
            canvas = Image.new("RGB", (200, 200), "white")
            canvas.paste(image, ((200 - image.width) // 2, (200 - image.height) // 2))
            bgr = cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)
            frames.extend([bgr] * 24)
            frames.extend([np.full((200, 200, 3), 255, np.uint8)] * 8)
        name = f"montage_{word}.mp4"
        save_landmark_preview(frames, output / name, 12)
        videos.append({"file": name, "title": f"{word} · held-out image montage",
                       "expected": word, "kind": "still_image_montage",
                       "description": f"Expected letters: {' → '.join(word)}. Constructed from randomly selected test images with blank gaps.",
                       "source_images": [by_label[label]["source_path"] for label in word]})
    reference = root.parent / "references" / "GloriaNeo-SgSL" / "streamlit" / "videos"
    for index, source in enumerate(sorted(reference.glob("video_*.mp4")), 1):
        relative_source = os.path.relpath(source, Path.cwd())
        capture = cv2.VideoCapture(relative_source)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 30)
        duration = capture.get(cv2.CAP_PROP_FRAME_COUNT) / fps
        capture.release()
        start = round(rng.uniform(0, max(0, duration - 6)), 2)
        destination = output / f"real_webcam_{index:02d}.mp4"
        subprocess.run([
            imageio_ffmpeg.get_ffmpeg_exe(), "-loglevel", "error", "-y",
            "-ss", str(start), "-i", relative_source, "-t", "6", "-an",
            "-vf", "scale=640:-2,fps=15", "-c:v", "libx264", "-crf", "20",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            os.path.relpath(destination, Path.cwd()),
        ], check=True)
        videos.append({"file": destination.name, "title": f"Real webcam {index} · 6 seconds",
                       "kind": "real_webcam_excerpt", "expected": None,
                       "description": "Real public webcam footage; no verified frame-level labels. Use to explore live-camera variation.",
                       "source": f"GloriaNeo/SgSL-Sign-Language-Recognition/streamlit/videos/{source.name}",
                       "source_start_seconds": start, "duration_seconds": 6})
    manifest["videos"] = videos
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    sheet = Image.new("RGB", (6 * 200, 4 * 230), "#f5f7fb")
    draw = ImageDraw.Draw(sheet)
    for index, entry in enumerate(manifest["images"]):
        image = Image.open(output / entry["file"]).convert("RGB")
        image = ImageOps.contain(image, (180, 190))
        x, y = (index % 6) * 200, (index // 6) * 230
        sheet.paste(image, (x + (200 - image.width) // 2, y + 6))
        draw.text((x + 10, y + 205), f"True label: {entry['label']}", fill="#12304a")
    sheet.save(output / "random_images_gallery.jpg", quality=92)
    print(f"Created {len(manifest['images'])} images and {len(videos)} clips", flush=True)


if __name__ == "__main__":
    main()
