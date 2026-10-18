"""Export clearly labeled held-out successes and failures for human review."""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import zipfile

from PIL import Image, ImageDraw, ImageOps
import cv2
import numpy as np

from sgsl_demo.landmarks import save_landmark_preview


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    records = json.loads((args.run / "test_predictions.json").read_text())
    output = args.run / "review_examples"
    output.mkdir(exist_ok=True)
    successes = []
    for label in sorted(set(row["label"] for row in records)):
        options = [row for row in records if row["label"] == label and row["correct"]]
        if options:
            successes.append(max(options, key=lambda row: row["confidence"]))
    failures = sorted([row for row in records if not row["correct"]], key=lambda row: -row["confidence"])[:12]
    with zipfile.ZipFile(args.archive) as archive:
        for row in successes + failures:
            image = Image.open(io.BytesIO(archive.read(row["path"]))).convert("RGB")
            name = f"{'correct' if row['correct'] else 'error'}_{row['label']}_{row['prediction']}.jpg"
            image.save(output / name, quality=95)
            row["exported_file"] = name
        for title, selected in (("successes", successes), ("errors", failures)):
            if not selected:
                continue
            columns = 6
            sheet = Image.new("RGB", (columns * 220, ((len(selected) + columns - 1) // columns) * 260), "white")
            draw = ImageDraw.Draw(sheet)
            for index, row in enumerate(selected):
                image = Image.open(output / row["exported_file"])
                image.thumbnail((200, 210))
                x, y = (index % columns) * 220, (index // columns) * 260
                sheet.paste(image, (x + (220 - image.width) // 2, y + 5))
                draw.text((x + 8, y + 220), f"True {row['label']} / Pred {row['prediction']} ({row['confidence']:.0%})", fill="black")
            sheet.save(output / f"{title}_gallery.jpg", quality=92)
    # A montage of held-out stills exercises multi-letter video decoding.
    lookup = {row["label"]: row for row in successes}
    letters = "CAB" if all(label in lookup for label in "CAB") else "".join(list(lookup)[:3])
    frames = []
    for label in letters:
        image = Image.open(output / lookup[label]["exported_file"]).convert("RGB")
        # Keep the source resolution; changing the image scale can change the
        # landmark detector on these small, low-light public images.
        image = ImageOps.contain(image, (200, 200))
        canvas = Image.new("RGB", (200, 200), "white")
        canvas.paste(image, ((200 - image.width) // 2, (200 - image.height) // 2))
        bgr = cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)
        frames.extend([bgr] * 24)
        frames.extend([np.full((200, 200, 3), 255, np.uint8)] * 8)
    save_landmark_preview(frames, output / "heldout_CAB_montage.mp4", fps=12)
    (output / "EXAMPLES.json").write_text(json.dumps({"correct": successes, "errors": failures,
        "montage": {"letters": letters, "description": "Held-out still-image montage; not a real signing performance."}}, indent=2))
    print(output, flush=True)


if __name__ == "__main__":
    main()
