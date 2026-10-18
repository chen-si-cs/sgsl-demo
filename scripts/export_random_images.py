"""Select one held-out image per letter without looking at model predictions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import zipfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=20261018)
    args = parser.parse_args()
    records = json.loads((args.run / "test_predictions.json").read_text())
    rng = random.Random(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    entries = []
    with zipfile.ZipFile(args.archive) as archive:
        for label in sorted(set(row["label"] for row in records)):
            candidates = sorted((row for row in records if row["label"] == label), key=lambda row: row["path"])
            chosen = rng.choice(candidates)
            suffix = Path(chosen["path"]).suffix.lower()
            filename = f"random_{label}{suffix}"
            # Preserve the exact encoded source pixels instead of re-saving JPEG.
            (args.output / filename).write_bytes(archive.read(chosen["path"]))
            entries.append({"file": filename, "label": label, "source_path": chosen["path"],
                            "split": "test", "selection": "uniform per class; no prediction/confidence filter"})
    manifest = {"seed": args.seed, "source": "prathumarikeri/american-sign-language-09az",
                "license": "CC BY-SA 4.0", "images": entries}
    (args.output / "random_source_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Exported {len(entries)} random test images", flush=True)


if __name__ == "__main__":
    main()
