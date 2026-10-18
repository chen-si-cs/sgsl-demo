"""Fetch the public alphabet source cited by the SgSL reference repository."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
import zipfile

import requests


SOURCE = "prathumarikeri/american-sign-language-09az"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    parser.add_argument("--archive-only", action="store_true", help="Index the ZIP without expanding small files on shared storage")
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    archive = args.destination / "american-sign-language-09az.zip"
    partial = archive.with_suffix(".zip.part")
    url = f"https://www.kaggle.com/api/v1/datasets/download/{SOURCE}"
    if not archive.exists():
        for attempt in range(3):
            try:
                with requests.get(url, stream=True, timeout=(30, 120)) as response:
                    response.raise_for_status()
                    total = 0
                    last_log = time.monotonic()
                    with partial.open("wb") as stream:
                        for chunk in response.iter_content(4 * 1024 * 1024):
                            if not chunk:
                                continue
                            stream.write(chunk)
                            total += len(chunk)
                            if time.monotonic() - last_log > 10:
                                print(f"downloaded {total / 1e6:.1f} MB", flush=True)
                                last_log = time.monotonic()
                with partial.open("rb") as stream:
                    if stream.read(4) != b"PK\x03\x04":
                        raise ValueError("Download is not a ZIP archive")
                partial.replace(archive)
                break
            except Exception as error:
                print(f"attempt {attempt + 1}: {type(error).__name__}: {error}", flush=True)
                if attempt == 2:
                    raise
                time.sleep(3)
    digest = hashlib.file_digest(archive.open("rb"), "sha256").hexdigest()
    extracted = args.destination / "images"
    extracted.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        for info in bundle.infolist():
            target = (extracted / info.filename).resolve()
            if not target.is_relative_to(extracted.resolve()):
                raise ValueError(f"Unsafe ZIP path: {info.filename}")
        bad = bundle.testzip()
        if bad:
            raise ValueError(f"Corrupt archive member: {bad}")
        members = [info.filename for info in bundle.infolist()
                   if Path(info.filename).suffix.lower() in {".jpg", ".jpeg", ".png"}]
        if not args.archive_only:
            bundle.extractall(extracted)
    images = [Path(member) for member in members]
    counts = {}
    for path in images:
        counts[path.parent.name] = counts.get(path.parent.name, 0) + 1
    receipt = {
        "source": SOURCE,
        "source_url": f"https://www.kaggle.com/datasets/{SOURCE}",
        "license_as_reported_by_kaggle": "CC BY-SA 4.0",
        "archive_bytes": archive.stat().st_size,
        "sha256": digest,
        "images": len(images),
        "archive_only": args.archive_only,
        "parent_counts": counts,
    }
    (args.destination / "DOWNLOAD_RECEIPT.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
