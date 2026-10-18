"""Extract all static alphabet images with a bounded CPU worker pool."""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import hashlib
import json
import os
from pathlib import Path
import time
import zipfile

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

from sgsl_demo.static_features import hand_features

EXTRACTOR = None
ARCHIVE = None


def init_worker(model_path: str, archive_path: str | None = None) -> None:
    global EXTRACTOR, ARCHIVE
    if archive_path:
        ARCHIVE = zipfile.ZipFile(archive_path)
    cv2.setNumThreads(1)
    EXTRACTOR = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_buffer=Path(model_path).read_bytes()),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.35,
        min_hand_presence_confidence=0.35,
    ))


def extract_one(item: tuple[str, str, str]) -> dict:
    path_string, relative, label = item
    try:
        raw = ARCHIVE.read(path_string) if ARCHIVE is not None else Path(path_string).read_bytes()
        bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("Image decode failed")
        height, width = bgr.shape[:2]
        factor = min(1.0, 640 / max(height, width))
        if factor < 1:
            bgr = cv2.resize(bgr, (round(width * factor), round(height * factor)))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        decoded_hash = hashlib.sha256(rgb.tobytes()).hexdigest()
        grey = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), (32, 32)).astype(np.float32)
        dct = cv2.dct(grey)[:8, :8]
        bits = dct > np.median(dct.ravel()[1:])
        phash = np.packbits(bits).tobytes().hex()
        result = EXTRACTOR.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
        packed = np.zeros((53, 4), np.float32)
        occupied = set()
        for index, hand in enumerate(result.hand_landmarks):
            category = result.handedness[index][0]
            offset = 0 if category.category_name.lower() == "left" else 21
            if offset in occupied:
                offset = 21 - offset
            occupied.add(offset)
            packed[offset:offset + 21, :3] = [[p.x, p.y, p.z] for p in hand]
            packed[offset:offset + 21, 3] = max(0.5, float(category.score))
        features, _ = hand_features(packed[None])
        row = {"path": relative, "label": label, "sha256": decoded_hash, "phash": phash,
               "detected": bool(len(features)), "hands": len(result.hand_landmarks)}
        if len(features):
            row["features"] = features[0]
            row["landmarks"] = packed
        return row
    except Exception as error:
        return {"path": relative, "label": label, "detected": False, "error": str(error)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("images", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--max-per-class", type=int, default=0, help="0 processes every image; a fixed cap speeds prototype runs")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    completed_path = args.output / "COMPLETE.json"
    if completed_path.exists():
        print(completed_path.read_text(), flush=True)
        return
    allowed = set("ABCDEFGHIKLMNOPQRSTUVWXY")  # J and Z require motion.
    items = []
    archive_path = str(args.images) if args.images.suffix.lower() == ".zip" else None
    if archive_path:
        with zipfile.ZipFile(archive_path) as archive:
            candidates = [Path(name) for name in sorted(archive.namelist())]
    else:
        candidates = sorted(args.images.rglob("*"))
    for path in candidates:
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        label = path.parent.name.upper()
        if label in allowed:
            relative = str(path) if archive_path else str(path.relative_to(args.images))
            items.append((str(path), relative, label))
    if not items:
        raise ValueError("No class-folder alphabet images found")
    if args.max_per_class:
        rng = np.random.default_rng(8001)
        selected = []
        for label in sorted(allowed):
            candidates = [item for item in items if item[2] == label]
            if len(candidates) > args.max_per_class:
                chosen = rng.choice(len(candidates), args.max_per_class, replace=False)
                candidates = [candidates[index] for index in sorted(chosen)]
            selected.extend(candidates)
        items = selected
    print(f"input_images={len(items)} workers={args.workers}", flush=True)
    start = time.monotonic()
    audit = []
    rows = []
    shard = 0

    def flush() -> None:
        nonlocal rows, shard
        if not rows:
            return
        path = args.output / f"shard_{shard:05d}.npz"
        np.savez_compressed(path, features=np.stack([r["features"] for r in rows]),
                            landmarks=np.stack([r["landmarks"] for r in rows]),
                            paths=np.asarray([r["path"] for r in rows]),
                            labels=np.asarray([r["label"] for r in rows]),
                            sha256=np.asarray([r["sha256"] for r in rows]),
                            phash=np.asarray([r["phash"] for r in rows]))
        rows = []
        shard += 1

    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker,
                             initargs=(str(args.model), archive_path)) as pool:
        iterator = iter(items)
        pending = {}
        for _ in range(args.workers * 3):
            item = next(iterator, None)
            if item is not None:
                pending[pool.submit(extract_one, item)] = item
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                pending.pop(future)
                row = future.result()
                audit.append({k: v for k, v in row.items() if k not in {"features", "landmarks"}})
                if row["detected"]:
                    rows.append(row)
                if len(rows) >= 512:
                    flush()
                if len(audit) % 500 == 0:
                    elapsed = time.monotonic() - start
                    print(f"processed={len(audit)}/{len(items)} detected={sum(r['detected'] for r in audit)} elapsed={elapsed:.1f}s", flush=True)
                item = next(iterator, None)
                if item is not None:
                    pending[pool.submit(extract_one, item)] = item
    flush()
    (args.output / "extraction_manifest.json").write_text(json.dumps(audit, indent=2))
    report = {"input_images": len(items), "detected_images": sum(r["detected"] for r in audit),
              "shards": shard, "elapsed_seconds": time.monotonic() - start,
              "classes": sorted(set(item[2] for item in items)),
              "max_per_class": args.max_per_class,
              "excluded_letters": ["J", "Z"]}
    completed_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
