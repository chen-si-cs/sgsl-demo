"""Compare frozen frame scores without passing any expected word to a decoder."""
import argparse
import json
from pathlib import Path

import numpy as np

from sgsl_demo.decoder import decode_consistent_letters, decode_letters
from sgsl_demo.landmarks import video_sample_indices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with np.load(args.cache, allow_pickle=False) as data:
        dense = data["probabilities"]
        classes = tuple(data["classes"].tolist())
        fps = float(data["fps"])
    results = []
    for mode in ("legacy_72", "legacy_10hz", "consensus_10hz"):
        indices = video_sample_indices(len(dense), fps, max_frames=72) if mode == "legacy_72" else video_sample_indices(len(dense), fps, sample_hz=10, max_frames=None)
        scores = dense[indices]
        valid = np.flatnonzero(scores.sum(axis=1) > 0)
        if mode.startswith("legacy"):
            tokens, _ = decode_letters(scores[valid], valid, classes, len(indices))
        else:
            tokens, _ = decode_consistent_letters(scores[valid], valid, classes, np.asarray(indices) / fps)
        for token in tokens:
            token["time_seconds"] = indices[token["sample_index"]] / fps
        results.append({"mode": mode, "samples": len(indices), "transcript": "".join(t["label"] for t in tokens),
                        "timeline": tokens})
    # A global peak is diagnostic only; no target word selects the emitted label.
    tail = dense[round(3.8 * fps):round(4.6 * fps)]
    report = {"comparison": results, "window_3_8_to_4_6_seconds_max_class_probability":
              {label: float(tail[:, i].max()) for i, label in enumerate(classes)}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
