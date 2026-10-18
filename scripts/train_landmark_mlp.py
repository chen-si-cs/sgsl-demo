"""Train a small hand-landmark MLP on one explicitly selected CUDA GPU."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import torch
from torch import nn


def split_groups(labels, hashes, phashes, paths, seed=8001):
    """Freeze splits before fitting, with exact/perceptual duplicates grouped."""
    rng = np.random.default_rng(seed)
    splits = np.full(len(labels), "", dtype="U5")
    parent = list(range(len(labels)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for values in (hashes, phashes):
        seen = {}
        for index, value in enumerate(values):
            key = str(value)
            if key in seen:
                parent[find(index)] = find(seen[key])
            else:
                seen[key] = index
    grouped = {}
    for index in range(len(labels)):
        grouped.setdefault(find(index), []).append(index)
    for label in sorted(set(labels)):
        groups = [indices for indices in grouped.values() if labels[indices[0]] == label]
        # Ambiguous duplicate groups carrying conflicting labels are excluded.
        groups = [indices for indices in groups if len(set(labels[indices])) == 1]
        rng.shuffle(groups)
        total = sum(map(len, groups))
        targets = {"train": total * 0.70, "val": total * 0.15, "test": total * 0.15}
        counts = {name: 0 for name in targets}
        # Allocate whole duplicate groups while preserving approximate proportions.
        for indices in sorted(groups, key=len, reverse=True):
            selected = max(targets, key=lambda name: (targets[name] - counts[name]) / targets[name])
            splits[indices] = selected
            counts[selected] += len(indices)
    for values in (hashes, phashes):
        used = {}
        for value, split in zip(values, splits):
            if not split:
                continue
            if str(value) in used and used[str(value)] != split:
                raise ValueError("Duplicate leakage across splits")
            used[str(value)] = split
    return splits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("features", type=Path)
    parser.add_argument("run", type=Path)
    parser.add_argument("--epochs", type=int, default=240)
    parser.add_argument("--seed", type=int, default=8001)
    args = parser.parse_args()
    args.run.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU training requested but CUDA is unavailable")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES to exactly one selected idle GPU")
    print(f"GPU={torch.cuda.get_device_name(0)} visible={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
    parts = []
    for path in sorted(args.features.glob("shard_*.npz")):
        with np.load(path, allow_pickle=False) as data:
            parts.append({key: data[key].copy() for key in ("features", "paths", "labels", "sha256", "phash")})
    all_data = {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
    order = np.argsort(all_data["paths"])
    all_data = {key: value[order] for key, value in all_data.items()}
    # Exact decoded-image duplicates are removed before splitting.
    _, unique = np.unique(all_data["sha256"], return_index=True)
    unique.sort()
    duplicate_count = len(order) - len(unique)
    all_data = {key: value[unique] for key, value in all_data.items()}
    labels = all_data["labels"]
    splits = split_groups(labels, all_data["sha256"], all_data["phash"], all_data["paths"], args.seed)
    classes = sorted(set(labels[splits != ""]))
    lookup = {label: index for index, label in enumerate(classes)}
    y = np.asarray([lookup.get(label, 0) for label in labels])
    masks = {name: splits == name for name in ("train", "val", "test")}
    for name, mask in masks.items():
        if set(labels[mask]) != set(classes):
            raise ValueError(f"{name} is missing classes: {set(classes) - set(labels[mask])}")
    manifest = [{"path": str(path), "label": str(label), "split": str(split),
                 "sha256": str(sha), "phash": str(ph)}
                for path, label, split, sha, ph in zip(all_data["paths"], labels, splits, all_data["sha256"], all_data["phash"])]
    manifest_text = json.dumps(manifest, indent=2)
    (args.run / "split_manifest.json").write_text(manifest_text)
    split_digest = hashlib.sha256(manifest_text.encode()).hexdigest()
    mean = all_data["features"][masks["train"]].mean(0)
    scale = all_data["features"][masks["train"]].std(0)
    scale[scale < 1e-4] = 1
    normalized = np.clip((all_data["features"] - mean) / scale, -8, 8)
    tensors = {name: (torch.tensor(normalized[mask], device="cuda"),
                      torch.tensor(y[mask], device="cuda", dtype=torch.long))
               for name, mask in masks.items()}
    model = nn.Sequential(nn.Linear(normalized.shape[1], 256), nn.ReLU(), nn.Dropout(0.15),
                          nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.10),
                          nn.Linear(128, len(classes))).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0015, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.02)
    best_score, best_loss, best_epoch, best_state = -1, float("inf"), 0, None
    history = []
    start = time.monotonic()
    train_x, train_y = tensors["train"]
    val_x, val_y = tensors["val"]
    for epoch in range(1, args.epochs + 1):
        model.train()
        permutation = torch.randperm(len(train_x), device="cuda")
        total_loss = 0.0
        for offset in range(0, len(train_x), 512):
            indices = permutation[offset:offset + 512]
            x = train_x[indices]
            x = x + torch.randn_like(x) * 0.015
            logits = model(x)
            loss = criterion(logits, train_y[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(indices)
        scheduler.step()
        model.eval()
        with torch.no_grad():
            val_logits = model(val_x)
            val_loss = float(nn.functional.cross_entropy(val_logits, val_y))
            val_accuracy = float((val_logits.argmax(1) == val_y).float().mean())
        row = {"epoch": epoch, "train_loss": total_loss / len(train_x),
               "val_accuracy": val_accuracy, "val_loss": val_loss,
               "elapsed_seconds": time.monotonic() - start}
        history.append(row)
        if val_accuracy > best_score or (val_accuracy == best_score and val_loss < best_loss):
            best_score, best_loss, best_epoch = val_accuracy, val_loss, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            torch.save({"state_dict": best_state, "epoch": epoch, "classes": classes,
                        "mean": mean, "scale": scale, "split_sha256": split_digest}, args.run / "best.pt")
        if epoch % 10 == 0 or epoch == 1:
            print(json.dumps(row), flush=True)
            (args.run / "history.json").write_text(json.dumps(history, indent=2))
        if epoch - best_epoch >= 45:
            print(f"early_stop epoch={epoch} best={best_epoch}", flush=True)
            break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = {name: model(x).cpu().numpy() for name, (x, _) in tensors.items()}
    # Scalar temperature selected using validation NLL, never the test labels.
    val_targets = y[masks["val"]]
    temperatures = np.geomspace(0.5, 3.0, 41)
    nlls = []
    for temperature in temperatures:
        z = logits["val"] / temperature
        z -= z.max(1, keepdims=True)
        log_probs = z - np.log(np.exp(z).sum(1, keepdims=True))
        nlls.append(-float(log_probs[np.arange(len(z)), val_targets].mean()))
    temperature = float(temperatures[int(np.argmin(nlls))])
    extraction = json.loads((args.features / "COMPLETE.json").read_text())
    report = {"source": "prathumarikeri/american-sign-language-09az", "license": "CC BY-SA 4.0",
              "task": "24 static ASL-based fingerspelling letters; J/Z excluded",
              "split_protocol": "70/15/15 image split grouped by exact perceptual hash; exact decoded duplicates removed",
              "signer_independent": False, "split_sha256": split_digest,
              "exact_duplicates_removed": duplicate_count, "extraction": extraction,
              "classes": classes, "best_epoch": best_epoch, "epochs_run": len(history),
              "temperature": temperature, "parameter_count": sum(p.numel() for p in model.parameters()),
              "training_seconds": time.monotonic() - start,
              "gpu": torch.cuda.get_device_name(0), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "limitations": "One public ASL image source; unknown signer identities; near duplicates with unequal hashes can remain; not SgSL word or unseen-signer evidence."}
    for name in masks:
        truth = y[masks[name]]
        predicted = logits[name].argmax(1)
        report[name] = {"samples": len(truth), "accuracy": float(accuracy_score(truth, predicted)),
                        "classification_report": classification_report(truth, predicted, target_names=classes, output_dict=True, zero_division=0),
                        "confusion_matrix": confusion_matrix(truth, predicted).tolist()}
    metadata = {"demo_only": False, "kind": "landmark_mlp_v1", "task": report["task"],
                "source": report["source"], "license": report["license"],
                "classes": classes, "split_sha256": split_digest,
                "val_accuracy": report["val"]["accuracy"], "test_accuracy": report["test"]["accuracy"],
                "signer_independent": False, "limitations": report["limitations"]}
    arrays = {"kind": np.asarray("landmark_mlp_v1"), "classes": np.asarray(classes),
              "mean": mean, "scale": scale, "temperature": np.float32(temperature),
              "metadata": np.asarray(json.dumps(metadata))}
    for layer, index in enumerate((0, 3, 6)):
        arrays[f"weight_{layer}"] = best_state[f"{index}.weight"].numpy()
        arrays[f"bias_{layer}"] = best_state[f"{index}.bias"].numpy()
    np.savez_compressed(args.run / "alphabet_mlp.npz", **arrays)
    numpy_logits = normalized[masks["test"]].copy()
    for layer in range(3):
        numpy_logits = numpy_logits @ arrays[f"weight_{layer}"].T + arrays[f"bias_{layer}"]
        if layer < 2:
            numpy_logits = np.maximum(numpy_logits, 0)
    report["cpu_export_max_logit_error"] = float(np.max(np.abs(numpy_logits - logits["test"])))
    if report["cpu_export_max_logit_error"] > 0.02:
        raise ValueError("CPU export disagrees with GPU model")
    report_path = args.run / "report.json"
    report_path.write_text(json.dumps(report, indent=2))
    (args.run / "history.json").write_text(json.dumps(history, indent=2))
    # Record confident correct and incorrect test files for a review gallery.
    z = logits["test"] / temperature
    z -= z.max(1, keepdims=True)
    probs = np.exp(z); probs /= probs.sum(1, keepdims=True)
    examples = []
    test_ids = np.where(masks["test"])[0]
    for row_index, data_index in enumerate(test_ids):
        best = int(probs[row_index].argmax())
        examples.append({"path": str(all_data["paths"][data_index]), "label": str(labels[data_index]),
                         "prediction": classes[best], "confidence": float(probs[row_index, best]),
                         "correct": bool(classes[best] == labels[data_index])})
    (args.run / "test_predictions.json").write_text(json.dumps(examples, indent=2))
    print(json.dumps({key: report[key] for key in ("best_epoch", "parameter_count", "training_seconds")}), flush=True)
    print(json.dumps({name: {key: report[name][key] for key in ("samples", "accuracy")} for name in masks}), flush=True)
    (args.run / "READY.json").write_text(json.dumps({"report": str(report_path), "split_sha256": split_digest,
                                                    "checkpoint": "alphabet_mlp.npz"}, indent=2))


if __name__ == "__main__":
    main()
