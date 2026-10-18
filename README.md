---
title: SgSL Landmark Demo
emoji: 🤟
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 6.30.0
app_file: app.py
pinned: false
license: mit
python_version: "3.11"
---

# SgSL landmark recognition prototype

A working alphabet demonstration: **image/video/webcam → MediaPipe hands and upper body → tiny landmark MLP → stable letters → word**. The app automatically loads the trained CPU checkpoint at `models/alphabet_mlp.npz`.

## Trained result

The first model was trained on **ws3, physical GPU 0**, the first idle GPU, on 18 October 2026. It uses the public [American Sign Language 0–9 / A–Z dataset](https://www.kaggle.com/datasets/prathumarikeri/american-sign-language-09az), one of the sources cited by the [GloriaNeo SgSL reference project](https://github.com/GloriaNeo/SgSL-Sign-Language-Recognition).

| Measurement | Result |
|---|---:|
| Static alphabet classes | 24; J/Z excluded |
| Selected source images | 36,000; fixed seed, 1,500/class |
| Usable MediaPipe hands | 27,244 / 36,000 = 75.7% |
| Exact decoded duplicates removed | 9,594 |
| Training / validation / test images | 12,334 / 2,656 / 2,649 |
| Validation accuracy | 99.28% |
| Held-out test accuracy | 98.94% |
| Held-out macro-F1 | 98.37% |
| MLP parameters | 109,976 |
| CPU checkpoint | 413,281 bytes |

**Accuracy is conditional on a usable detected hand.** Images were split before fitting; exact decoded duplicates were removed and identical perceptual hashes kept in the same split. The scores describe this public ASL image source. Signer identities are unknown, nearby frames with different perceptual hashes can remain correlated, and the result does not establish signer-independent SgSL accuracy or word-level sign recognition. The curated example gallery is for demonstration and is separate from the aggregate test report.

The model uses 288 hand features (wrist-relative coordinates, 210 pairwise distances, 15 joint-angle cosines), followed by a 256 → 128 → 24 MLP. Training used PyTorch on GPU; serving uses NumPy on CPU. Scalar temperature calibration used validation data. Video inference averages frame evidence, smooths predictions and requires a stable hold before emitting a letter. Repeated letters require a brief no-hand gap.

See `training_results/ws3_alphabet_v1/report.json`, `history.json` and `interface_check.json` for measured results and interface checks. The full split manifest remains on the server.

## Run locally

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

For the existing workspace environment, run from the parent GP8001 directory:

```powershell
.\.venv\Scripts\python.exe .\sgsl-demo\app.py
```

Open the Gradio URL. In **Video / webcam**, click the supplied example and **Recognise clip**. The stable letter sequence should be **CAB**; press **Add accepted letter to word** to place it in the message field. This example is a montage of three held-out still images with blank gaps, not a real continuous signing performance. The **Image baseline** tab has held-out single-letter examples.

The example picker also includes **24 uniformly sampled held-out images** (one per static letter, seed 20261018), **five constructed clips** (HI, YES, HELLO, plus two random held letters), and **three real six-second webcam excerpts**. Sampling does not filter by model score or correctness, so errors and abstentions can appear. Each example displays its true label or identifies unannotated real footage. The **Random image** and **Random clip** buttons choose and immediately run an example. Picker labels are not inputs to the classifier. See `examples/random_demo/manifest.json` for provenance.

Webcam use records a clip and submits it when you press the button; it is not continuous live streaming. The current Gradio version sends media to the server for MediaPipe processing, so use non-sensitive demo recordings. Browser-local landmark extraction from the proposal remains future work.

## Upload to Hugging Face

Create a Gradio Space using free CPU hardware and upload this repository's files, including `models/`, `src/`, and `examples/`. README metadata configures the app. The installed `imageio-ffmpeg` package provides browser-playable H.264 previews. No GPU, PyTorch or external API key is required for serving.

## Remote training layout

Code, environments, runs and data are kept apart:

```text
/data/users/chen/gp8001/
  code/sgsl_20261018_v1/        source code and model assets
  envs/landmark/               private Python environment
  runs/alphabet_20261018_v1/    checkpoints, frozen split, metrics, examples
/data/cold_storage/chen/gp8001/
  raw/asl_09az/                official dataset ZIP and download receipt
  landmarks/asl_09az_v1/       cached features, extraction audit and completion receipt
```

The raw ZIP is read directly to avoid expanding 142,261 small files on shared HDD storage. A partial image expansion from initial preparation is retained but is not used by the run. Training selects the first GPU without compute processes and with under 512 MiB memory use; it does not stop other processes.

`scripts/run_ws3_alphabet.sh` reproduces the prepared run. Its defaults use a fixed 1,500-image cap per static class; use `--max-per-class 0` in the extraction command to process every source image. GPU training dependencies are private on the server and are intentionally absent from the CPU serving requirements.

## Train actual SgSL word signs

Use approved, labeled isolated clips and a small vocabulary reviewed by fluent SgSL users. The NTU–SADeaf Sign Bank is a vocabulary reference; this repository does not assume permission to scrape or redistribute its videos.

```text
data/clips/
  HELLO/signer01_take01.mp4
  HELLO/signer02_take01.mp4
  THANK_YOU/signer01_take01.mp4
  THANK_YOU/signer02_take01.mp4
```

```powershell
$env:PYTHONPATH = ".\src"
.\.venv\Scripts\python.exe scripts\extract_dataset.py data\clips data\landmarks
.\.venv\Scripts\python.exe scripts\train.py data\landmarks models\custom\presentation.npz
$env:SGSL_MODEL_PATH = ".\models\custom\presentation.npz"
.\.venv\Scripts\python.exe app.py
```

The clip-training script is a separate lightweight temporal baseline. Start with 3–5 signs and 10–20 clips per sign; hold out signers where possible. Describe a one-person recording model as person-specific. A strong alphabet result alone does not validate word recognition.

## Key files

- `app.py`: Gradio interface, example selection and message assembly.
- `src/sgsl_demo/landmarks.py`: official MediaPipe hand/pose extraction and H.264 overlays.
- `src/sgsl_demo/static_features.py`: hand geometry shared by training and CPU inference.
- `src/sgsl_demo/mlp_classifier.py`: NumPy MLP inference and temporal averaging.
- `src/sgsl_demo/decoder.py`: confidence gating and stable-letter decoding.
- `scripts/extract_alphabet.py`: bounded parallel extraction from the public dataset ZIP.
- `scripts/train_landmark_mlp.py`: frozen duplicate-group splits, GPU training, calibration and CPU export.
- `models/demo_alphabet_classifier.npz`: older overfit smoke fixture retained for regression checks.

This is a low-risk classroom prototype. The app supports correction and typed fallback. See `THIRD_PARTY_NOTICES.md` for source attribution and data licenses.
