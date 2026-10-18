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

The model uses 288 hand features (wrist-relative coordinates, 210 pairwise distances, 15 joint-angle cosines), followed by a 256 → 128 → 24 MLP. Training used PyTorch on GPU; serving uses NumPy on CPU. Scalar temperature calibration used validation data. Video inference now uses a fixed **10 Hz source-time grid**, picks each sample's highest-probability class and confirms it using a causal trailing **0.3-second consensus window**: at least two confident samples, a two-thirds majority, mean score above the confidence threshold, and at least a 0.10 margin over the runner-up. The report records the highest-confidence evidence frame for each emitted token. A single peak cannot override inconsistent frames. Repeated letters require a 0.3-second no-hand gap. Clips requiring more than 600 samples must be shortened rather than silently undersampled.

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

On Windows, use this command to keep the local server running in the background after the launching terminal or assistant turn ends:

```powershell
powershell -File .\sgsl-demo\scripts\start_local_demo.ps1
```

The launcher checks the existing server, opens no extra terminal window, and writes diagnostics under `sgsl-demo/.logs/`. If a cached browser page shows an error after the local process stops, rerun the launcher and refresh the page. Hugging Face Spaces manages its own app process and does not use this Windows launcher.

Open the Gradio URL. In **Video / webcam**, click the supplied example and **Recognise clip**. The stable letter sequence should be **CAB**; press **Add accepted letter to word** to place it in the message field. This example is a montage of three held-out still images with blank gaps, not a real continuous signing performance. The **Image baseline** tab has held-out single-letter examples.

The example picker also includes **24 uniformly sampled held-out images** (one per static letter, seed 20261018), **five constructed clips** (HI, YES, HELLO, plus two random held letters), and **three real six-second webcam excerpts**. Sampling does not filter by model score or correctness, so errors and abstentions can appear. Each example displays its true label or identifies unannotated real footage. The **Random image** and **Random clip** buttons choose and immediately run an example. Picker labels are not inputs to the classifier. See `examples/random_demo/manifest.json` for provenance.

The **Presentation examples** picker includes `examples/featured/sgsl_bochap_04_10_consensus_excerpt.mp4`: source seconds **4-10** of Lily Goh / ExtraOrdinary Horizons' *#SgSL Sign for Bochap* (YouTube `LxdjT1g3REc`). Click **Recognise clip** for fresh CPU inference. The **Processed version for slides** accordion previews and downloads its processed MP4. The preview defaults to **0.5x playback**: the six-second source plays for twelve seconds, with pitch-preserving slowed audio and an explicitly source-time HUD clock. Change the playback slider for 0.25x/0.5x/0.75x/1x. Playback speed does not change which source frames are inferred, create extra evidence, or remove blur.

Also included: Deaf.SG's *Singapore Sign Language (SgSL) Lesson: Alphabet - A to Z* (YouTube `6EW4PVT_d_4`). **ABCD opening** is an explicitly curated successful 0-6.5 second excerpt, exported as `deafsg_abcd_10hz_processed.mp4` (13 seconds at 0.5x). A fresh run emits A, B, C, D at source times 0.1, 2.0, 3.7, 5.6 s with the unchanged checkpoint and decoder. This is a selected demo success, not a general accuracy result. The **full lesson** remains available as `deafsg_alphabet_10hz_processed.mp4` (51.067 source seconds / 102.133 playback seconds); its output is `ABCDOCEFGHHMIAKLREKAOQRESXRVWXA`, with substitutions and extra letters. Hand detection coverage is 90.4% for the full run and 100% for the curated opening; detection coverage is not recognition accuracy. J/Z remain unsupported, and original printed alphabet letters are references only, not prediction inputs. Each example has a separate complete score/provenance report.

HF and offline exports use the same prediction/decoder path: top-left simplified detected skeleton, top-right current frame's thresholded top-1 class, and bottom confirmed incremental characters. At threshold 0.65 the new decoder emits **BECHANV**, recovering H but still confusing the P-like handshape with N and adding V during the later gesture. The user's expected reference **BECHAP** is not fed into inference; the original source title spells the word *Bochap*. No manual word correction is applied. `sgsl_bochap_04_10_consensus_processed.predictions.json` contains all 60 frame-score distributions, consensus/peak evidence, source/playback timestamps and provenance. This is not sentence translation. Older clips remain intact locally but are not selected or included in the new update ZIP. See third-party notices before public redistribution.

A frozen audit of all 180 original frames is summarised in `training_results/bochap_04_10/sampling_comparison.json`:

| Sampling / decoder | Output |
|---|---|
| Original 72 uneven samples / old dwell | BECANV |
| Exact 10 Hz / old dwell | BEANV |
| Exact 10 Hz / time-window consensus | BECHANV |

In the audited source-relative interval 3.8-4.6 s, P's maximum model probability is 1.42%, versus N's 88.95%. That is a classifier-level confusion, not just a missed sample. These diagnostics do not establish whether image blur caused the classifier error, and this one-clip comparison is not a new accuracy benchmark.

For updated HF deployment, extract the latest update ZIP into the existing Space root, including `app.py`, all included `src/sgsl_demo/` modules, the manifest and referenced files in `examples/featured/`, and README/notices. Keep root `packages.txt` and all existing model/source files. CPU Basic remains sufficient; no new serving dependency, checkpoint, or GPU is required. HF does not need `data/offline/` or the training scripts. These files are prepared locally; no existing Space is updated automatically. The complete lesson took about 71 seconds to process/export locally; use the short ABCD example for a quicker CPU demo. This is not a live-speed guarantee.

Webcam use records a clip and submits it when you press the button; it is not continuous live streaming. The current Gradio version sends media to the server for MediaPipe processing, so use non-sensitive demo recordings. Browser-local landmark extraction from the proposal remains future work.

## Upload to Hugging Face

Create a Gradio Space using free CPU hardware and upload this repository's files, including `models/`, `src/`, `examples/`, and the root-level `packages.txt`. README metadata configures the app. Spaces installs the Linux libraries from `packages.txt` at build time; MediaPipe needs `libgles2` and `libegl1` even when using its CPU delegate. The installed `imageio-ffmpeg` package provides browser-playable H.264 previews. No GPU, PyTorch or external API key is required for serving.

If a Space raises `OSError: libGLESv2.so.2: cannot open shared object file`, upload `packages.txt` beside `app.py` and commit the change to trigger a rebuild. A runtime restart alone does not install missing system packages.

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

### Offline slide-video captions

`python -m sgsl_demo.offline_captions input.mp4 captions.json output.mp4` burns timed captions into a reserved center-bottom strip and preserves the signing image and any source audio. It exports H.264 MP4 and a timing/provenance receipt. This is an **offline renderer, not a sign translator or live-speed benchmark**. Install the `offline` optional dependencies for the authoring tools; they are not needed by the HF app.

Caption JSON must identify its provenance:

```json
{
  "kind": "reference",
  "source": "Human-reviewed timed translation",
  "attribution": "Video creator / source",
  "segments": [{"start": 0.0, "end": 2.0, "text": "Example reference caption"}]
}
```

Use `kind: prediction` only for actual inference outputs and additionally identify `model`. The video visibly distinguishes reference captions from model predictions. Reference labels and source subtitles must not be passed off as predictions.

The public NTU-SADeaf Sign Bank publishes isolated vocabulary entries; its FAQ says these were extracted from sentence recordings, but that does **not** establish that the original labelled sentence corpus is publicly downloadable. The linked YouTube narrative is not covered by the alphabet checkpoint's training task. A sentence-recognition demo remains blocked on approved labelled phrase/sentence training clips and evaluation on separate recordings. Keep downloaded media under ignored `data/offline/`; do not add it to public HF/GitHub examples without checking redistribution rights.

- `app.py`: Gradio interface, example selection and message assembly.
- `src/sgsl_demo/landmarks.py`: official MediaPipe hand/pose extraction and H.264 overlays.
- `src/sgsl_demo/static_features.py`: hand geometry shared by training and CPU inference.
- `src/sgsl_demo/mlp_classifier.py`: NumPy MLP inference and temporal averaging.
- `src/sgsl_demo/decoder.py`: confidence gating and stable-letter decoding.
- `scripts/extract_alphabet.py`: bounded parallel extraction from the public dataset ZIP.
- `scripts/train_landmark_mlp.py`: frozen duplicate-group splits, GPU training, calibration and CPU export.
- `models/demo_alphabet_classifier.npz`: older overfit smoke fixture retained for regression checks.

This is a low-risk classroom prototype. The app supports correction and typed fallback. See `THIRD_PARTY_NOTICES.md` for source attribution and data licenses.
