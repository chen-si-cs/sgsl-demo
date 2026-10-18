from __future__ import annotations

import os
import json
import random
import sys
from pathlib import Path

import gradio as gr


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from sgsl_demo.pipeline import RecognitionPipeline, format_result  # noqa: E402


DEFAULT_MODEL = ROOT / "models" / "alphabet_mlp.npz"
if not DEFAULT_MODEL.exists():
    DEFAULT_MODEL = ROOT / "models" / "demo_alphabet_classifier.npz"
MODEL_PATH = Path(os.getenv("SGSL_MODEL_PATH", DEFAULT_MODEL))
PIPELINE = RecognitionPipeline(MODEL_PATH)
EXAMPLE_MANIFEST_PATH = ROOT / "examples" / "random_demo" / "manifest.json"
EXAMPLE_MANIFEST = json.loads(EXAMPLE_MANIFEST_PATH.read_text(encoding="utf-8")) if EXAMPLE_MANIFEST_PATH.exists() else {}
IMAGE_CHOICES = [(str(ROOT / "examples" / "random_demo" / entry["file"]),
                  f"True label: {entry['label']} · random held-out test image")
                 for entry in EXAMPLE_MANIFEST.get("images", [])]
VIDEO_CHOICES = [(str(ROOT / "examples" / "random_demo" / entry["file"]),
                  f"{entry['title']}. {entry['description']}")
                 for entry in EXAMPLE_MANIFEST.get("videos", [])]
FEATURED_ROOT = ROOT / "examples" / "featured"
FEATURED_MANIFEST_PATH = FEATURED_ROOT / "manifest.json"
FEATURED_MANIFEST = json.loads(FEATURED_MANIFEST_PATH.read_text(encoding="utf-8")) if FEATURED_MANIFEST_PATH.exists() else {}
FEATURED_VIDEOS = FEATURED_MANIFEST.get("videos", [])


def choose_random_image():
    return random.choice(IMAGE_CHOICES)


def choose_random_video():
    return random.choice(VIDEO_CHOICES)


def choose_gallery_image(event: gr.SelectData):
    return IMAGE_CHOICES[int(event.index)]


def run_video(video_path: str, threshold: float, playback_speed: float = .5):
    result = PIPELINE.predict_video(video_path, threshold, sample_hz=10, playback_speed=playback_speed)
    headline, scores, details = format_result(result)
    label = result.sequence_text or (result.prediction.label if result.prediction.accepted else "")
    if result.timeline:
        rows = [[round(token["time_seconds"], 2), token["label"], f"{token['confidence']:.1%}"]
                for token in result.timeline]
    else:
        rows = [["", name, f"{score:.1%}"] for name, score in scores.items()]
    return result.preview_path, headline, rows, details, label


def run_image(image, threshold: float):
    overlay, result = PIPELINE.predict_image(image, threshold)
    headline, scores, details = format_result(result)
    label = result.prediction.label if result.prediction.accepted else ""
    return overlay, headline, scores, details, label


def append_letter(current: str, label: str) -> str:
    if not label or label in {"UNCERTAIN", "NO_HAND"}:
        return current
    return f"{current}{label}"


CSS = """
.demo-banner {border: 1px solid #f1b24b; background: #fff8e8; border-radius: 14px; padding: 14px 16px;}
.result-card {font-size: 1.45rem; font-weight: 750; color: #073b67;}
.small-note {color: #526578; font-size: .92rem;}
"""


with gr.Blocks(title="SgSL Landmark Demo") as demo:
    gr.Markdown(
        "# SgSL landmark recognition prototype\n"
        "MediaPipe hands + upper body → normalized landmark sequence → tiny CPU classifier"
    )
    metadata = PIPELINE.classifier.metadata
    if metadata.get("kind") == "landmark_mlp_v1":
        gr.HTML(
            "<div class='demo-banner'><b>24-letter alphabet demo</b> · "
            f"Validation {metadata['val_accuracy']:.1%} · Held-out image test {metadata['test_accuracy']:.1%}. "
            "Scores apply to detected hands; dataset detection coverage was 75.7%. "
            "Trained from public ASL alphabet images. Hold each letter briefly; J and Z require a motion model. "
            "Upload non-sensitive demo media only; processing happens on the server.</div>"
        )
    else:
        gr.HTML(
            "<div class='demo-banner'><b>Technical smoke checkpoint:</b> an overfit alphabet model "
            "for verifying the pipeline. Upload non-sensitive demo media only; processing happens on the server.</div>"
        )
    with gr.Row():
        threshold = gr.Slider(0.0, 0.95, value=0.65, step=0.05, label="Accept prediction above")
        transcript = gr.Textbox(label="Confirmed letters / word", placeholder="Accepted letters appear here")
        clear = gr.Button("Clear word")

    with gr.Tabs():
        with gr.Tab("Video / webcam"):
            playback_speed = gr.Slider(.25, 1, value=.5, step=.25, label="Preview playback speed (inference samples source at 10 Hz)")
            with gr.Row():
                video = gr.Video(sources=["upload", "webcam"], label="Record or upload one held sign")
                preview = gr.Video(label="Landmarks + current letter + composing characters", interactive=False)
            video_example_note = gr.Textbox(label="Selected example", interactive=False,
                                            placeholder="Upload your own clip or select an example below.")
            with gr.Row():
                video_run = gr.Button("Recognise clip", variant="primary")
                random_video = gr.Button("🎲 Random clip", visible=bool(VIDEO_CHOICES))
            video_headline = gr.Markdown(elem_classes=["result-card"])
            video_scores = gr.Dataframe(headers=["Time (s)", "Letter", "Confidence"],
                                        datatype=["str", "str", "str"],
                                        interactive=False, label="Stable predictions")
            video_details = gr.Markdown(elem_classes=["small-note"])
            video_label = gr.Textbox(visible=False)
            add_video = gr.Button("Add accepted letter to word")
            if FEATURED_VIDEOS:
                gr.Markdown("### Presentation examples\nLoad a raw clip and press **Recognise clip** for fresh CPU inference. "
                            "The clip may contain fingerspelling and word signs. Output is alphabet-model predictions, not sentence translation.")
                gr.Examples([[str(FEATURED_ROOT / entry["file"]),
                              f"{entry['title']}. {entry['description']}"] for entry in FEATURED_VIDEOS],
                            inputs=[video, video_example_note], label="Load a presentation example",
                            example_labels=[entry["title"] for entry in FEATURED_VIDEOS])
                with gr.Accordion("Processed version for slides", open=False):
                    for entry in FEATURED_VIDEOS:
                        processed = FEATURED_ROOT / entry["processed_file"]
                        if processed.exists():
                            gr.Video(value=str(processed), label=f"Offline output: {entry['title']}", interactive=False)
                            gr.DownloadButton("Download processed MP4", value=str(processed))
            primary_video = ROOT / "examples" / "heldout" / "heldout_CAB_montage.mp4"
            if primary_video.exists() and metadata.get("kind") == "landmark_mlp_v1":
                gr.Markdown("Example: **C → A → B**, a montage of held-out still images with blank gaps.")
                gr.Examples([[str(primary_video), "CAB · curated held-out still-image montage with blank gaps."]],
                            inputs=[video, video_example_note], label="Starter demo", example_labels=["CAB · starter montage"])
            elif (ROOT / "examples" / "example_C.mp4").exists():
                gr.Examples([[str(ROOT / "examples" / "example_C.mp4")]], inputs=[video])
            if VIDEO_CHOICES:
                gr.Markdown("### More clips\nSelect a clip, then press **Recognise clip**, or use **Random clip** to select and run one.")
                gr.Examples([list(choice) for choice in VIDEO_CHOICES],
                            inputs=[video, video_example_note], examples_per_page=8,
                            label="Letter montages and real webcam footage",
                            example_labels=[entry["title"] for entry in EXAMPLE_MANIFEST["videos"]])

        with gr.Tab("Image baseline"):
            with gr.Row():
                image = gr.Image(sources=["upload", "webcam"], type="numpy", label="Upload or capture a hand sign")
                image_overlay = gr.Image(label="Landmark overlay", interactive=False)
            image_example_note = gr.Textbox(label="Selected example", interactive=False,
                                            placeholder="Upload your own image or select a labeled sample below.")
            with gr.Row():
                image_run = gr.Button("Recognise image", variant="primary")
                random_image = gr.Button("🎲 Random image", visible=bool(IMAGE_CHOICES))
            image_headline = gr.Markdown(elem_classes=["result-card"])
            image_scores = gr.Label(num_top_classes=5, label="Top scores")
            image_details = gr.Markdown(elem_classes=["small-note"])
            image_label = gr.Textbox(visible=False)
            add_image = gr.Button("Add accepted letter to word")
            if metadata.get("kind") == "landmark_mlp_v1":
                image_examples = [ROOT / "examples" / "heldout" / f"correct_{label}_{label}.jpg" for label in "CABLY"]
            else:
                image_examples = [ROOT / "examples" / name for name in ("example_B.jpg", "example_C.jpg", "example_Y.jpg")]
            image_examples = [[str(path)] for path in image_examples if path.exists()]
            if image_examples:
                starter_rows = [[row[0], f"True label: {Path(row[0]).stem.split('_')[-1]} · curated starter image"]
                                for row in image_examples]
                gr.Examples(starter_rows, inputs=[image, image_example_note], label="Starter examples")
            if IMAGE_CHOICES:
                gr.Markdown("### Random test images\nOne sample per letter, selected without filtering by prediction or confidence. The true label is for comparison only.")
                image_gallery = gr.Gallery(
                    value=[(path, f"True {entry['label']}")
                           for (path, _), entry in zip(IMAGE_CHOICES, EXAMPLE_MANIFEST["images"])],
                    label="24 randomly selected held-out images · click a thumbnail",
                    columns=6, rows=2, height=460, allow_preview=False, interactive=False,
                )

        with gr.Tab("What is real here?"):
            gr.Markdown(
                """
### Implemented

- Official Google MediaPipe hand and pose landmarkers run on CPU.
- The current Gradio demo sends uploaded media to the Space backend, where MediaPipe runs; browser-local extraction from the proposal is future work.
- The trained alphabet MLP uses wrist-relative coordinates, pairwise landmark distances and finger angles.
- Source video is sampled on a fixed 10 Hz source-time grid. Each sample uses its highest-probability class; at least two confident samples must agree in a trailing 0.3-second window, with a two-thirds majority and a class-score margin. A lone confidence spike cannot emit a letter. A no-hand gap lets you repeat a letter.
- Preview defaults to 0.5x playback for inspection. Slower playback does not create extra source frames or remove motion blur; the source-time clock remains visible.
- Video output preserves source timing: a top-left landmark-only inset, top-right current letter and bottom incremental character sequence share the same classifier/decoder as the exported presentation video. This is offline processing, not a live-speed claim.
- Low-confidence results abstain instead of silently adding a character.
- Custom isolated-sign clips can be extracted and trained with the included scripts.

### Not claimed

- This is a within-source ASL image evaluation. Signer identities are unknown; it does not measure SgSL word recognition or unseen-signer performance.
- The trained static alphabet model excludes J and Z, which require motion. The old smoke checkpoint has its own smaller vocabulary.
- The NTU–SADeaf Sign Bank is a vocabulary reference, not bundled training data.
- Use non-sensitive demo recordings only; Gradio may create temporary server-side files.
"""
            )

    video_run.click(
        run_video,
        inputs=[video, threshold, playback_speed],
        outputs=[preview, video_headline, video_scores, video_details, video_label],
    )
    image_run.click(
        run_image,
        inputs=[image, threshold],
        outputs=[image_overlay, image_headline, image_scores, image_details, image_label],
    )
    random_video.click(choose_random_video, outputs=[video, video_example_note]).then(
        run_video, inputs=[video, threshold, playback_speed],
        outputs=[preview, video_headline, video_scores, video_details, video_label],
    )
    random_image.click(choose_random_image, outputs=[image, image_example_note]).then(
        run_image, inputs=[image, threshold],
        outputs=[image_overlay, image_headline, image_scores, image_details, image_label],
    )
    if IMAGE_CHOICES:
        image_gallery.select(choose_gallery_image, outputs=[image, image_example_note]).then(
            run_image, inputs=[image, threshold],
            outputs=[image_overlay, image_headline, image_scores, image_details, image_label],
        )
    add_video.click(append_letter, inputs=[transcript, video_label], outputs=transcript)
    add_image.click(append_letter, inputs=[transcript, image_label], outputs=transcript)
    clear.click(lambda: "", outputs=transcript)


if __name__ == "__main__":
    demo.launch(css=CSS)
