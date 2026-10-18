# Offline sentence demo: prepared, not trained

## Follow-up: requested landmark / character HUD

The follow-up request uses only source seconds 20-26. Raw and processed six-second clips are packaged in `examples/featured/`, with source manifest and actual frame-prediction/timeline JSON. The Gradio app loads the **raw** excerpt as a featured example and offers the processed MP4 for download. Newly processed video uploads use the same HUD code.

At the unchanged threshold 0.65 / dwell 3 sampled frames, the decoder emits **A** at 2.4 seconds, with 86.1% hand-detection coverage over 72 sampled frames. The output preserves 150 source frames at 25 fps, with no manually supplied sentence or word labels. This is not phrase recognition. The display is offline time-synchronised, not measured live inference. All 18 tests pass, including future-token leakage, missing-landmark handling and timing tests. The new rendered preview was visually inspected.

The MediaPipe+CPU classification+export took approximately 6.6 seconds on this local machine for the six-second clip; this single measurement does not establish live throughput. No new training was run and no HF upload was performed. The final test run passes 19 tests after adding a one-token headline regression test. The raw featured clip was loaded through the browser picker, and Gradio API inference independently returned the same A token at 2.4 s. `output/sgsl_hf_featured_update.zip` in the parent workspace is the root-layout update bundle; it excludes environments, private downloads and model weights already present in the existing Space.

## Completed

- Downloaded the requested complete 57-second video locally (720p, 25 fps, video-only format 136): https://www.youtube.com/watch?v=Vko4It1y7rY
- Creator/title: Jie Ying, *SgSL3 Video Assignment - My Country | Learning SgSL*.
- Download and derived media remain under ignored `data/offline/youtube/`, not public app examples.
- Added `sgsl_demo.offline_captions`: timed caption rendering, explicit reference/prediction mode, source/model provenance, H.264 MP4, reserved center-bottom strip, frame-timing receipt, optional source audio preservation.
- Rendered a six-second layout preview from source seconds 20-26 (`layout_signing_preview.mp4`). Its placeholder text explicitly says it is not a translation. The original video's burned-in text is source content, not a model output.
- All 14 repository tests pass, including five new renderer tests. Visually inspected the preview.

## Blocking data issue

The public NTU-SADeaf Sign Bank provides isolated sign entries. Its FAQ describes collecting signs in sentences and extracting target vocabulary; this is not a release of a labelled sentence-training corpus.

- https://blogs.ntu.edu.sg/sgslsignbank/faqs/
- https://blogs.ntu.edu.sg/sgslsignbank/word/?frm-word=Able

No NTU phrase model was trained. The ASL alphabet checkpoint cannot translate this continuous SgSL narrative. No sentence predictions or sentence-accuracy results have been produced.

Need the specific approved NTU phrase/sentence dataset archive or link, its labels and reuse conditions, then check coverage of the target video's signs and evaluate on separate recordings. Source subtitles/reference translations must stay distinct from predicted captions. Memorising the target video is not held-out evaluation.

Public source pages reviewed on 18 October 2026. The download metadata supplied no explicit reuse license; do not assume public redistribution permission or add the clip to HF/GitHub examples.
