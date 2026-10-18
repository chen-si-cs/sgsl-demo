# Replacement excerpt: source seconds 4-10

## Follow-up: 10 Hz consensus and slower playback

Current featured files use prefix `sgsl_bochap_04_10_consensus`. The raw source is unchanged (6 seconds, 180 frames, 30 fps). The processed output preserves all 180 frames at 15 fps, hence 12 seconds / 0.5x playback. Audio is pitch-preserving time-stretched; the HUD clock reports source seconds. A short audio stream must not truncate video frames; a new regression test covers that edge case.

Inference uses source frames 0,3,...,177 (60 samples at exactly 10 Hz). Causal consensus uses a 0.3-second trailing window, at least two confident top-1 votes, a two-thirds majority, mean confidence >=0.65, runner-up margin >=0.10 and 0.3-second no-hand reset. Peak evidence is recorded separately, never used to overrule inconsistent frames. Expected reference words are not classifier inputs.

New output: **BECHANV**. H is recovered, but N/P confusion and the later V persist. Dense audit of all 180 source frames plus frozen decoder comparisons are in `sampling_comparison.json`; the feature/score cache remains under ignored `data/offline/bochap_sampling_audit/`. In source-relative seconds 3.8-4.6, maximum P probability is 1.42%, maximum N probability is 88.95%. Sampling alone cannot select P with these scores. Motion blur remains an unproven cause, not an established diagnosis. User-provided expected reference is BECHAP; the source title says Bochap. Neither is fed into inference.

All 29 tests pass, including causal-prefix invariance, confidence-spike rejection, short-letter recovery, fixed-rate sampling, and source/playback timing/audio preservation. Updated Gradio configuration builds successfully with the new example and default 0.5x playback control. No checkpoint training or HF upload was performed.

## Previous export (retained for comparison)

Source: https://www.youtube.com/watch?v=LxdjT1g3REc

Creator: Lily Goh *Deaf Singapore* (ExtraOrdinary Horizons).

Title: #SgSL Sign for Bochap.

Extracted exactly six seconds, starting at 4.0 s, re-encoded H.264/AAC with original audio. Raw and processed versions both preserve 180 frames at 30 fps. The processed version adds the same top-left landmark inset, top-right current letter/uncertainty and bottom accumulating characters used by HF inference.

The unchanged alphabet checkpoint/threshold (0.65, dwell 3 samples) emits **BECANV**, not a correct transcription of the source word. Coverage is 59/72 sampled frames = 81.94%. There is no target-word correction and no source label enters inference. Full timeline, frame scores and model/source hashes are in `examples/featured/sgsl_bochap_04_10_processed.predictions.json`.

The featured manifest now selects only this replacement; old source/exports remain intact. The replacement HF ZIP includes only files referenced by the current featured manifest, not the old My Country media. No HF upload or model retraining was performed.
