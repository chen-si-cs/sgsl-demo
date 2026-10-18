# New alphabet lesson comparison

Source: https://www.youtube.com/watch?v=6EW4PVT_d_4

Deaf.SG, *Singapore Sign Language (SgSL) Lesson: Alphabet - A to Z*, published 20 August 2018. Source branding and printed letters remain visible. They are not read by or passed into the classifier. License metadata does not establish open redistribution permission.

The unchanged alphabet checkpoint, 10 Hz source grid, 0.65 confidence threshold and causal 0.3-second consistency decoder were used. No model training, dictionary constraint or label-based correction was applied.

## Full video

- Source: 51.0667 s, 1532 frames at 30 fps.
- Export: 102.1333 s, 1532 frames at 15 fps (0.5x, with time-stretched audio).
- 511 sampled frames; 90.41% hand detection coverage, not recognition accuracy.
- Decoded output: ABCDOCEFGHHMIAKLREKAOQRESXRVWXA.
- J and Z are not in the static model vocabulary. Other substitutions, missed letters and extra tokens remain.
- Processing/export took approximately 70.6 seconds locally; not a live-speed guarantee.

## Curated opening for the slide

The first 6.5 source seconds were selected after inspecting the full run because this opening works well. A fresh independent pipeline invocation on the re-encoded cut emits **ABCD**; source times are 0.1, 2.0, 3.7 and 5.6 s. There is no manually supplied transcript in inference. This is an explicitly curated demonstration, not unbiased validation.

- Source cut: 6.5 s, 195 frames at 30 fps.
- Export: 13 s, 195 frames at 15 fps.
- 65 sampled frames; 100% hand detection coverage.
- Processing/export took approximately 6.2 seconds locally.

Both full lesson and successful opening are added alongside the prior Bochap example. Reports with complete scores and source/model hashes are beside the corresponding files in `examples/featured/`. No HF upload was performed.
