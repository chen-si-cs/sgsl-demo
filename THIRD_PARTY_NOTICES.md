# Third-party notices

## Trained alphabet source and examples

The randomly sampled images and constructed montages under `examples/random_demo/` also derive from this CC BY-SA 4.0 dataset. Their exact source paths, random seed, labels and selection rule are recorded in `examples/random_demo/manifest.json`. Sampling does not filter by prediction correctness or confidence.

`models/alphabet_mlp.npz` was trained from [American Sign Language 0–9 / A–Z](https://www.kaggle.com/datasets/prathumarikeri/american-sign-language-09az), by Prathum Arikeri (Kaggle account `prathumarikeri`). The source lists **Creative Commons Attribution-ShareAlike 4.0 (CC BY-SA 4.0)**. The images under `examples/heldout/`, review galleries, and montage derive from that source and retain the source's CC BY-SA 4.0 terms; they are not covered by the repository's MIT software license. These images are selected for a public academic demonstration. The dataset is ASL-labeled; no SgSL equivalence is asserted.

The source download SHA-256 and exact source paths for selected examples are recorded in the server download receipt and `training_results/ws3_alphabet_v1/review_examples/EXAMPLES.json`. The derived model's metadata carries the source and license attribution.

## MediaPipe model bundles

`models/mediapipe/hand_landmarker.task` and `models/mediapipe/pose_landmarker_lite.task` are official Google MediaPipe model bundles downloaded from `storage.googleapis.com/mediapipe-models/`. MediaPipe is distributed under the Apache License 2.0. Review the upstream model card and license before redistribution.

Local SHA-256 values at repository creation:

- `hand_landmarker.task`: `fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1`
- `pose_landmarker_lite.task`: `59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a`

## GloriaNeo SgSL reference repository

The three `examples/random_demo/real_webcam_*.mp4` files are six-second excerpts of the upstream `streamlit/videos/video_1.mp4`, `video_2.mp4` and `video_3.mp4` files. They are unannotated public demonstration footage; their offsets and source paths appear in the example manifest.

The smoke-model source images and included example fixtures derive from the MIT-licensed repository [GloriaNeo/SgSL-Sign-Language-Recognition](https://github.com/GloriaNeo/SgSL-Sign-Language-Recognition), copyright Gloria Neo and contributors. The upstream repository describes an alphabet-level proof of concept and cites ASL image sources used for its alphabet data. These fixtures must not be represented as a word-level or signer-independent SgSL benchmark.

