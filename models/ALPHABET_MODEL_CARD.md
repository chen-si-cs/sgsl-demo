# Trained alphabet MLP

- File: `alphabet_mlp.npz` (413,281 bytes), trained on ws3 GPU 0.
- Source: `prathumarikeri/american-sign-language-09az`, Kaggle, CC BY-SA 4.0 as listed by the source.
- Task: 24 static ASL-based fingerspelling letters; excludes J and Z.
- Features: 288 wrist-relative coordinates, pairwise distances and angle cosines.
- Network: 288 → 256 → 128 → 24, ReLU; 109,976 parameters.
- Split: fixed before training, 70/15/15 approximate proportions, exact decoded duplicates removed, identical 64-bit perceptual hashes grouped.
- Actual split sizes: train 12,334; validation 2,656; test 2,649.
- Validation accuracy: 0.9928463855. Test accuracy: 0.9894299736. Test macro-F1: 0.9837289492.
- Coverage: MediaPipe found usable hands in 27,244 of the 36,000 selected source images (75.7%); the classifier metrics apply to detected hands.
- Best epoch: 94; stopped at epoch 139 by the validation stopping rule.
- CPU export maximum logit difference from GPU inference: 0.0000038147.
- Frame sequences: confidence averaging and dwell-time decoding; no learned motion model.
- Limitations: source-correlated images, unknown signer identities, no SgSL word evaluation, no verified unknown-gesture rejection, camera-domain variation. Temperature calibrates known-class scores; it does not guarantee open-set safety.

The test gallery contains selected high-confidence correct examples plus selected errors. The montage is constructed from held-out still images, with blank gaps, and is labeled accordingly.
