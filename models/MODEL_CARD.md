# Bundled smoke checkpoint

`demo_alphabet_classifier.npz` is a technical fixture, not a validated SgSL model.

- Architecture: standardized linear softmax classifier over a 24-frame descriptor.
- Input: 53 landmarks × 4 channels per frame; descriptor also includes velocity and acceleration.
- Classes: `A B C D E F H I K L M N O P Q R S T U V W X Y`.
- Source: MediaPipe landmarks extracted from public validation images in the MIT-licensed GloriaNeo repository, then augmented in landmark space.
- Observed bootstrap fit: 100% on the augmented training sequences. This is overfit training agreement, not held-out accuracy.
- Missing: G, J and Z were not represented by usable detected training samples in this bootstrap.
- Intended use: verify that the UI, video decoding, MediaPipe extraction, temporal aggregation, confidence gating and overlay rendering work on CPU.
- Prohibited interpretation: do not cite its confidence or training fit as SgSL recognition accuracy.

Replace this file with a checkpoint trained from approved, labeled clips and evaluate it using signer-held-out data before reporting recognition performance.

