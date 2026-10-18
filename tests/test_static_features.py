import numpy as np

from sgsl_demo.static_features import hand_features


def test_hand_features_preserve_shape_under_translation_and_scale():
    rng = np.random.default_rng(8001)
    frame = np.zeros((53, 4), dtype=np.float32)
    frame[21:42, :3] = rng.normal(size=(21, 3)) * 0.1 + 0.5
    frame[21:42, 3] = 0.95
    moved = frame.copy()
    moved[21:42, :3] = frame[21:42, :3] * 1.7 + np.asarray([0.1, -0.2, 0.3])
    features, indices = hand_features(np.stack([frame, moved]))
    assert features.shape == (2, 288)
    assert indices.tolist() == [0, 1]
    np.testing.assert_allclose(features[0], features[1], atol=2e-5)


def test_missing_hand_does_not_create_a_prediction_feature():
    features, indices = hand_features(np.zeros((5, 53, 4), dtype=np.float32))
    assert features.shape == (0, 288)
    assert indices.size == 0
