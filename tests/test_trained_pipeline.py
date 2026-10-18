from pathlib import Path

from sgsl_demo.pipeline import RecognitionPipeline


ROOT = Path(__file__).resolve().parents[1]


def test_trained_model_decodes_heldout_letter_montage():
    pipeline = RecognitionPipeline(ROOT / "models" / "alphabet_mlp.npz")
    result = pipeline.predict_video(ROOT / "examples" / "heldout" / "heldout_CAB_montage.mp4")
    assert result.sequence_text == "CAB"
    assert [row["label"] for row in result.timeline] == ["C", "A", "B"]
