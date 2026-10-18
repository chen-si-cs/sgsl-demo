from pathlib import Path
import shutil

from sgsl_demo.landmarks import MediaPipeLandmarkExtractor


ROOT = Path(__file__).resolve().parents[1]


def test_uploaded_video_outside_workspace_with_unicode_path(tmp_path):
    upload_dir = tmp_path / "uploaded_测试"
    upload_dir.mkdir()
    upload = upload_dir / "webcam.mp4"
    shutil.copyfile(ROOT / "examples" / "random_demo" / "real_webcam_01.mp4", upload)
    with MediaPipeLandmarkExtractor() as extractor:
        result = extractor.extract_video(upload, max_frames=3)
    assert len(result.landmarks) == 3
    assert result.input_fps > 0
