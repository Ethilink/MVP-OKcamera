import pytest

from orc_model.components.detector import Detector


def test_missing_weights_file_raises_clear_error():
    with pytest.raises(FileNotFoundError):
        Detector("/nonexistent/path.onnx")
