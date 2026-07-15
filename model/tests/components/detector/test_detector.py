import pytest
import numpy as np

from orc_model.components.detector import Detector
from orc_model.components.detector import detector as detector_module


def test_missing_weights_file_raises_clear_error():
    with pytest.raises(FileNotFoundError):
        Detector("/nonexistent/path.onnx")


def test_onnx_io_contract_is_resolved_once_at_session_load(monkeypatch, tmp_path):
    class ValueInfo:
        def __init__(self, name):
            self.name = name

    class FakeSession:
        def __init__(self):
            self.input_reads = 0
            self.output_reads = 0
            self.requested_outputs = []

        def get_inputs(self):
            self.input_reads += 1
            return [ValueInfo("images")]

        def get_outputs(self):
            self.output_reads += 1
            return [ValueInfo("labels"), ValueInfo("masks"), ValueInfo("dets")]

        def run(self, output_names, _feeds):
            self.requested_outputs.append(output_names)
            return [np.zeros((1, 1, 4)), np.zeros((1, 1, 2)), np.zeros((1, 1, 1, 1))]

    session = FakeSession()
    monkeypatch.setattr(detector_module.onnxruntime, "InferenceSession", lambda *_a, **_kw: session)
    monkeypatch.setattr(detector_module, "preprocess", lambda _image: np.zeros((1, 3, 768, 768)))
    sentinel = object()
    monkeypatch.setattr(detector_module, "decode_predictions", lambda *_a, **_kw: sentinel)
    weights = tmp_path / "model.onnx"
    weights.write_bytes(b"fake")
    detector = Detector(weights)

    image = np.zeros((10, 20, 3), dtype=np.uint8)
    assert detector.predict(image) is sentinel
    assert detector.predict(image) is sentinel

    assert session.input_reads == 1
    assert session.output_reads == 1
    assert session.requested_outputs == [["dets", "labels", "masks"]] * 2
