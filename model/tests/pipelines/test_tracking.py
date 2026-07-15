import numpy as np
import pytest
import supervision as sv

from orc_model.pipelines.tracking import _filter_workspace, load_tracker


def test_workspace_filter_removes_detections_below_the_configured_table_boundary():
    detections = sv.Detections(
        xyxy=np.array(
            [
                [10.0, 20.0, 30.0, 40.0],
                [10.0, 91.0, 30.0, 99.0],
            ],
            dtype=np.float32,
        ),
        mask=np.zeros((2, 100, 100), dtype=bool),
        confidence=np.array([0.8, 0.9], dtype=np.float32),
        class_id=np.array([0, 0]),
        data={"class_name": np.array(["surgical_instrument", "surgical_instrument"])},
    )

    filtered = _filter_workspace(detections, frame_height=100, max_center_y_ratio=0.88)

    assert len(filtered) == 1
    assert np.array_equal(filtered.xyxy[0], detections.xyxy[0])
    assert filtered.confidence[0] == detections.confidence[0]


def test_workspace_filter_can_be_disabled():
    detections = sv.Detections(
        xyxy=np.array([[10.0, 91.0, 30.0, 99.0]], dtype=np.float32),
        confidence=np.array([0.9], dtype=np.float32),
        class_id=np.array([0]),
    )

    assert _filter_workspace(detections, frame_height=100, max_center_y_ratio=None) is detections


@pytest.mark.parametrize("ratio", [0.0, -0.1, 1.1])
def test_workspace_filter_rejects_invalid_boundaries_even_when_detections_are_empty(ratio):
    with pytest.raises(ValueError, match="workspace_max_center_y_ratio"):
        _filter_workspace(sv.Detections.empty(), frame_height=100, max_center_y_ratio=ratio)


def test_load_tracker_rejects_invalid_workspace_boundary_before_loading_models(tmp_path):
    with pytest.raises(ValueError, match="workspace_max_center_y_ratio"):
        load_tracker(tmp_path / "missing.onnx", workspace_max_center_y_ratio=1.1)
