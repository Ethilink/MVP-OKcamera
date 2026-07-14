"""Unit tests for `backend.thumbnails` — the consumer-side crop helper.

Pure OpenCV + base64, no FastAPI/session/capture. Every failure path must return
`None` (a null tile) rather than raise, so one bad box can't 500 `/status`.
"""

from __future__ import annotations

import base64

import cv2
import numpy as np

from backend.thumbnails import Detection, build_detections, crop_thumbnail


def _frame(h: int = 120, w: int = 160, value: int = 90) -> np.ndarray:
    return np.full((h, w, 3), value, dtype=np.uint8)


def _decode_data_uri(data_uri: str) -> np.ndarray:
    assert data_uri.startswith("data:image/jpeg;base64,")
    raw = base64.b64decode(data_uri.split(",", 1)[1], validate=True)
    return cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)


class TestCropThumbnail:
    def test_valid_box_returns_square_jpeg_data_uri(self) -> None:
        data_uri = crop_thumbnail(_frame(), (20.0, 30.0, 60.0, 90.0), size=96)

        image = _decode_data_uri(data_uri)
        assert image.shape == (96, 96, 3)

    def test_aspect_ratio_is_preserved_by_letterboxing(self) -> None:
        # A wide, short box → the content band is centered with black bars above
        # and below, not stretched to fill the square.
        color_frame = _frame(value=0)
        color_frame[40:60, 10:150] = (0, 0, 255)  # a wide red bar

        image = _decode_data_uri(crop_thumbnail(color_frame, (10.0, 40.0, 150.0, 60.0), size=96))

        # Top and bottom rows are letterbox padding (near black); the middle band
        # carries the (red) content.
        assert image[0].max() < 40
        assert image[-1].max() < 40
        assert int(image[48, :, 2].max()) > 120  # red channel present mid-canvas

    def test_box_partly_outside_the_frame_is_clamped(self) -> None:
        # Right/bottom overflow must clamp to the frame, still producing a tile.
        data_uri = crop_thumbnail(_frame(), (140.0, 100.0, 400.0, 400.0), size=96)

        assert _decode_data_uri(data_uri).shape == (96, 96, 3)

    def test_zero_area_box_returns_none(self) -> None:
        assert crop_thumbnail(_frame(), (50.0, 50.0, 50.0, 50.0)) is None

    def test_sub_pixel_sliver_box_returns_none(self) -> None:
        # A collapsed box at fractional coords must not slip through floor/ceil
        # as a 1-px sliver stretched into the tile.
        assert crop_thumbnail(_frame(), (5.5, 30.0, 5.5, 90.0)) is None
        assert crop_thumbnail(_frame(), (20.0, 44.5, 60.0, 45.4)) is None

    def test_box_fully_outside_the_frame_returns_none(self) -> None:
        assert crop_thumbnail(_frame(w=160, h=120), (500.0, 500.0, 520.0, 520.0)) is None

    def test_off_frame_box_does_not_leak_padded_edge_pixels(self) -> None:
        # Entirely left of the frame (x2 < 0). Padding must NOT pull column 0 in
        # and encode a strip of unrelated edge pixels — it must return None.
        frame = _frame(w=100, h=100)
        assert crop_thumbnail(frame, (-10.0, 10.0, -1.0, 20.0)) is None
        assert crop_thumbnail(frame, (10.0, -30.0, 20.0, -1.0)) is None  # above frame

    def test_non_finite_box_returns_none(self) -> None:
        assert crop_thumbnail(_frame(), (float("nan"), 0.0, 10.0, 10.0)) is None
        assert crop_thumbnail(_frame(), (0.0, 0.0, float("inf"), 10.0)) is None

    def test_none_frame_returns_none(self) -> None:
        assert crop_thumbnail(None, (0.0, 0.0, 10.0, 10.0)) is None

    def test_reversed_corners_are_still_cropped(self) -> None:
        # x2<x1 / y2<y1 must be normalized, not rejected.
        data_uri = crop_thumbnail(_frame(), (60.0, 90.0, 20.0, 30.0), size=96)

        assert _decode_data_uri(data_uri).shape == (96, 96, 3)

    def test_reads_from_a_read_only_frame(self) -> None:
        # capture publishes read-only frames; cropping must not need to write.
        frame = _frame()
        frame.setflags(write=False)

        assert crop_thumbnail(frame, (20.0, 30.0, 60.0, 90.0)) is not None


class TestBuildDetections:
    def test_sorted_by_tracker_id_with_labels(self) -> None:
        detections = build_detections(
            _frame(),
            ((3, (10.0, 10.0, 40.0, 40.0)), (1, (50.0, 50.0, 90.0, 90.0))),
        )

        assert [d.tracker_id for d in detections] == [1, 3]
        assert [d.label for d in detections] == ["Instrument 1", "Instrument 3"]
        assert all(isinstance(d, Detection) for d in detections)
        assert all(d.thumbnail.startswith("data:image/jpeg;base64,") for d in detections)

    def test_one_bad_box_yields_a_null_tile_without_dropping_the_others(self) -> None:
        detections = build_detections(
            _frame(),
            (
                (1, (10.0, 10.0, 40.0, 40.0)),          # good
                (2, (500.0, 500.0, 520.0, 520.0)),      # off-frame → null
                (3, (60.0, 60.0, 100.0, 100.0)),        # good
            ),
        )

        by_id = {d.tracker_id: d for d in detections}
        assert set(by_id) == {1, 2, 3}
        assert by_id[1].thumbnail is not None
        assert by_id[2].thumbnail is None
        assert by_id[3].thumbnail is not None

    def test_empty_boxes_yields_empty_list(self) -> None:
        assert build_detections(_frame(), ()) == []

    def test_none_frame_yields_null_tiles_not_an_error(self) -> None:
        detections = build_detections(None, ((1, (0.0, 0.0, 10.0, 10.0)),))

        assert detections[0].tracker_id == 1
        assert detections[0].thumbnail is None
