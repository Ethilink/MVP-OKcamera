"""Unit tests for `backend.thumbnails` — the consumer-side crop helper.

Pure OpenCV + base64, no FastAPI/session/capture. Every failure path must return
`None` (a null tile) rather than raise, so one bad box can't 500 `/status`.
"""

from __future__ import annotations

import base64

import cv2
import numpy as np

from backend.capture import DetectionBox
from backend.render import UNKNOWN_COLOUR, catalog_colour
from backend.thumbnails import Detection, build_detections, crop_thumbnail


def _frame(h: int = 120, w: int = 160, value: int = 90) -> np.ndarray:
    return np.full((h, w, 3), value, dtype=np.uint8)


def _box(tracker_id: int, xyxy: tuple[float, float, float, float], resolving: bool = False) -> DetectionBox:
    """A `DetectionBox` — the frozen dataclass `build_detections` consumes
    (T11/B1). `resolving` feeds the tile's identity state (T11/R3); defaults False."""
    return DetectionBox(tracker_id=tracker_id, xyxy=xyxy, resolving=resolving)


def _decode_data_uri(data_uri: str, *, unchanged: bool = False) -> np.ndarray:
    raw = base64.b64decode(data_uri.split(",", 1)[1], validate=True)
    mode = cv2.IMREAD_UNCHANGED if unchanged else cv2.IMREAD_COLOR
    return cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), mode)


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

    def test_mask_returns_a_transparent_png_cutout(self) -> None:
        frame = _frame(value=0)
        frame[30:90, 20:60] = (10, 90, 220)
        mask = np.zeros((60, 40), dtype=bool)
        mask[8:52, 12:28] = True

        data_uri = crop_thumbnail(
            frame,
            (20.0, 30.0, 60.0, 90.0),
            mask=mask,
            size=96,
        )

        assert data_uri.startswith("data:image/png;base64,")
        image = _decode_data_uri(data_uri, unchanged=True)
        assert image.shape == (96, 96, 4)
        assert image[0, 0, 3] == 0
        assert image[:, :, 3].max() == 255
        assert image[:, :, 3].min() == 0

    def test_empty_mask_falls_back_to_the_rectangular_jpeg_crop(self) -> None:
        data_uri = crop_thumbnail(
            _frame(),
            (20.0, 30.0, 60.0, 90.0),
            mask=np.zeros((60, 40), dtype=bool),
            size=96,
        )

        assert data_uri.startswith("data:image/jpeg;base64,")

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
    # A representative demo catalog + a roster recognising a subset of it.
    _CATALOG = frozenset(range(1, 9))
    _ROSTER = frozenset({1, 3})

    def test_sorted_by_tracker_id_with_labels(self) -> None:
        detections = build_detections(
            _frame(),
            (_box(3, (10.0, 10.0, 40.0, 40.0)), _box(1, (50.0, 50.0, 90.0, 90.0))),
            self._ROSTER,
            self._CATALOG,
        )

        assert [d.tracker_id for d in detections] == [1, 3]
        assert [d.label for d in detections] == ["Instrument 1", "Instrument 3"]
        assert all(isinstance(d, Detection) for d in detections)
        assert all(d.thumbnail.startswith("data:image/jpeg;base64,") for d in detections)

    def test_masked_detection_uses_a_transparent_png_thumbnail(self) -> None:
        box = DetectionBox(
            tracker_id=1,
            xyxy=(20.0, 30.0, 60.0, 90.0),
            resolving=False,
            mask=np.ones((60, 40), dtype=bool),
        )

        [detection] = build_detections(
            _frame(), (box,), self._ROSTER, self._CATALOG
        )

        assert detection.thumbnail.startswith("data:image/png;base64,")

    def test_one_bad_box_yields_a_null_tile_without_dropping_the_others(self) -> None:
        detections = build_detections(
            _frame(),
            (
                _box(1, (10.0, 10.0, 40.0, 40.0)),          # good
                _box(2, (500.0, 500.0, 520.0, 520.0)),      # off-frame → null
                _box(3, (60.0, 60.0, 100.0, 100.0)),        # good
            ),
            self._ROSTER,
            self._CATALOG,
        )

        by_id = {d.tracker_id: d for d in detections}
        assert set(by_id) == {1, 2, 3}
        assert by_id[1].thumbnail is not None
        assert by_id[2].thumbnail is None
        assert by_id[3].thumbnail is not None
        # A failed crop must NOT lose the tile's identity: state/label/colour come
        # from the classifier, independent of whether the crop encoded. id 2 is not
        # in the roster {1,3}, so it stays a settled Unknown even with a null tile.
        assert by_id[2].state == "unknown"
        assert by_id[2].label == "Unknown"
        assert by_id[2].colour == UNKNOWN_COLOUR
        assert by_id[1].state == "recognised" and by_id[1].label == "Instrument 1"

    def test_empty_boxes_yields_empty_list(self) -> None:
        assert build_detections(_frame(), (), self._ROSTER, self._CATALOG) == []

    def test_none_frame_yields_null_tiles_not_an_error(self) -> None:
        detections = build_detections(
            None, (_box(1, (0.0, 0.0, 10.0, 10.0)),), self._ROSTER, self._CATALOG
        )

        assert detections[0].tracker_id == 1
        assert detections[0].thumbnail is None


class TestBuildDetectionsIdentityState:
    """T11/R3 + backend test 8: each tile's state/label/colour follows the single
    identity policy against the same-tick roster/catalog and the box's resolving
    flag, and a non-recognised tile never exposes its raw id as a label (D4)."""

    _CATALOG = frozenset(range(1, 9))

    def test_recognised_tile_gets_instrument_label_and_catalog_colour(self) -> None:
        [tile] = build_detections(
            _frame(), (_box(3, (10.0, 10.0, 40.0, 40.0)),), frozenset({3}), self._CATALOG
        )

        assert tile.state == "recognised"
        assert tile.label == "Instrument 3"
        assert tile.colour == catalog_colour(self._CATALOG, 3)
        assert tile.colour != UNKNOWN_COLOUR

    def test_resolving_non_roster_tile_is_recognising_gray_and_nameless(self) -> None:
        [tile] = build_detections(
            _frame(),
            (_box(1042, (10.0, 10.0, 40.0, 40.0), resolving=True),),
            frozenset(),
            self._CATALOG,
        )

        assert tile.state == "recognising"
        assert tile.label == ""            # no name while resolving
        assert "1042" not in tile.label    # never the raw id
        assert tile.colour == UNKNOWN_COLOUR

    def test_settled_non_roster_tile_is_unknown_gray(self) -> None:
        [tile] = build_detections(
            _frame(),
            (_box(1042, (10.0, 10.0, 40.0, 40.0), resolving=False),),
            frozenset(),
            self._CATALOG,
        )

        assert tile.state == "unknown"
        assert tile.label == "Unknown"
        assert "1042" not in tile.label
        assert tile.colour == UNKNOWN_COLOUR
