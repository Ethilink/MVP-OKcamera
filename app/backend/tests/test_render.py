"""T10 B-V (overlay encoding) + T03 AC10, against `backend.render`.

MIGRATED for T10: the overlay entry point is now `OverlayRenderer`
(`__call__(frame, dets, roster, t)`), which replaces the plain `render`
function as `CaptureLoop`'s default `render_fn`. T03's AC10 pair (draw in
place; colour by track) is kept, re-aimed at the renderer's NOT-RECORDING
state — the state AC10 described, and the one B-V1 freezes as "exactly today's
overlay". `render` itself is gone from the spec's public interface, so nothing
here imports it.

REVISED 2026-07-16 (the wait-state fix): the renderer no longer owns a
`pending_s` timer or a first-seen map. Whether a not-in-roster track spins
(resolving) or shows the settled "Unknown" label is the linker's per-detection
`data["resolving"]` flag, so a deferred-but-genuine track can no longer flicker
to "Unknown" before it links. Tests drive that flag directly via
`boxed_detections(..., resolving=[...])` instead of advancing a clock.

No threads, no camera, no real clock: `OverlayRenderer` is driven with
hand-picked `t` floats (still needed — the spinner animates with `t`), exactly
like `Session`. Frames are small so mask/box math stays cheap, but not tiny —
the probes sample box interiors, and labels need room.

**Tunables are never hard-asserted** (per the spec's "Tunable parameters"): the
palette's 8 hex values are asserted DISTINCT and STABLE, never enumerated; the
gray is asserted achromatic (see `tests/overlay_probe.py`), never spelled out.
"""

from __future__ import annotations

import re

import numpy as np

from backend.render import ROSTER_PALETTE, OverlayRenderer, roster_colour
from tests.overlay_probe import (
    COLOUR_SIMILARITY_MIN,
    GRAY_CHROMA_MAX,
    blank_frame,
    box_mean_bgr,
    boxed_detections,
    chroma_ratio,
    colour_similarity,
    drew_any_instrument_label,
    label_spy,
)

# Boxes are large and clear of the top edge: the probes sample an inset
# interior (skipping the outline), and a label drawn above a box must not spill
# into the sample. Two side-by-side boxes let one frame carry two ids.
_LEFT_BOX = (30, 60, 120, 150)
_RIGHT_BOX = (150, 60, 240, 150)
_FRAME_SHAPE = (200, 260)  # (height, width)

_T0 = 100.0  # an arbitrary non-zero monotonic origin


def _recording_renderer() -> OverlayRenderer:
    renderer = OverlayRenderer()
    renderer.set_recording(True)
    return renderer


def _draw(renderer, dets, roster, t: float = _T0) -> np.ndarray:
    return renderer(blank_frame(*_FRAME_SHAPE), dets, roster, t)


def _one(box, tracker_id: int, resolving: bool | None = None):
    return boxed_detections(
        [box],
        [tracker_id],
        _FRAME_SHAPE,
        resolving=None if resolving is None else [resolving],
    )


class TestBV1NotRecordingIsTodaysOverlay:
    """B-V1: in the `set_recording(False)` state the renderer draws exactly
    today's overlay — per-track colours + "Instrument {id}" labels — for EVERY
    detection, roster member or not, and ignores the resolving flag entirely.
    Setup/finished must look unchanged; the roster is stale there by design.
    (Also carries T03's AC10.)"""

    def test_b_v1_ac10_render_mutates_in_place_and_preserves_shape_and_dtype(
        self,
    ) -> None:
        frame = blank_frame(*_FRAME_SHAPE)
        before = frame.copy()
        renderer = OverlayRenderer()
        renderer.set_recording(False)

        result = renderer(frame, _one(_LEFT_BOX, 7), frozenset({7}), _T0)

        assert result is frame
        assert result.shape == before.shape
        assert result.dtype == before.dtype
        assert not np.array_equal(frame, before)

    def test_b_v1_ac10_different_tracker_ids_get_different_colours(self) -> None:
        renderer = OverlayRenderer()
        renderer.set_recording(False)
        dets = boxed_detections([_LEFT_BOX, _RIGHT_BOX], [1, 2], _FRAME_SHAPE)

        result = _draw(renderer, dets, frozenset({1, 2}), _T0)

        colour_a = box_mean_bgr(result, _LEFT_BOX)
        colour_b = box_mean_bgr(result, _RIGHT_BOX)
        assert not np.allclose(colour_a, colour_b, atol=5)

    def test_b_v1_not_recording_labels_every_id_instrument_n_even_off_roster(
        self,
    ) -> None:
        renderer = OverlayRenderer()
        renderer.set_recording(False)
        dets = boxed_detections([_LEFT_BOX, _RIGHT_BOX], [1, 9], _FRAME_SHAPE)

        with label_spy() as texts:
            _draw(renderer, dets, frozenset({1}), _T0)  # id 9 is NOT in the roster

        assert "Instrument 1" in texts
        assert "Instrument 9" in texts  # setup does not know about "Unknown"
        assert "Unknown" not in texts

    def test_b_v1_not_recording_off_roster_id_is_not_grayed_out(self) -> None:
        renderer = OverlayRenderer()
        renderer.set_recording(False)

        result = _draw(renderer, _one(_LEFT_BOX, 9), frozenset({1, 2, 3}), _T0)

        drawn = box_mean_bgr(result, _LEFT_BOX)
        assert drawn.max() > 0  # sanity: something was drawn
        assert chroma_ratio(drawn) > GRAY_CHROMA_MAX

    def test_b_v1_not_recording_ignores_a_resolving_flag(self) -> None:
        # A resolving flag on the detections is a recording-time concept; the
        # setup overlay pays it no attention and still labels the track.
        renderer = OverlayRenderer()
        renderer.set_recording(False)

        with label_spy() as texts:
            result = _draw(renderer, _one(_LEFT_BOX, 9, resolving=True), frozenset({1}), _T0)

        assert "Instrument 9" in texts
        assert "Unknown" not in texts
        assert chroma_ratio(box_mean_bgr(result, _LEFT_BOX)) > GRAY_CHROMA_MAX

    def test_b_v1_a_fresh_renderer_is_not_recording(self) -> None:
        # CaptureLoop builds its default renderer at construction, long before
        # any Start, and the session boots into SETUP — so a renderer that has
        # never been told anything must draw the setup overlay.
        renderer = OverlayRenderer()

        with label_spy() as texts:
            _draw(renderer, _one(_LEFT_BOX, 9), frozenset({1}), _T0)

        assert "Instrument 9" in texts
        assert "Unknown" not in texts

    def test_b_v1_set_recording_false_returns_to_todays_overlay(self) -> None:
        # Stop must undo Start: a settled Unknown goes back to being labelled.
        renderer = _recording_renderer()
        _draw(renderer, _one(_LEFT_BOX, 9, resolving=False), frozenset({1}), _T0)  # gray Unknown

        renderer.set_recording(False)
        with label_spy() as texts:
            _draw(renderer, _one(_LEFT_BOX, 9), frozenset({1}), _T0 + 2.0)

        assert "Instrument 9" in texts
        assert "Unknown" not in texts


class TestBV2RecordingRosterIdKeepsItsColour:
    """B-V2: while recording, a detection whose tracker_id is in the roster is
    drawn in `roster_colour(roster, id)` — the same hex `/status` reports — and
    labelled "Instrument {id}". The colour follows the id through the roster,
    so a returned instrument regains it with no renderer memory involved."""

    def test_b_v2_recording_roster_id_is_labelled_instrument_n(self) -> None:
        renderer = _recording_renderer()

        with label_spy() as texts:
            _draw(renderer, _one(_LEFT_BOX, 5), frozenset({5, 6, 7}), _T0)

        assert "Instrument 5" in texts
        assert "Unknown" not in texts

    def test_b_v2_recording_mask_colour_matches_the_hex_roster_colour_reports(
        self,
    ) -> None:
        roster = frozenset({5, 6, 7})
        renderer = _recording_renderer()

        result = _draw(renderer, _one(_LEFT_BOX, 5), roster, _T0)

        drawn = box_mean_bgr(result, _LEFT_BOX)
        assert drawn.max() > 0  # sanity: something was drawn
        assert colour_similarity(drawn, roster_colour(roster, 5)) >= COLOUR_SIMILARITY_MIN

    def test_b_v2_recording_mask_colour_follows_the_roster_index_not_the_tracker_id(
        self,
    ) -> None:
        # Same tracker_id, two rosters that place it at different sorted
        # positions -> two different palette slots. Today's overlay (which keys
        # off the tracker_id alone) cannot tell these apart.
        first_in_roster = _draw(
            _recording_renderer(), _one(_LEFT_BOX, 5), frozenset({5, 6, 7}), _T0
        )
        second_in_roster = _draw(
            _recording_renderer(), _one(_LEFT_BOX, 5), frozenset({1, 5}), _T0
        )

        as_index_0 = box_mean_bgr(first_in_roster, _LEFT_BOX)
        as_index_1 = box_mean_bgr(second_in_roster, _LEFT_BOX)
        assert as_index_0.max() > 0  # sanity: something was drawn
        assert not np.allclose(as_index_0, as_index_1, atol=5)

    def test_b_v2_recording_the_same_roster_slot_draws_the_same_colour(self) -> None:
        # The mirror image of the test above: different tracker_ids sitting at
        # the same sorted position in their rosters share a colour.
        five_of_567 = _draw(
            _recording_renderer(), _one(_LEFT_BOX, 5), frozenset({5, 6, 7}), _T0
        )
        two_of_2_9_11 = _draw(
            _recording_renderer(), _one(_LEFT_BOX, 2), frozenset({2, 9, 11}), _T0
        )

        assert np.allclose(
            box_mean_bgr(five_of_567, _LEFT_BOX),
            box_mean_bgr(two_of_2_9_11, _LEFT_BOX),
            atol=5,
        )

    def test_b_v2_two_roster_ids_in_one_frame_get_different_colours(self) -> None:
        roster = frozenset({1, 2, 3})
        renderer = _recording_renderer()
        dets = boxed_detections([_LEFT_BOX, _RIGHT_BOX], [1, 2], _FRAME_SHAPE)

        result = _draw(renderer, dets, roster, _T0)

        assert not np.allclose(
            box_mean_bgr(result, _LEFT_BOX), box_mean_bgr(result, _RIGHT_BOX), atol=5
        )

    def test_b_v2_a_returned_instrument_regains_the_colour_the_api_reports(self) -> None:
        roster = frozenset({1, 2, 3})
        renderer = _recording_renderer()
        # id 2 (the one that leaves) sits in the LEFT box; id 1 stays put in the
        # RIGHT one throughout, so the sampled box only ever holds id 2.
        both = boxed_detections([_LEFT_BOX, _RIGHT_BOX], [2, 1], _FRAME_SHAPE)
        without_id_2 = _one(_RIGHT_BOX, 1)

        before = _draw(renderer, both, roster, _T0)
        for step in range(1, 30):  # id 2 is off the table for 30s
            _draw(renderer, without_id_2, roster, _T0 + step)
        after = _draw(renderer, both, roster, _T0 + 30.0)

        colour_before = box_mean_bgr(before, _LEFT_BOX)
        colour_after = box_mean_bgr(after, _LEFT_BOX)
        assert colour_similarity(colour_before, roster_colour(roster, 2)) >= COLOUR_SIMILARITY_MIN
        assert np.allclose(colour_before, colour_after, atol=5)


class TestBV2RosterColourIsAPureStablePalette:
    """B-V2/B-A1: `roster_colour(roster, id)` = palette[index of id in
    sorted(roster) mod 8] — pure, and stable for a whole recording because the
    roster is frozen. The 8 hex VALUES are a tunable: asserted distinct and
    well-formed, never enumerated."""

    def test_b_v2_roster_palette_holds_eight_distinct_colours(self) -> None:
        assert len(ROSTER_PALETTE) == 8
        assert len(set(ROSTER_PALETTE)) == 8

    def test_b_v2_roster_palette_entries_are_hex_colour_strings(self) -> None:
        for colour in ROSTER_PALETTE:
            assert re.fullmatch(r"#[0-9a-fA-F]{6}", colour), colour

    def test_b_v2_roster_colour_indexes_the_palette_by_sorted_position(self) -> None:
        roster = frozenset({4, 11, 2, 7})

        for index, tracker_id in enumerate(sorted(roster)):
            assert roster_colour(roster, tracker_id) == ROSTER_PALETTE[index]

    def test_b_v2_roster_colour_gives_a_full_eight_roster_eight_distinct_colours(
        self,
    ) -> None:
        roster = frozenset({3, 5, 7, 9, 10, 11, 12, 14})  # OC-SORT's raw counter

        colours = [roster_colour(roster, tracker_id) for tracker_id in sorted(roster)]

        assert len(set(colours)) == 8

    def test_b_v2_roster_colour_wraps_the_palette_past_the_eighth_id(self) -> None:
        roster = frozenset(range(1, 11))  # 10 ids, 8 colours
        ordered = sorted(roster)

        colours = [roster_colour(roster, tracker_id) for tracker_id in ordered]

        assert len(set(colours[:8])) == 8  # the first eight are all distinct
        assert colours[8] == colours[0]  # ... then it wraps
        assert colours[9] == colours[1]

    def test_b_v2_roster_colour_is_stable_across_repeated_calls(self) -> None:
        roster = frozenset({3, 5, 7})

        assert roster_colour(roster, 5) == roster_colour(roster, 5)

    def test_b_v2_roster_colour_is_pure_and_ignores_call_order(self) -> None:
        roster = frozenset({3, 5, 7})

        first_pass = [roster_colour(roster, i) for i in (7, 3, 5)]
        second_pass = [roster_colour(roster, i) for i in (3, 5, 7)]

        assert first_pass == [second_pass[2], second_pass[0], second_pass[1]]


class TestBV3ResolvingUnknownIsGrayAndSpins:
    """B-V3 (revised): while recording, a not-in-roster detection whose
    `data["resolving"]` is True draws gray, carries NO "Instrument" and NO
    "Unknown" label, and shows a resolving indicator that visibly animates
    across frames. The flag is the linker's actual wait-state, not a timer."""

    def test_b_v3_resolving_not_in_roster_track_is_gray(self) -> None:
        renderer = _recording_renderer()

        result = _draw(renderer, _one(_LEFT_BOX, 9, resolving=True), frozenset({1, 2, 3}))

        drawn = box_mean_bgr(result, _LEFT_BOX)
        assert drawn.max() > 0  # sanity: something was drawn
        assert chroma_ratio(drawn) <= GRAY_CHROMA_MAX

    def test_b_v3_resolving_not_in_roster_track_carries_no_settled_label(self) -> None:
        renderer = _recording_renderer()

        with label_spy() as texts:
            _draw(renderer, _one(_LEFT_BOX, 9, resolving=True), frozenset({1, 2, 3}))

        assert not drew_any_instrument_label(texts)
        assert "Unknown" not in texts  # still deciding — no settled label yet

    def test_b_v3_the_resolving_indicator_animates_across_frames(self) -> None:
        # Identical detections, identical black frame, only `t` moves: any
        # pixel difference is the resolving indicator doing its job. A static
        # gray box would make every frame byte-identical.
        renderer = _recording_renderer()
        roster = frozenset({1, 2, 3})
        dets = _one(_LEFT_BOX, 9, resolving=True)

        frames = [
            _draw(renderer, dets, roster, _T0 + step * 0.1).tobytes()
            for step in range(6)  # 0.0 .. 0.5s
        ]

        assert len(set(frames)) >= 2

    def test_b_v3_resolving_is_decided_per_row_not_per_renderer(self) -> None:
        # Two not-in-roster ids in ONE frame with opposite flags: the resolving
        # one spins gray with no label, the settled one is a gray "Unknown".
        renderer = _recording_renderer()
        roster = frozenset({1, 2, 3})
        dets = boxed_detections(
            [_LEFT_BOX, _RIGHT_BOX], [9, 10], _FRAME_SHAPE, resolving=[False, True]
        )

        with label_spy() as texts:
            result = _draw(renderer, dets, roster)

        assert "Unknown" in texts  # id 9 settled
        assert not drew_any_instrument_label(texts)
        resolving_box = box_mean_bgr(result, _RIGHT_BOX)  # id 10, resolving
        assert resolving_box.max() > 0  # sanity: something was drawn
        assert chroma_ratio(resolving_box) <= GRAY_CHROMA_MAX  # ... and gray meanwhile


class TestBV4SettledUnknownIsGrayUnknown:
    """B-V4 (revised): a not-in-roster detection whose `data["resolving"]` is
    False (or absent) draws solid gray with the text label "Unknown" — and is
    never labelled "Instrument N", in any state, at any age."""

    def _draw_settled(self, renderer, roster, resolving=False) -> tuple:
        dets = _one(_LEFT_BOX, 9, resolving=resolving)
        with label_spy() as texts:
            result = _draw(renderer, dets, roster)
        return result, texts

    def test_b_v4_settled_not_in_roster_track_is_labelled_unknown(self) -> None:
        _, texts = self._draw_settled(_recording_renderer(), frozenset({1, 2, 3}))

        assert "Unknown" in texts

    def test_b_v4_settled_not_in_roster_track_is_gray(self) -> None:
        result, _ = self._draw_settled(_recording_renderer(), frozenset({1, 2, 3}))

        drawn = box_mean_bgr(result, _LEFT_BOX)
        assert drawn.max() > 0  # sanity: something was drawn
        assert chroma_ratio(drawn) <= GRAY_CHROMA_MAX

    def test_b_v4_a_track_with_no_resolving_flag_settles_immediately(self) -> None:
        # A producer that omits the flag (a bare double) gets the safe default:
        # settled Unknown, never a spurious spinner.
        _, texts = self._draw_settled(
            _recording_renderer(), frozenset({1, 2, 3}), resolving=None
        )

        assert "Unknown" in texts

    def test_b_v4_settled_not_in_roster_track_is_never_labelled_instrument_n(
        self,
    ) -> None:
        _, texts = self._draw_settled(_recording_renderer(), frozenset({1, 2, 3}))

        assert not drew_any_instrument_label(texts)

    def test_b_v4_a_long_lived_unknown_never_becomes_an_instrument(self) -> None:
        renderer = _recording_renderer()
        roster = frozenset({1, 2, 3})

        with label_spy() as texts:
            for step in range(40):  # 20s of continuous presence, always settled
                _draw(renderer, _one(_LEFT_BOX, 9, resolving=False), roster, _T0 + step * 0.5)

        assert not drew_any_instrument_label(texts)
        assert "Unknown" in texts

    def test_b_v4_resolving_and_settled_unknowns_are_gray_not_a_ninth_colour(self) -> None:
        # The whole point of B-V3/B-V4: an unknown must never read as one more
        # instrument. Both states of unknown stay gray while a roster instrument
        # in the same frame keeps its colour.
        renderer = _recording_renderer()
        roster = frozenset({1, 2, 3})
        settled_box, resolving_box, roster_box = (
            (20, 60, 100, 150),
            (110, 60, 190, 150),
            (200, 60, 250, 150),
        )

        result = renderer(
            blank_frame(*_FRAME_SHAPE),
            boxed_detections(
                [settled_box, resolving_box, roster_box],
                [9, 10, 2],
                _FRAME_SHAPE,
                resolving=[False, True, False],
            ),
            roster,
            _T0,
        )

        settled = box_mean_bgr(result, settled_box, inset=8)
        resolving = box_mean_bgr(result, resolving_box, inset=8)
        instrument = box_mean_bgr(result, roster_box, inset=8)
        assert settled.max() > 0 and resolving.max() > 0  # sanity: both were drawn
        assert chroma_ratio(settled) <= GRAY_CHROMA_MAX
        assert chroma_ratio(resolving) <= GRAY_CHROMA_MAX
        assert chroma_ratio(instrument) > GRAY_CHROMA_MAX  # ... the roster id is not gray


class TestResolvingIsAPerFrameFlagNotRendererMemory:
    """The wait-state fix (2026-07-16): the renderer keeps NO first-seen clock.
    Resolving is read fresh from `data["resolving"]` every frame, so it never
    latches — it can flip either way between frames and always reflects the
    linker's current decision, and a track can be born settled on frame one."""

    def test_a_settled_flag_shows_unknown_even_on_the_first_recording_frame(self) -> None:
        # No renderer-side "young" grace period: settled means settled at once.
        renderer = _recording_renderer()

        with label_spy() as texts:
            _draw(renderer, _one(_LEFT_BOX, 9, resolving=False), frozenset({1, 2, 3}))

        assert "Unknown" in texts

    def test_resolving_can_flip_to_settled_between_frames(self) -> None:
        # A deferred track that finally settles Unknown: was spinning, now "Unknown".
        renderer = _recording_renderer()
        roster = frozenset({1, 2, 3})

        with label_spy() as first:
            _draw(renderer, _one(_LEFT_BOX, 9, resolving=True), roster, _T0)
        with label_spy() as second:
            _draw(renderer, _one(_LEFT_BOX, 9, resolving=False), roster, _T0 + 0.1)

        assert "Unknown" not in first  # still deciding
        assert "Unknown" in second  # ... then settled

    def test_settled_does_not_latch_a_track_can_resume_resolving(self) -> None:
        # The mirror: a settled reading does not permanently mark the id — if the
        # linker reports it resolving again, the spinner comes back (no memory).
        renderer = _recording_renderer()
        roster = frozenset({1, 2, 3})

        with label_spy() as first:
            _draw(renderer, _one(_LEFT_BOX, 9, resolving=False), roster, _T0)
        with label_spy() as second:
            _draw(renderer, _one(_LEFT_BOX, 9, resolving=True), roster, _T0 + 0.1)

        assert "Unknown" in first
        assert "Unknown" not in second  # resolving again — no latched settlement
