"""T11/R1+R2 overlay tests, against `backend.render`.

REWRITTEN for T11/R1: there is no longer a separate setup appearance. The overlay
entry point `OverlayRenderer` (`__call__(frame, dets, roster, catalog, t)`) is
STATELESS and applies ONE roster-aware identity policy in EVERY phase — so the old
"set_recording(False) draws today's raw-track overlay" contract (T10 B-V1) is gone
and inverted here: setup now grays a not-in-roster object and labels a recognised
one `Instrument N`, exactly like recording. There is no recording flag any more.

T11/R2: known colours come from `catalog_colour(catalog, id)` — palette slot by
position in `sorted(catalog)` — NOT from the current roster, so a partial roster
can never shift an already-known instrument's colour (D5). `roster_colour` is gone.

T03's AC10 pair (draw in place; colour by track) is preserved, re-aimed at the
unified renderer. The resolving-vs-settled split (T10 B-V3/B-V4) is unchanged: it
is the linker's per-detection `data["resolving"]` flag, driven directly via
`boxed_detections(..., resolving=[...])`, never a renderer timer.

No threads, no camera, no real clock: `OverlayRenderer` is driven with hand-picked
`t` floats (still needed — the spinner animates with `t`). Frames are small so
mask/box math stays cheap, but not tiny — the probes sample box interiors, and
labels need room.

**Tunables are never hard-asserted** (per the spec's "Tunable parameters"): the
palette's 8 hex values are asserted DISTINCT and STABLE, never enumerated; the
gray is asserted achromatic (see `tests/overlay_probe.py`), never spelled out.
"""

from __future__ import annotations

import re

import numpy as np

from backend.render import CATALOG_PALETTE, UNKNOWN_COLOUR, OverlayRenderer, catalog_colour
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

# A representative demo catalog (the 8 loaded specimens), plus room to spare for
# tests that need ids outside it.
_CATALOG = frozenset(range(1, 13))


def _draw(renderer, dets, roster, catalog=_CATALOG, t: float = _T0) -> np.ndarray:
    return renderer(blank_frame(*_FRAME_SHAPE), dets, roster, catalog, t)


def _one(box, tracker_id: int, resolving: bool | None = None):
    return boxed_detections(
        [box],
        [tracker_id],
        _FRAME_SHAPE,
        resolving=None if resolving is None else [resolving],
    )


class TestOverlayIsRosterAwareInEveryPhase:
    """T11/R1: the SAME identity policy applies whether or not recording. A
    recognised (in-roster) id is coloured and labelled `Instrument N`; a
    not-in-roster id is gray — spinning while resolving, a settled `Unknown`
    otherwise — and is NEVER labelled `Instrument N`. Setup and recording are
    visually identical for unchanged detections. (Also carries T03's AC10.)"""

    def test_ac10_render_mutates_in_place_and_preserves_shape_and_dtype(self) -> None:
        frame = blank_frame(*_FRAME_SHAPE)
        before = frame.copy()
        renderer = OverlayRenderer()

        result = renderer(frame, _one(_LEFT_BOX, 7), frozenset({7}), _CATALOG, _T0)

        assert result is frame
        assert result.shape == before.shape
        assert result.dtype == before.dtype
        assert not np.array_equal(frame, before)

    def test_ac10_different_recognised_ids_get_different_colours(self) -> None:
        renderer = OverlayRenderer()
        dets = boxed_detections([_LEFT_BOX, _RIGHT_BOX], [1, 2], _FRAME_SHAPE)

        result = _draw(renderer, dets, frozenset({1, 2}))

        colour_a = box_mean_bgr(result, _LEFT_BOX)
        colour_b = box_mean_bgr(result, _RIGHT_BOX)
        assert not np.allclose(colour_a, colour_b, atol=5)

    def test_recognised_id_is_labelled_instrument_n(self) -> None:
        renderer = OverlayRenderer()

        with label_spy() as texts:
            _draw(renderer, _one(_LEFT_BOX, 5), frozenset({5, 6, 7}))

        assert "Instrument 5" in texts
        assert "Unknown" not in texts

    def test_not_in_roster_settled_id_is_gray_unknown_not_instrument(self) -> None:
        # Even in setup (a fresh renderer, never told recording): a not-in-roster
        # object is gray "Unknown", never "Instrument 9".
        renderer = OverlayRenderer()

        with label_spy() as texts:
            result = _draw(renderer, _one(_LEFT_BOX, 9, resolving=False), frozenset({1, 2, 3}))

        assert "Unknown" in texts
        assert "Instrument 9" not in texts
        drawn = box_mean_bgr(result, _LEFT_BOX)
        assert drawn.max() > 0  # sanity: something was drawn
        assert chroma_ratio(drawn) <= GRAY_CHROMA_MAX

    def test_not_in_roster_resolving_id_is_gray_with_no_label(self) -> None:
        renderer = OverlayRenderer()

        with label_spy() as texts:
            result = _draw(renderer, _one(_LEFT_BOX, 9, resolving=True), frozenset({1, 2, 3}))

        assert not drew_any_instrument_label(texts)
        assert "Unknown" not in texts  # still deciding — no settled label
        assert chroma_ratio(box_mean_bgr(result, _LEFT_BOX)) <= GRAY_CHROMA_MAX

    def test_render_is_a_pure_function_of_its_inputs(self) -> None:
        # R1's headline: the overlay carries no phase/recording state, so two fresh
        # renderers draw byte-identical frames for the same inputs — the setup frame
        # before Track and the first recording frame after Track cannot differ for
        # unchanged detections (a recognised id + a settled non-roster id).
        dets = boxed_detections(
            [_LEFT_BOX, _RIGHT_BOX], [2, 9], _FRAME_SHAPE, resolving=[False, False]
        )
        roster = frozenset({2})

        first = _draw(OverlayRenderer(), dets, roster)
        second = _draw(OverlayRenderer(), dets, roster)

        assert np.array_equal(first, second)


class TestRecognisedColourFollowsCatalog:
    """T11/R2: a recognised detection is drawn in `catalog_colour(catalog, id)` —
    the same hex `/status` reports — so a returned instrument regains it with no
    renderer memory, and the colour follows the CATALOG index, not the roster."""

    def test_recognised_mask_colour_matches_the_hex_catalog_colour_reports(self) -> None:
        catalog = frozenset({5, 6, 7})
        renderer = OverlayRenderer()

        result = _draw(renderer, _one(_LEFT_BOX, 5), frozenset({5}), catalog)

        drawn = box_mean_bgr(result, _LEFT_BOX)
        assert drawn.max() > 0  # sanity: something was drawn
        assert colour_similarity(drawn, catalog_colour(catalog, 5)) >= COLOUR_SIMILARITY_MIN

    def test_mask_colour_follows_the_catalog_index_not_the_tracker_id(self) -> None:
        # Same tracker_id 5, recognised in two catalogs that place it at different
        # sorted positions -> two different palette slots.
        as_index_0 = _draw(
            OverlayRenderer(), _one(_LEFT_BOX, 5), frozenset({5}), frozenset({5, 6, 7})
        )
        as_index_1 = _draw(
            OverlayRenderer(), _one(_LEFT_BOX, 5), frozenset({5}), frozenset({1, 5})
        )

        colour_0 = box_mean_bgr(as_index_0, _LEFT_BOX)
        colour_1 = box_mean_bgr(as_index_1, _LEFT_BOX)
        assert colour_0.max() > 0  # sanity: something was drawn
        assert not np.allclose(colour_0, colour_1, atol=5)

    def test_the_same_catalog_slot_draws_the_same_colour(self) -> None:
        # Different tracker_ids sitting at the same sorted position in their
        # catalogs share a colour.
        five_of_567 = _draw(
            OverlayRenderer(), _one(_LEFT_BOX, 5), frozenset({5}), frozenset({5, 6, 7})
        )
        two_of_2_9_11 = _draw(
            OverlayRenderer(), _one(_LEFT_BOX, 2), frozenset({2}), frozenset({2, 9, 11})
        )

        assert np.allclose(
            box_mean_bgr(five_of_567, _LEFT_BOX),
            box_mean_bgr(two_of_2_9_11, _LEFT_BOX),
            atol=5,
        )

    def test_a_partial_roster_does_not_shift_a_known_colour(self) -> None:
        # The D5 payoff: with the full catalog {1..8}, id 5's colour is the same
        # whether only 5 is recognised or all eight are — because the colour keys
        # off the CATALOG, not the (growing) roster. roster_colour got this wrong.
        catalog = frozenset(range(1, 9))
        alone = _draw(OverlayRenderer(), _one(_LEFT_BOX, 5), frozenset({5}), catalog)
        full = _draw(
            OverlayRenderer(), _one(_LEFT_BOX, 5), frozenset(range(1, 9)), catalog
        )

        colour_alone = box_mean_bgr(alone, _LEFT_BOX)
        colour_full = box_mean_bgr(full, _LEFT_BOX)
        assert colour_alone.max() > 0  # sanity
        assert np.allclose(colour_alone, colour_full, atol=5)
        assert colour_similarity(colour_alone, catalog_colour(catalog, 5)) >= COLOUR_SIMILARITY_MIN

    def test_two_recognised_ids_in_one_frame_get_different_colours(self) -> None:
        catalog = frozenset({1, 2, 3})
        renderer = OverlayRenderer()
        dets = boxed_detections([_LEFT_BOX, _RIGHT_BOX], [1, 2], _FRAME_SHAPE)

        result = _draw(renderer, dets, frozenset({1, 2}), catalog)

        assert not np.allclose(
            box_mean_bgr(result, _LEFT_BOX), box_mean_bgr(result, _RIGHT_BOX), atol=5
        )

    def test_a_returned_instrument_regains_the_colour_the_api_reports(self) -> None:
        catalog = frozenset({1, 2, 3})
        roster = frozenset({1, 2, 3})
        renderer = OverlayRenderer()
        # id 2 sits in the LEFT box; id 1 stays put in the RIGHT one throughout,
        # so the sampled box only ever holds id 2.
        both = boxed_detections([_LEFT_BOX, _RIGHT_BOX], [2, 1], _FRAME_SHAPE)
        without_id_2 = _one(_RIGHT_BOX, 1)

        before = _draw(renderer, both, roster, catalog, _T0)
        for step in range(1, 30):  # id 2 off the table for 30 s
            _draw(renderer, without_id_2, roster, catalog, _T0 + step)
        after = _draw(renderer, both, roster, catalog, _T0 + 30.0)

        colour_before = box_mean_bgr(before, _LEFT_BOX)
        colour_after = box_mean_bgr(after, _LEFT_BOX)
        assert colour_similarity(colour_before, catalog_colour(catalog, 2)) >= COLOUR_SIMILARITY_MIN
        assert np.allclose(colour_before, colour_after, atol=5)


class TestCatalogColourIsAPureStablePalette:
    """T11/R2: `catalog_colour(catalog, id)` = palette[index of id in
    sorted(catalog) mod 8] — pure, and stable for the tracker's whole life because
    the catalog is constant. The 8 hex VALUES are a tunable: asserted distinct and
    well-formed, never enumerated. An id outside the catalog gets the gray."""

    def test_catalog_palette_holds_eight_distinct_colours(self) -> None:
        assert len(CATALOG_PALETTE) == 8
        assert len(set(CATALOG_PALETTE)) == 8

    def test_catalog_palette_entries_are_hex_colour_strings(self) -> None:
        for colour in CATALOG_PALETTE:
            assert re.fullmatch(r"#[0-9a-fA-F]{6}", colour), colour

    def test_catalog_colour_indexes_the_palette_by_sorted_position(self) -> None:
        catalog = frozenset({4, 11, 2, 7})

        for index, specimen_id in enumerate(sorted(catalog)):
            assert catalog_colour(catalog, specimen_id) == CATALOG_PALETTE[index]

    def test_catalog_colour_gives_a_full_eight_catalog_eight_distinct_colours(self) -> None:
        catalog = frozenset(range(1, 9))

        colours = [catalog_colour(catalog, specimen_id) for specimen_id in sorted(catalog)]

        assert len(set(colours)) == 8

    def test_catalog_colour_wraps_the_palette_past_the_eighth_id(self) -> None:
        catalog = frozenset(range(1, 11))  # 10 ids, 8 colours
        ordered = sorted(catalog)

        colours = [catalog_colour(catalog, specimen_id) for specimen_id in ordered]

        assert len(set(colours[:8])) == 8  # the first eight are all distinct
        assert colours[8] == colours[0]  # ... then it wraps
        assert colours[9] == colours[1]

    def test_catalog_colour_is_gray_for_an_id_outside_the_catalog(self) -> None:
        assert catalog_colour(frozenset({1, 2, 3}), 9) == UNKNOWN_COLOUR
        assert catalog_colour(frozenset(), 1) == UNKNOWN_COLOUR

    def test_catalog_colour_is_stable_and_order_independent(self) -> None:
        catalog = frozenset({3, 5, 7})

        first_pass = [catalog_colour(catalog, i) for i in (7, 3, 5)]
        second_pass = [catalog_colour(catalog, i) for i in (3, 5, 7)]

        assert catalog_colour(catalog, 5) == catalog_colour(catalog, 5)
        assert first_pass == [second_pass[2], second_pass[0], second_pass[1]]


class TestResolvingUnknownIsGrayAndSpins:
    """T11/B-V3 (unchanged behaviour, now phase-independent): a not-in-roster
    detection whose `data["resolving"]` is True draws gray, carries NO
    "Instrument" and NO "Unknown" label, and shows a resolving indicator that
    visibly animates across frames. The flag is the linker's wait-state, not a
    timer."""

    def test_resolving_not_in_roster_track_is_gray(self) -> None:
        result = _draw(OverlayRenderer(), _one(_LEFT_BOX, 9, resolving=True), frozenset({1, 2, 3}))

        drawn = box_mean_bgr(result, _LEFT_BOX)
        assert drawn.max() > 0  # sanity: something was drawn
        assert chroma_ratio(drawn) <= GRAY_CHROMA_MAX

    def test_resolving_not_in_roster_track_carries_no_settled_label(self) -> None:
        with label_spy() as texts:
            _draw(OverlayRenderer(), _one(_LEFT_BOX, 9, resolving=True), frozenset({1, 2, 3}))

        assert not drew_any_instrument_label(texts)
        assert "Unknown" not in texts  # still deciding — no settled label yet

    def test_the_resolving_indicator_animates_across_frames(self) -> None:
        # Identical detections, identical black frame, only `t` moves: any pixel
        # difference is the resolving indicator doing its job.
        renderer = OverlayRenderer()
        roster = frozenset({1, 2, 3})
        dets = _one(_LEFT_BOX, 9, resolving=True)

        frames = [
            _draw(renderer, dets, roster, _CATALOG, _T0 + step * 0.1).tobytes()
            for step in range(6)  # 0.0 .. 0.5 s
        ]

        assert len(set(frames)) >= 2

    def test_resolving_is_decided_per_row_not_per_renderer(self) -> None:
        # Two not-in-roster ids in ONE frame with opposite flags: the resolving
        # one spins gray with no label, the settled one is a gray "Unknown".
        renderer = OverlayRenderer()
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


class TestSettledUnknownIsGrayUnknown:
    """T11/B-V4 (unchanged behaviour, now phase-independent): a not-in-roster
    detection whose `data["resolving"]` is False (or absent) draws solid gray with
    the text label "Unknown" — and is never labelled "Instrument N", in any state,
    at any age."""

    def _draw_settled(self, roster, resolving=False) -> tuple:
        dets = _one(_LEFT_BOX, 9, resolving=resolving)
        with label_spy() as texts:
            result = _draw(OverlayRenderer(), dets, roster)
        return result, texts

    def test_settled_not_in_roster_track_is_labelled_unknown(self) -> None:
        _, texts = self._draw_settled(frozenset({1, 2, 3}))

        assert "Unknown" in texts

    def test_settled_not_in_roster_track_is_gray(self) -> None:
        result, _ = self._draw_settled(frozenset({1, 2, 3}))

        drawn = box_mean_bgr(result, _LEFT_BOX)
        assert drawn.max() > 0  # sanity: something was drawn
        assert chroma_ratio(drawn) <= GRAY_CHROMA_MAX

    def test_a_track_with_no_resolving_flag_settles_immediately(self) -> None:
        # A producer that omits the flag (a bare double) gets the safe default:
        # settled Unknown, never a spurious spinner.
        _, texts = self._draw_settled(frozenset({1, 2, 3}), resolving=None)

        assert "Unknown" in texts

    def test_settled_not_in_roster_track_is_never_labelled_instrument_n(self) -> None:
        _, texts = self._draw_settled(frozenset({1, 2, 3}))

        assert not drew_any_instrument_label(texts)

    def test_a_long_lived_unknown_never_becomes_an_instrument(self) -> None:
        renderer = OverlayRenderer()
        roster = frozenset({1, 2, 3})

        with label_spy() as texts:
            for step in range(40):  # 20 s of continuous presence, always settled
                _draw(renderer, _one(_LEFT_BOX, 9, resolving=False), roster, _CATALOG, _T0 + step * 0.5)

        assert not drew_any_instrument_label(texts)
        assert "Unknown" in texts

    def test_resolving_and_settled_unknowns_are_gray_not_a_ninth_colour(self) -> None:
        # The whole point of B-V3/B-V4: an unknown must never read as one more
        # instrument. Both states of unknown stay gray while a recognised
        # instrument in the same frame keeps its colour.
        renderer = OverlayRenderer()
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
            _CATALOG,
            _T0,
        )

        settled = box_mean_bgr(result, settled_box, inset=8)
        resolving = box_mean_bgr(result, resolving_box, inset=8)
        instrument = box_mean_bgr(result, roster_box, inset=8)
        assert settled.max() > 0 and resolving.max() > 0  # sanity: both were drawn
        assert chroma_ratio(settled) <= GRAY_CHROMA_MAX
        assert chroma_ratio(resolving) <= GRAY_CHROMA_MAX
        assert chroma_ratio(instrument) > GRAY_CHROMA_MAX  # ... the recognised id is not gray


class TestResolvingIsAPerFrameFlagNotRendererMemory:
    """The wait-state fix (2026-07-16): the renderer keeps NO first-seen clock.
    Resolving is read fresh from `data["resolving"]` every frame, so it never
    latches — it can flip either way between frames and always reflects the
    linker's current decision, and a track can be born settled on frame one."""

    def test_a_settled_flag_shows_unknown_even_on_the_first_frame(self) -> None:
        with label_spy() as texts:
            _draw(OverlayRenderer(), _one(_LEFT_BOX, 9, resolving=False), frozenset({1, 2, 3}))

        assert "Unknown" in texts

    def test_resolving_can_flip_to_settled_between_frames(self) -> None:
        renderer = OverlayRenderer()
        roster = frozenset({1, 2, 3})

        with label_spy() as first:
            _draw(renderer, _one(_LEFT_BOX, 9, resolving=True), roster, _CATALOG, _T0)
        with label_spy() as second:
            _draw(renderer, _one(_LEFT_BOX, 9, resolving=False), roster, _CATALOG, _T0 + 0.1)

        assert "Unknown" not in first  # still deciding
        assert "Unknown" in second  # ... then settled

    def test_settled_does_not_latch_a_track_can_resume_resolving(self) -> None:
        renderer = OverlayRenderer()
        roster = frozenset({1, 2, 3})

        with label_spy() as first:
            _draw(renderer, _one(_LEFT_BOX, 9, resolving=False), roster, _CATALOG, _T0)
        with label_spy() as second:
            _draw(renderer, _one(_LEFT_BOX, 9, resolving=True), roster, _CATALOG, _T0 + 0.1)

        assert "Unknown" in first
        assert "Unknown" not in second  # resolving again — no latched settlement
