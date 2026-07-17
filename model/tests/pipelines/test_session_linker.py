"""Behavioral tests for `SessionLinker` (model/docs/linker-design.md).

These tests exercise the public behavior contract through `update()`,
`reset()`, and the `roster`
property. They inject a small scriptable fake matcher (the ChampionMethod
duck-type: build_gallery/score/accept) so no HuggingFace model ever loads.

Do NOT hard-assert tunable parameter values/formulas (enrolment_window_s,
evidence_window_s, evidence_frames, absent_death_s, min_mask_area_px, the
quality-ranking formula, log formats) — see the design's parameter table.
Window/threshold frame counts used below are always derived from the
constructor args passed in each test via `round(seconds * fps)`, never
hardcoded independently of those args.

T08 (`docs/wayfinder/session-linker/assets/T08-T10/T08-SPEC.md`) replaced two
conventions this file used to encode, so read the id arithmetic carefully:

  * `session_id == raw_id` is GONE (SPEC B-N2/B-N3). Session ids are assigned
    at the freeze: bound identities take their specimen number, session-only
    identities take `base + k` for k = 1, 2, ... in ascending RAW-id order,
    where `base` is the highest loaded specimen number (0 when no persistent
    galleries were loaded). Every linker built WITHOUT persistent galleries
    therefore renumbers its roster to a contiguous `{1..N}` — so a fake
    matcher scripted by session id must be keyed by the RENUMBERED id, not by
    the raw tracker id the test fed in.
  * "unknown emits its raw id" is GONE (SPEC B-N4). Pending, deferred and
    settled-Unknown tracks emit `raw_id + unknown_id_offset`. Tests pass an
    EXPLICIT `unknown_id_offset` (never assert the default constant — it is a
    tunable) and assert the offset arithmetic, which is what keeps the roster
    and the unknown id space disjoint.
"""

from __future__ import annotations

import logging
import re

import numpy as np
import supervision as sv

from orc_model.pipelines.matching.interface import REJECT
from orc_model.pipelines.session_linker import SessionLinker

# --------------------------------------------------------------------------
# Frame / detections synthesis helpers
# --------------------------------------------------------------------------

FRAME_H = 64
FRAME_W = 64
BACKGROUND_BGR = (128, 128, 128)

# Non-overlapping boxes, all safely inset from the frame edge (B6.3's
# quality gate requires the box not touch/exceed the frame edge).
BOX_A = (4, 4, 24, 24)
BOX_B = (40, 4, 60, 24)
BOX_C = (4, 40, 24, 60)
BOX_D = (40, 40, 60, 60)

# An explicit, deliberately non-default offset (SPEC "Tunable parameters":
# never assert DEFAULT_UNKNOWN_ID_OFFSET). Chosen well clear of every raw id
# used below so `raw_id + UNKNOWN_OFFSET` can never collide with a session id.
UNKNOWN_OFFSET = 5000


def new_frame() -> np.ndarray:
    """A fresh BGR background frame — never reused/mutated across calls."""
    return np.full((FRAME_H, FRAME_W, 3), BACKGROUND_BGR, dtype=np.uint8)


def rgb_marker_for_bgr(bgr):
    """The marker FakeMatcher.score() will recover from a painted row's
    RGB-converted crop (see FakeMatcher docstring)."""
    b, g, r = bgr
    return (r, g, b)


def build_call(rows, data=None):
    """rows: list of dicts with keys tracker_id, box, bgr, (confidence, class_id).

    Paints each row's box into a fresh frame and returns (frame, detections).
    Every synthesized row is deliberately "quality": mask area well above any
    min_mask_area_px used in these tests, box inset from the frame edge.
    """
    frame = new_frame()
    n = len(rows)
    xyxy = np.zeros((n, 4), dtype=np.float32)
    confidence = np.zeros((n,), dtype=np.float32)
    class_id = np.zeros((n,), dtype=np.int64)
    tracker_id = np.zeros((n,), dtype=np.int64)
    mask = np.zeros((n, FRAME_H, FRAME_W), dtype=bool)
    for i, row in enumerate(rows):
        x0, y0, x1, y1 = row["box"]
        xyxy[i] = (x0, y0, x1, y1)
        confidence[i] = row.get("confidence", 0.9)
        class_id[i] = row.get("class_id", 0)
        tracker_id[i] = row["tracker_id"]
        mask[i, y0:y1, x0:x1] = True
        frame[y0:y1, x0:x1] = row["bgr"]
    kwargs = dict(xyxy=xyxy, confidence=confidence, class_id=class_id, tracker_id=tracker_id, mask=mask)
    if data is not None:
        kwargs["data"] = data
    detections = sv.Detections(**kwargs)
    return frame, detections


def empty_call():
    return new_frame(), sv.Detections.empty()


def output_tracker_id(out: sv.Detections, rows, raw_id: int) -> int:
    """The OUTPUT tracker_id at the row position `raw_id` was placed at
    (B1.1/B1.2 guarantee row-alignment; only tracker_id may change)."""
    idx = next(i for i, r in enumerate(rows) if r["tracker_id"] == raw_id)
    assert len(out) > idx, f"expected an output row for raw id {raw_id}, got {len(out)} row(s)"
    return int(out.tracker_id[idx])


# --------------------------------------------------------------------------
# Fake matcher — ChampionMethod duck-type
# --------------------------------------------------------------------------


class FakeGallery:
    """Opaque gallery stand-in: `.views` (n, D) ndarray + `.identity`."""

    def __init__(self, views: np.ndarray, identity=None):
        self.views = views
        self.identity = identity


def persistent_gallery(specimen: int, n_views: int = 5) -> FakeGallery:
    """A pre-embedded persistent specimen gallery, as `load_persistent_galleries`
    would hand to the constructor. D=4 matches FakeMatcher.build_gallery's views
    so `persistent ∪ Start` concatenates (SPEC B-B5)."""
    return FakeGallery(views=np.zeros((n_views, 4), dtype=np.float32), identity=specimen)


class FakeMatcher:
    """Scriptable fake: build_gallery/score/accept.

    `program(marker, responses)` queues one (scores_dict, accept_result) pair
    per expected score()/accept() decision for that marker, consumed FIFO.
    `marker` is a hashable id a test
    controls by painting a distinct solid color into the query row's box:
    score() recovers it from the RGB-converted crop's first pixel, so
    scripting never depends on matcher call order across rows in a batch.

    accept() re-derives which marker a `scores` dict belongs to by matching
    its *content* against the currently staged entry — robust whether the
    real implementation passes the exact object score() returned or an
    equal copy of it.

    NOTE for T08 binding (SPEC B-B2/B-B3): the enrolment-freeze bind calls
    `score()` and applies its OWN tau/margin gate — it never calls `accept()`.
    So a binding script is programmed as `[(scores, None)]`: `score()` reads
    `queue[0]` without popping, and the unused accept slot is never consumed.
    """

    def __init__(self):
        self.build_gallery_calls: list[dict] = []
        self.score_calls: list[dict] = []
        self.accept_calls: list[dict] = []
        self._scripts: dict[tuple, list[tuple[dict, object]]] = {}

    def program(self, marker, responses):
        self._scripts[marker] = list(responses)

    def build_gallery(self, crops, masks, meta):
        self.build_gallery_calls.append(
            {
                "crops": [c.copy() for c in crops],
                "masks": [m.copy() for m in masks],
                "meta": dict(meta),
            }
        )
        n = max(len(crops), 1)
        return FakeGallery(views=np.zeros((n, 4), dtype=np.float32))

    def score(self, query_crops, query_masks, meta, galleries):
        marker = tuple(int(v) for v in query_crops[0][0, 0])
        queue = self._scripts.get(marker)
        assert queue, f"FakeMatcher.score() called for marker {marker} with no programmed response left"
        scores, _ = queue[0]
        self.score_calls.append(
            {
                "marker": marker,
                "meta": dict(meta),
                "gallery_keys": frozenset(galleries.keys()),
                "gallery_view_counts": {k: len(g.views) for k, g in galleries.items()},
                "crops": [c.copy() for c in query_crops],
            }
        )
        return dict(scores)

    def accept(self, scores):
        marker = self._match_marker(scores)
        queue = self._scripts[marker]
        _, accept_result = queue.pop(0)
        self.accept_calls.append({"marker": marker, "result": accept_result})
        return accept_result

    def _match_marker(self, scores):
        for marker, queue in self._scripts.items():
            if queue and queue[0][0] == scores:
                return marker
        raise AssertionError(f"FakeMatcher.accept() called with unrecognized scores {scores}")


def crop_markers(crops):
    return {tuple(int(v) for v in c[0, 0]) for c in crops}


def build_gallery_calls_for_marker(matcher: FakeMatcher, marker) -> list[dict]:
    """The build_gallery() calls whose crops carry `marker`'s paint.

    Identifying a gallery build by its CROP CONTENT rather than by
    `meta["identity"]` keeps these assertions off an internal the SPEC never
    pins down: at the enrolment freeze the Start galleries are built BEFORE
    binding resolves (B-B1), so a session id does not exist yet and the meta
    tag's value there is the implementation's business, not the contract's.
    """
    return [c for c in matcher.build_gallery_calls if marker in crop_markers(c["crops"])]


def score_calls_for_marker(matcher: FakeMatcher, marker) -> list[dict]:
    return [c for c in matcher.score_calls if c["marker"] == marker]


# --------------------------------------------------------------------------
# B1 — Emission contract
# --------------------------------------------------------------------------


def test_b1_2_passthrough_strictly_before_the_freeze():
    fps = 4.0
    enrolment_window_s = 0.75
    window = round(enrolment_window_s * fps)  # frames the enrolment window spans
    matcher = FakeMatcher()
    linker = SessionLinker(
        matcher,
        fps=fps,
        unknown_id_offset=UNKNOWN_OFFSET,
        enrolment_window_s=enrolment_window_s,
        evidence_window_s=10.0,
        evidence_frames=1,
        absent_death_s=10.0,
        min_mask_area_px=100,
    )
    raw_id = 111
    # Every call STRICTLY BEFORE the freeze is a pure pass-through (B-N5). The
    # boundary is exclusive: the last call of the window IS the freeze, and it
    # belongs to the post-freeze side of the contract (asserted below).
    for i in range(window - 1):
        rows = [{"tracker_id": raw_id, "box": BOX_A, "bgr": (10, 10, 10)}]
        frame, dets = build_call(rows)
        out = linker.update(dets, frame)
        assert output_tracker_id(out, rows, raw_id) == raw_id, (
            f"call {i + 1}/{window}: enrolment window still open, expected pure pass-through"
        )
        assert linker.roster == frozenset(), "roster must stay empty until the window closes"

    # The freeze call itself: the roster publishes and the emitted id flips
    # raw -> session on this very call, not the following one (B-N5, clarified).
    rows = [{"tracker_id": raw_id, "box": BOX_A, "bgr": (10, 10, 10)}]
    frame, dets = build_call(rows)
    out = linker.update(dets, frame)
    # B-N2/B-N3: no persistent galleries -> base = 0 -> the sole session-only
    # identity is renumbered to 1, NOT promoted to its raw id.
    assert output_tracker_id(out, rows, raw_id) == 1, "the freeze frame itself already emits the session id"
    assert linker.roster == frozenset({1}), "window closed on the last call; sole id had 100% presence"


def test_b1_3_empty_frames_advance_the_enrolment_window_clock():
    fps = 4.0
    enrolment_window_s = 0.75
    window = round(enrolment_window_s * fps)  # chosen so window == 3: 2-of-3 presences is a majority
    matcher = FakeMatcher()
    linker = SessionLinker(
        matcher,
        fps=fps,
        unknown_id_offset=UNKNOWN_OFFSET,
        enrolment_window_s=enrolment_window_s,
        evidence_window_s=10.0,
        evidence_frames=1,
        absent_death_s=10.0,
        min_mask_area_px=100,
    )
    raw_id = 2101

    # Call 1: opens the window.
    rows = [{"tracker_id": raw_id, "box": BOX_A, "bgr": (60, 70, 80)}]
    frame, dets = build_call(rows)
    linker.update(dets, frame)
    assert linker.roster == frozenset()

    # Middle calls: EMPTY frames. If they didn't advance the clock, the
    # window would still be waiting for content and would never close on
    # the schedule this test checks below.
    for _ in range(window - 2):
        frame, dets = empty_call()
        out = linker.update(dets, frame)
        assert len(out) == 0
        assert linker.roster == frozenset()

    # Final (window-th) call: non-empty again, closing the window. raw_id
    # was present in 2 of `window` calls — a majority — so it enrolls.
    rows = [{"tracker_id": raw_id, "box": BOX_A, "bgr": (60, 70, 80)}]
    frame, dets = build_call(rows)
    linker.update(dets, frame)
    assert linker.roster == frozenset({1})


def test_b1_1_output_is_new_object_and_input_is_never_mutated():
    fps = 4.0
    enrolment_window_s = 0.25
    window = round(enrolment_window_s * fps)
    matcher = FakeMatcher()
    linker = SessionLinker(
        matcher,
        fps=fps,
        unknown_id_offset=UNKNOWN_OFFSET,
        enrolment_window_s=enrolment_window_s,
        evidence_window_s=10.0,
        evidence_frames=1,
        absent_death_s=10.0,
        min_mask_area_px=100,
    )
    assert window >= 1  # both rows below enrol in a single call regardless

    rows = [
        {"tracker_id": 2401, "box": BOX_A, "bgr": (12, 34, 56), "confidence": 0.81, "class_id": 2},
        {"tracker_id": 2402, "box": BOX_B, "bgr": (65, 43, 21), "confidence": 0.42, "class_id": 5},
    ]
    data = {"tag": np.array(["first", "second"])}
    frame, dets = build_call(rows, data=data)

    xyxy_before = dets.xyxy.copy()
    confidence_before = dets.confidence.copy()
    class_id_before = dets.class_id.copy()
    tracker_id_before = dets.tracker_id.copy()
    mask_before = dets.mask.copy()
    data_before = {k: np.array(v, copy=True) for k, v in dets.data.items()}
    frame_before = frame.copy()

    out = linker.update(dets, frame)

    assert out is not dets, "update() must return a new/derived Detections object"
    assert len(out) == len(rows), "row count must be preserved"

    # Input must be untouched.
    assert np.array_equal(dets.xyxy, xyxy_before)
    assert np.array_equal(dets.confidence, confidence_before)
    assert np.array_equal(dets.class_id, class_id_before)
    assert np.array_equal(dets.tracker_id, tracker_id_before)
    assert np.array_equal(dets.mask, mask_before)
    for key, value in data_before.items():
        assert np.array_equal(dets.data[key], value)
    assert np.array_equal(frame, frame_before), "the input frame must never be mutated"

    # Output fields other than tracker_id must be row-aligned copies of the input.
    assert np.array_equal(out.xyxy, xyxy_before)
    assert np.array_equal(out.confidence, confidence_before)
    assert np.array_equal(out.class_id, class_id_before)
    assert np.array_equal(out.mask, mask_before)
    for key, value in data_before.items():
        assert key in out.data
        assert np.array_equal(out.data[key], value)


def test_b1_4_return_reemits_original_id_from_decision_call_onward():
    fps = 4.0
    enrolment_window_s = 0.25
    window = round(enrolment_window_s * fps)
    absent_death_s = 0.5
    death_threshold = round(absent_death_s * fps)
    matcher = FakeMatcher()
    linker = SessionLinker(
        matcher,
        fps=fps,
        unknown_id_offset=UNKNOWN_OFFSET,
        enrolment_window_s=enrolment_window_s,
        evidence_window_s=10.0,
        evidence_frames=2,  # two quality crops needed, so we can observe Pending vs decided
        absent_death_s=absent_death_s,
        min_mask_area_px=100,
    )
    raw_roster_id = 601
    session_id = 1  # sole session-only identity, no galleries -> base 0 + k=1
    for _ in range(window):
        rows = [{"tracker_id": raw_roster_id, "box": BOX_A, "bgr": (80, 80, 80)}]
        frame, dets = build_call(rows)
        linker.update(dets, frame)
    assert linker.roster == frozenset({session_id})

    for _ in range(death_threshold + 1):  # "more than" the threshold -> dies
        frame, dets = empty_call()
        linker.update(dets, frame)

    return_bgr = (11, 22, 33)
    marker = rgb_marker_for_bgr(return_bgr)
    matcher.program(marker, [({session_id: 0.9}, session_id)])
    return_raw_id = 602

    # First quality appearance: evidence window not yet closed -> Pending
    # -> emits the offset unknown id, no decision, no score() call yet.
    rows = [{"tracker_id": return_raw_id, "box": BOX_B, "bgr": return_bgr}]
    frame, dets = build_call(rows)
    out = linker.update(dets, frame)
    assert output_tracker_id(out, rows, return_raw_id) == return_raw_id + UNKNOWN_OFFSET
    assert not matcher.score_calls

    # Second quality appearance: the evidence window closes THIS call, the
    # batched decision resolves synchronously, and THIS call's own output
    # already carries the original session id, not the offset unknown id.
    rows = [{"tracker_id": return_raw_id, "box": BOX_B, "bgr": return_bgr}]
    frame, dets = build_call(rows)
    out = linker.update(dets, frame)
    assert output_tracker_id(out, rows, return_raw_id) == session_id

    # The mapping persists for the rest of the recording.
    rows = [{"tracker_id": return_raw_id, "box": BOX_B, "bgr": return_bgr}]
    frame, dets = build_call(rows)
    out = linker.update(dets, frame)
    assert output_tracker_id(out, rows, return_raw_id) == session_id


# --------------------------------------------------------------------------
# B2 — Enrolment & roster freeze
# --------------------------------------------------------------------------


def test_b2_roster_freezes_after_enrolment_window_majority_rule():
    fps = 4.0
    enrolment_window_s = 0.75
    window = round(enrolment_window_s * fps)  # chosen so window == 3
    matcher = FakeMatcher()
    linker = SessionLinker(
        matcher,
        fps=fps,
        unknown_id_offset=UNKNOWN_OFFSET,
        enrolment_window_s=enrolment_window_s,
        evidence_window_s=10.0,
        evidence_frames=1,
        absent_death_s=10.0,
        min_mask_area_px=100,
    )
    id_full_presence = 211  # present every call -> majority, enrolls
    id_minority = 212  # present on only the first call -> not a majority, does not enrol
    id_late_majority = 213  # present on all calls but the first -> still a majority, enrols

    for i in range(window):
        rows = [{"tracker_id": id_full_presence, "box": BOX_A, "bgr": (10, 10, 10)}]
        if i == 0:
            rows.append({"tracker_id": id_minority, "box": BOX_B, "bgr": (20, 20, 20)})
        else:
            rows.append({"tracker_id": id_late_majority, "box": BOX_C, "bgr": (30, 30, 30)})
        frame, dets = build_call(rows)
        linker.update(dets, frame)
        if i < window - 1:
            assert linker.roster == frozenset()

    # The two enrolled raw ids (211, 213) renumber to 1 and 2 in ascending
    # raw-id order; 212 never enrolled and so consumes no session number.
    assert linker.roster == frozenset({1, 2})


def test_b2_roster_property_is_frozenset_and_empty_before_freeze():
    fps = 4.0
    enrolment_window_s = 0.25
    window = round(enrolment_window_s * fps)  # freezes on the very first call
    matcher = FakeMatcher()
    linker = SessionLinker(
        matcher,
        fps=fps,
        unknown_id_offset=UNKNOWN_OFFSET,
        enrolment_window_s=enrolment_window_s,
        evidence_window_s=10.0,
        evidence_frames=1,
        absent_death_s=10.0,
        min_mask_area_px=100,
    )
    assert linker.roster == frozenset()
    assert isinstance(linker.roster, frozenset)

    raw_id = 221
    for _ in range(window):
        rows = [{"tracker_id": raw_id, "box": BOX_A, "bgr": (40, 40, 40)}]
        frame, dets = build_call(rows)
        linker.update(dets, frame)

    assert linker.roster == frozenset({1})
    assert isinstance(linker.roster, frozenset)


def test_b2_post_freeze_newcomer_never_joins_the_roster_and_emits_an_offset_id():
    fps = 4.0
    enrolment_window_s = 0.25
    window = round(enrolment_window_s * fps)
    matcher = FakeMatcher()
    linker = SessionLinker(
        matcher,
        fps=fps,
        unknown_id_offset=UNKNOWN_OFFSET,
        enrolment_window_s=enrolment_window_s,
        evidence_window_s=100.0,
        evidence_frames=5,  # kept Pending for the whole test, never decided
        absent_death_s=100.0,
        min_mask_area_px=100,
    )
    raw_roster_id = 231
    session_id = 1
    for _ in range(window):
        rows = [{"tracker_id": raw_roster_id, "box": BOX_A, "bgr": (50, 50, 50)}]
        frame, dets = build_call(rows)
        linker.update(dets, frame)
    assert linker.roster == frozenset({session_id})

    newcomer_id = 331  # born strictly after the freeze -> never joins the roster
    for _ in range(3):
        rows = [
            {"tracker_id": raw_roster_id, "box": BOX_A, "bgr": (50, 50, 50)},
            {"tracker_id": newcomer_id, "box": BOX_B, "bgr": (60, 60, 60)},
        ]
        frame, dets = build_call(rows)
        out = linker.update(dets, frame)
        # B-N4: a Pending track inside its evidence window emits an offset id.
        assert output_tracker_id(out, rows, newcomer_id) == newcomer_id + UNKNOWN_OFFSET
        assert output_tracker_id(out, rows, raw_roster_id) == session_id
    assert linker.roster == frozenset({session_id}), "a post-freeze newcomer must never join the frozen roster"


# --------------------------------------------------------------------------
# B3 — Death (Active -> Missing)
# --------------------------------------------------------------------------


def test_b3_death_after_absence_threshold_marks_identity_missing():
    fps = 4.0
    enrolment_window_s = 0.25
    window = round(enrolment_window_s * fps)
    absent_death_s = 0.5
    death_threshold = round(absent_death_s * fps)
    matcher = FakeMatcher()
    linker = SessionLinker(
        matcher,
        fps=fps,
        unknown_id_offset=UNKNOWN_OFFSET,
        enrolment_window_s=enrolment_window_s,
        evidence_window_s=10.0,
        evidence_frames=1,
        absent_death_s=absent_death_s,
        min_mask_area_px=100,
    )
    raw_roster_id = 401
    session_id = 1
    for _ in range(window):
        rows = [{"tracker_id": raw_roster_id, "box": BOX_A, "bgr": (70, 70, 70)}]
        frame, dets = build_call(rows)
        linker.update(dets, frame)
    assert linker.roster == frozenset({session_id})

    for _ in range(death_threshold + 1):  # "more than" the threshold -> dies
        frame, dets = empty_call()
        linker.update(dets, frame)

    return_bgr = (5, 15, 200)
    marker = rgb_marker_for_bgr(return_bgr)
    matcher.program(marker, [({session_id: 0.9}, session_id)])
    return_raw_id = 402
    rows = [{"tracker_id": return_raw_id, "box": BOX_B, "bgr": return_bgr}]
    frame, dets = build_call(rows)
    linker.update(dets, frame)

    matching_calls = score_calls_for_marker(matcher, marker)
    assert matching_calls, "expected the return track's evidence window to close and trigger a scoring call"
    assert session_id in matching_calls[0]["gallery_keys"], (
        "the session identity must be an admissible scoring candidate — proves it moved Active -> Missing "
        "after the absence threshold was crossed"
    )


def test_b3_raw_id_reactivation_after_death_must_pass_the_matcher_again():
    fps = 4.0
    absent_death_s = 0.5
    death_threshold = round(absent_death_s * fps)
    matcher = FakeMatcher()
    linker = SessionLinker(
        matcher,
        fps=fps,
        unknown_id_offset=UNKNOWN_OFFSET,
        enrolment_window_s=0.25,
        evidence_window_s=10.0,
        evidence_frames=1,
        absent_death_s=absent_death_s,
        min_mask_area_px=100,
    )
    raw_roster_id = 451
    session_id = 1
    rows = [{"tracker_id": raw_roster_id, "box": BOX_A, "bgr": (70, 70, 70)}]
    frame, dets = build_call(rows)
    linker.update(dets, frame)
    assert linker.roster == frozenset({session_id})

    for _ in range(death_threshold + 1):
        frame, dets = empty_call()
        linker.update(dets, frame)

    reactivated_bgr = (5, 15, 200)
    marker = rgb_marker_for_bgr(reactivated_bgr)
    matcher.program(marker, [({session_id: 0.1}, REJECT)])
    rows = [{"tracker_id": raw_roster_id, "box": BOX_B, "bgr": reactivated_bgr}]
    frame, dets = build_call(rows)
    linker.update(dets, frame)

    assert score_calls_for_marker(matcher, marker), (
        "once an identity is declared Missing, even a recycled raw tracker id must be revalidated"
    )


# --------------------------------------------------------------------------
# B4 — Birth & evidence window
# --------------------------------------------------------------------------


def test_b4_4_no_matcher_calls_when_nothing_is_missing():
    fps = 4.0
    enrolment_window_s = 0.25
    window = round(enrolment_window_s * fps)
    matcher = FakeMatcher()
    linker = SessionLinker(
        matcher,
        fps=fps,
        unknown_id_offset=UNKNOWN_OFFSET,
        enrolment_window_s=enrolment_window_s,
        evidence_window_s=10.0,
        evidence_frames=1,
        absent_death_s=100.0,  # nothing ever dies in this test
        min_mask_area_px=100,
    )
    raw_roster_id = 821
    session_id = 1
    for _ in range(window):
        rows = [{"tracker_id": raw_roster_id, "box": BOX_A, "bgr": (90, 90, 90)}]
        frame, dets = build_call(rows)
        linker.update(dets, frame)
    assert linker.roster == frozenset({session_id})

    newcomer_id = 901
    rows = [
        {"tracker_id": raw_roster_id, "box": BOX_A, "bgr": (90, 90, 90)},
        {"tracker_id": newcomer_id, "box": BOX_B, "bgr": (95, 95, 95)},
    ]
    frame, dets = build_call(rows)
    out = linker.update(dets, frame)  # evidence_frames=1 -> window closes now; missing is empty

    assert not matcher.score_calls, "no Missing identities exist; must settle Unknown without any matcher call"
    assert output_tracker_id(out, rows, newcomer_id) == newcomer_id + UNKNOWN_OFFSET

    rows2 = [{"tracker_id": newcomer_id, "box": BOX_B, "bgr": (95, 95, 95)}]
    frame2, dets2 = build_call(rows2)
    out2 = linker.update(dets2, frame2)
    assert output_tracker_id(out2, rows2, newcomer_id) == newcomer_id + UNKNOWN_OFFSET
    assert not matcher.score_calls


# --------------------------------------------------------------------------
# B5 — Batched decision
# --------------------------------------------------------------------------


def test_b5_scores_against_full_roster_when_only_one_identity_is_missing():
    fps = 4.0
    absent_death_s = 0.5
    death_threshold = round(absent_death_s * fps)
    matcher = FakeMatcher()
    linker = SessionLinker(
        matcher,
        fps=fps,
        unknown_id_offset=UNKNOWN_OFFSET,
        enrolment_window_s=0.25,
        evidence_window_s=10.0,
        evidence_frames=1,
        absent_death_s=absent_death_s,
        min_mask_area_px=100,
    )
    raw_missing_id, raw_active_id = 1101, 1102
    missing_id, active_id = 1, 2  # renumbered in ascending raw-id order
    rows = [
        {"tracker_id": raw_missing_id, "box": BOX_A, "bgr": (11, 11, 11)},
        {"tracker_id": raw_active_id, "box": BOX_B, "bgr": (22, 22, 22)},
    ]
    frame, dets = build_call(rows)
    linker.update(dets, frame)
    assert linker.roster == frozenset({missing_id, active_id})

    for _ in range(death_threshold + 1):
        rows = [{"tracker_id": raw_active_id, "box": BOX_B, "bgr": (22, 22, 22)}]
        frame, dets = build_call(rows)
        linker.update(dets, frame)

    return_bgr = (1, 2, 200)
    marker = rgb_marker_for_bgr(return_bgr)
    matcher.program(marker, [({missing_id: 0.9, active_id: 0.1}, missing_id)])
    rows = [{"tracker_id": 1201, "box": BOX_C, "bgr": return_bgr}]
    frame, dets = build_call(rows)
    out = linker.update(dets, frame)

    assert output_tracker_id(out, rows, 1201) == missing_id
    assert matcher.score_calls[-1]["gallery_keys"] == frozenset({missing_id, active_id}), (
        "SCI must compare against the complete frozen roster even when only one identity is eligible to link"
    )


def test_b5_best_match_to_active_identity_stays_unknown():
    fps = 4.0
    absent_death_s = 0.5
    death_threshold = round(absent_death_s * fps)
    matcher = FakeMatcher()
    linker = SessionLinker(
        matcher,
        fps=fps,
        unknown_id_offset=UNKNOWN_OFFSET,
        enrolment_window_s=0.25,
        evidence_window_s=10.0,
        evidence_frames=1,
        absent_death_s=absent_death_s,
        min_mask_area_px=100,
    )
    raw_missing_id, raw_active_id = 1301, 1302
    missing_id, active_id = 1, 2
    rows = [
        {"tracker_id": raw_missing_id, "box": BOX_A, "bgr": (33, 33, 33)},
        {"tracker_id": raw_active_id, "box": BOX_B, "bgr": (44, 44, 44)},
    ]
    frame, dets = build_call(rows)
    linker.update(dets, frame)
    assert linker.roster == frozenset({missing_id, active_id})

    for _ in range(death_threshold + 1):
        rows = [{"tracker_id": raw_active_id, "box": BOX_B, "bgr": (44, 44, 44)}]
        frame, dets = build_call(rows)
        linker.update(dets, frame)

    foreign_bgr = (3, 4, 220)
    marker = rgb_marker_for_bgr(foreign_bgr)
    matcher.program(marker, [({missing_id: 0.4, active_id: 0.95}, active_id)])
    raw_id = 1401
    rows = [{"tracker_id": raw_id, "box": BOX_C, "bgr": foreign_bgr}]
    frame, dets = build_call(rows)
    out = linker.update(dets, frame)

    assert output_tracker_id(out, rows, raw_id) == raw_id + UNKNOWN_OFFSET, (
        "an active identity may provide comparison evidence but must never be an eligible link target"
    )


def test_b5_coasting_identity_handoff_waits_for_missing_then_revalidates():
    fps = 4.0
    absent_death_s = 0.5
    matcher = FakeMatcher()
    linker = SessionLinker(
        matcher,
        fps=fps,
        unknown_id_offset=UNKNOWN_OFFSET,
        enrolment_window_s=0.25,
        evidence_window_s=10.0,
        evidence_frames=1,
        absent_death_s=absent_death_s,
        min_mask_area_px=100,
    )
    raw_roster_id = 1501
    session_id = 1
    rows = [{"tracker_id": raw_roster_id, "box": BOX_A, "bgr": (33, 33, 33)}]
    frame, dets = build_call(rows)
    linker.update(dets, frame)
    assert linker.roster == frozenset({session_id})

    # The old raw ID has been absent for one frame but has not crossed the
    # strict death threshold yet.
    frame, dets = empty_call()
    linker.update(dets, frame)

    return_bgr = (9, 19, 209)
    marker = rgb_marker_for_bgr(return_bgr)
    matcher.program(
        marker,
        [
            ({session_id: 0.95}, session_id),
            ({session_id: 0.95}, session_id),
        ],
    )
    return_raw_id = 1601
    rows = [{"tracker_id": return_raw_id, "box": BOX_B, "bgr": return_bgr}]

    # At exactly the death threshold the accepted roster identity is still
    # Active but its old raw ID is coasting off-screen. Do not force-link and
    # do not permanently settle this likely tracker handoff as Unknown. B-N4:
    # a deferred track is not mapped to a roster identity, so it emits offset.
    frame, dets = build_call(rows)
    provisional = linker.update(dets, frame)
    assert output_tracker_id(provisional, rows, return_raw_id) == return_raw_id + UNKNOWN_OFFSET

    # One frame later the old raw ID becomes Missing. The deferred row is
    # scored again against the same complete roster and can now link safely.
    frame, dets = build_call(rows)
    linked = linker.update(dets, frame)
    assert output_tracker_id(linked, rows, return_raw_id) == session_id

    matching_calls = score_calls_for_marker(matcher, marker)
    assert len(matching_calls) == 2
    assert all(call["gallery_keys"] == frozenset({session_id}) for call in matching_calls)


def test_b5_4_simultaneous_conflict_links_best_row_and_leaves_loser_unknown():
    fps = 4.0
    enrolment_window_s = 0.25
    window = round(enrolment_window_s * fps)
    absent_death_s = 0.5
    death_threshold = round(absent_death_s * fps)
    matcher = FakeMatcher()
    linker = SessionLinker(
        matcher,
        fps=fps,
        unknown_id_offset=UNKNOWN_OFFSET,
        enrolment_window_s=enrolment_window_s,
        evidence_window_s=10.0,
        evidence_frames=1,
        absent_death_s=absent_death_s,
        min_mask_area_px=100,
    )
    raw_a, raw_b = 1201, 1202
    id_a, id_b = 1, 2
    for _ in range(window):
        rows = [
            {"tracker_id": raw_a, "box": BOX_A, "bgr": (11, 11, 11)},
            {"tracker_id": raw_b, "box": BOX_B, "bgr": (22, 22, 22)},
        ]
        frame, dets = build_call(rows)
        linker.update(dets, frame)
    assert linker.roster == frozenset({id_a, id_b})

    for _ in range(death_threshold + 1):
        frame, dets = empty_call()
        linker.update(dets, frame)

    bgr_loser = (1, 2, 200)
    bgr_winner = (3, 4, 220)
    marker_loser = rgb_marker_for_bgr(bgr_loser)
    marker_winner = rgb_marker_for_bgr(bgr_winner)

    # Both rows accept id_a. The higher scorer receives that identity; the
    # other row must remain Unknown rather than being encouraged toward id_b
    # by removing id_a from SRC's comparison dictionary.
    scores_loser = {id_a: 0.6, id_b: 0.2}
    scores_winner = {id_a: 0.95, id_b: 0.1}
    matcher.program(marker_loser, [(scores_loser, id_a)])
    matcher.program(marker_winner, [(scores_winner, id_a)])

    raw_loser, raw_winner = 1301, 1302
    rows = [
        {"tracker_id": raw_loser, "box": BOX_C, "bgr": bgr_loser},
        {"tracker_id": raw_winner, "box": BOX_D, "bgr": bgr_winner},
    ]
    frame, dets = build_call(rows)
    out = linker.update(dets, frame)

    assert output_tracker_id(out, rows, raw_winner) == id_a, "the higher round-1 scorer wins the contested identity"
    assert output_tracker_id(out, rows, raw_loser) == raw_loser + UNKNOWN_OFFSET, (
        "a collision loser must stay Unknown instead of falling through to a less likely identity"
    )

    loser_calls = score_calls_for_marker(matcher, marker_loser)
    assert len(loser_calls) == 1
    assert loser_calls[0]["gallery_keys"] == frozenset({id_a, id_b})

    winner_calls = score_calls_for_marker(matcher, marker_winner)
    assert len(winner_calls) == 1


def test_b5_5_reject_settles_unknown_and_an_unchanged_crop_does_not_retry():
    # Replaces the old `test_b5_5_reject_stays_unknown_permanently`: settling is
    # no longer terminal (recovery is now event-driven — see the B-UR section).
    # What survives from the old contract is the *stable* half: a rejection
    # settles Unknown and an UNCHANGED settled crop must never re-invoke the
    # matcher. The recoverable half (a materially changed crop re-arms) is pinned
    # by test_ur1/test_ur5 below; static-no-retry over many frames by test_ur2.
    fps = 4.0
    enrolment_window_s = 0.25
    window = round(enrolment_window_s * fps)
    absent_death_s = 0.5
    death_threshold = round(absent_death_s * fps)
    matcher = FakeMatcher()
    linker = SessionLinker(
        matcher,
        fps=fps,
        unknown_id_offset=UNKNOWN_OFFSET,
        enrolment_window_s=enrolment_window_s,
        evidence_window_s=10.0,
        evidence_frames=1,
        absent_death_s=absent_death_s,
        min_mask_area_px=100,
        # cooldown 0 so the no-retry-on-unchanged assertion below actually
        # exercises the change gate rather than being shielded by a cooldown.
        unknown_recheck_cooldown_s=0.0,
    )
    raw_roster_id = 1001
    session_id = 1
    for _ in range(window):
        rows = [{"tracker_id": raw_roster_id, "box": BOX_A, "bgr": (100, 100, 100)}]
        frame, dets = build_call(rows)
        linker.update(dets, frame)
    assert linker.roster == frozenset({session_id})
    for _ in range(death_threshold + 1):
        frame, dets = empty_call()
        linker.update(dets, frame)

    reject_bgr = (44, 55, 66)
    marker = rgb_marker_for_bgr(reject_bgr)
    matcher.program(marker, [({session_id: 0.5}, REJECT)])
    unknown_raw_id = 1101
    rows = [{"tracker_id": unknown_raw_id, "box": BOX_B, "bgr": reject_bgr}]
    frame, dets = build_call(rows)
    out = linker.update(dets, frame)
    assert output_tracker_id(out, rows, unknown_raw_id) == unknown_raw_id + UNKNOWN_OFFSET

    calls_so_far = len(score_calls_for_marker(matcher, marker))
    assert calls_so_far == 1, "a rejected row is scored once against the stable comparison dictionary"

    # Reappearance with the SAME crop and the SAME candidate set: no material
    # change and no target-set expansion, so the settled Unknown is not re-armed
    # and the matcher is never called again.
    rows2 = [{"tracker_id": unknown_raw_id, "box": BOX_B, "bgr": reject_bgr}]
    frame2, dets2 = build_call(rows2)
    out2 = linker.update(dets2, frame2)
    assert output_tracker_id(out2, rows2, unknown_raw_id) == unknown_raw_id + UNKNOWN_OFFSET
    assert len(score_calls_for_marker(matcher, marker)) == calls_so_far, (
        "an unchanged settled-Unknown raw id must not trigger another matcher call"
    )


def test_b5_6_no_forced_link_when_matcher_rejects_despite_high_score():
    fps = 4.0
    enrolment_window_s = 0.25
    window = round(enrolment_window_s * fps)
    absent_death_s = 0.5
    death_threshold = round(absent_death_s * fps)
    matcher = FakeMatcher()
    linker = SessionLinker(
        matcher,
        fps=fps,
        unknown_id_offset=UNKNOWN_OFFSET,
        enrolment_window_s=enrolment_window_s,
        evidence_window_s=10.0,
        evidence_frames=1,
        absent_death_s=absent_death_s,
        min_mask_area_px=100,
    )
    raw_roster_id = 1401
    session_id = 1
    for _ in range(window):
        rows = [{"tracker_id": raw_roster_id, "box": BOX_A, "bgr": (150, 150, 150)}]
        frame, dets = build_call(rows)
        linker.update(dets, frame)
    assert linker.roster == frozenset({session_id})
    for _ in range(death_threshold + 1):
        frame, dets = empty_call()
        linker.update(dets, frame)

    bgr = (77, 88, 99)
    marker = rgb_marker_for_bgr(bgr)
    # A very high score must not matter: accept() explicitly rejects both rounds.
    matcher.program(marker, [({session_id: 0.99}, REJECT), ({session_id: 0.98}, REJECT)])
    raw_id = 1501
    rows = [{"tracker_id": raw_id, "box": BOX_B, "bgr": bgr}]
    frame, dets = build_call(rows)
    out = linker.update(dets, frame)

    assert output_tracker_id(out, rows, raw_id) == raw_id + UNKNOWN_OFFSET, (
        "a high score alone must never force a link — only accept() plus assignment victory can create one"
    )


# --------------------------------------------------------------------------
# B6 — Crops, masks, RGB, quality
# --------------------------------------------------------------------------


def test_b6_2_crops_handed_to_matcher_are_rgb_not_bgr():
    fps = 4.0
    enrolment_window_s = 0.5
    window = round(enrolment_window_s * fps)
    matcher = FakeMatcher()
    linker = SessionLinker(
        matcher,
        fps=fps,
        unknown_id_offset=UNKNOWN_OFFSET,
        enrolment_window_s=enrolment_window_s,
        evidence_window_s=10.0,
        evidence_frames=1,
        absent_death_s=10.0,
        min_mask_area_px=100,
    )
    raw_roster_id = 2001
    painted_bgr = (10, 20, 30)  # B=10, G=20, R=30
    for _ in range(window):
        rows = [{"tracker_id": raw_roster_id, "box": BOX_A, "bgr": painted_bgr}]
        frame, dets = build_call(rows)
        linker.update(dets, frame)
    assert linker.roster == frozenset({1})

    assert matcher.build_gallery_calls, "expected a Start-gallery build_gallery() call for the frozen roster identity"
    crops = matcher.build_gallery_calls[0]["crops"]
    assert crops, "expected at least one buffered quality crop for the Start gallery"
    pixel = tuple(int(v) for v in crops[0][0, 0])
    assert pixel == (30, 20, 10), (
        f"expected the crop's first pixel RGB-converted from painted BGR {painted_bgr} to be (30, 20, 10), "
        f"got {pixel}"
    )


# --------------------------------------------------------------------------
# B8 — reset()
# --------------------------------------------------------------------------


def test_b8_reset_drops_all_state():
    fps = 4.0
    enrolment_window_s = 0.25
    window = round(enrolment_window_s * fps)
    absent_death_s = 0.5
    death_threshold = round(absent_death_s * fps)
    matcher = FakeMatcher()
    linker = SessionLinker(
        matcher,
        fps=fps,
        unknown_id_offset=UNKNOWN_OFFSET,
        enrolment_window_s=enrolment_window_s,
        evidence_window_s=10.0,
        evidence_frames=1,
        absent_death_s=absent_death_s,
        min_mask_area_px=100,
    )
    raw_roster_id = 2201
    session_id = 1
    for _ in range(window):
        rows = [{"tracker_id": raw_roster_id, "box": BOX_A, "bgr": (33, 44, 55)}]
        frame, dets = build_call(rows)
        linker.update(dets, frame)
    for _ in range(death_threshold + 1):
        frame, dets = empty_call()
        linker.update(dets, frame)

    return_bgr = (66, 77, 88)
    marker = rgb_marker_for_bgr(return_bgr)
    matcher.program(marker, [({session_id: 0.9}, session_id)])
    return_raw_id = 2301
    rows = [{"tracker_id": return_raw_id, "box": BOX_B, "bgr": return_bgr}]
    frame, dets = build_call(rows)
    out = linker.update(dets, frame)
    assert output_tracker_id(out, rows, return_raw_id) == session_id  # sanity: link happened pre-reset
    assert linker.roster == frozenset({session_id})

    linker.reset()
    assert linker.roster == frozenset()

    # A fresh non-empty call after reset() opens a brand-new enrolment window
    # from scratch, and (window == 1) that call is immediately the freeze.
    #
    # The old raw_id -> session_id map is gone. Enrolling the SAME raw id that a
    # moment ago resolved to session 1 — this time alongside a LOWER raw id —
    # renumbers it to session 2 (B-N2, ascending raw-id order). Pinning it to a
    # DIFFERENT number than it wore pre-reset is what makes this discriminating:
    # a surviving map would resurrect session 1 for it.
    lower_raw_id = 1101  # sorts below return_raw_id -> takes session 1
    rows2 = [
        {"tracker_id": lower_raw_id, "box": BOX_A, "bgr": (11, 22, 33)},
        {"tracker_id": return_raw_id, "box": BOX_B, "bgr": return_bgr},
    ]
    frame2, dets2 = build_call(rows2)
    out2 = linker.update(dets2, frame2)
    assert linker.roster == frozenset({1, 2})
    assert output_tracker_id(out2, rows2, lower_raw_id) == 1
    assert output_tracker_id(out2, rows2, return_raw_id) == 2, (
        "reset() dropped the pre-reset raw -> session map: this raw id resolved to session 1 before the "
        "reset and must be renumbered from scratch on the new roster, not resurrected"
    )


# --------------------------------------------------------------------------
# Boundary and reset behavior
# --------------------------------------------------------------------------


def test_b3_3_no_death_at_exactly_the_absence_threshold():
    fps = 4.0
    enrolment_window_s = 0.25
    window = round(enrolment_window_s * fps)
    absent_death_s = 0.5
    death_threshold = round(absent_death_s * fps)
    matcher = FakeMatcher()
    linker = SessionLinker(
        matcher,
        fps=fps,
        unknown_id_offset=UNKNOWN_OFFSET,
        enrolment_window_s=enrolment_window_s,
        evidence_window_s=10.0,
        evidence_frames=1,
        absent_death_s=absent_death_s,
        min_mask_area_px=100,
    )
    raw_survivor_id, raw_doomed_id = 601, 602
    survivor_id, doomed_id = 1, 2  # renumbered in ascending raw-id order
    # Distinct, non-gray paints so each identity's crops are identifiable by
    # marker without depending on build_gallery()'s meta tag.
    survivor_bgr, doomed_bgr = (70, 71, 72), (80, 81, 82)
    survivor_marker = rgb_marker_for_bgr(survivor_bgr)
    for _ in range(window):
        rows = [
            {"tracker_id": raw_survivor_id, "box": BOX_A, "bgr": survivor_bgr},
            {"tracker_id": raw_doomed_id, "box": BOX_B, "bgr": doomed_bgr},
        ]
        frame, dets = build_call(rows)
        linker.update(dets, frame)
    assert linker.roster == frozenset({survivor_id, doomed_id})

    survivor_gallery_calls_before = len(build_gallery_calls_for_marker(matcher, survivor_marker))
    assert survivor_gallery_calls_before == 1, "sanity: only the Start gallery so far"

    # `death_threshold` consecutive absences for BOTH ids — exactly AT the
    # threshold, never exceeding it.
    for _ in range(death_threshold):
        frame, dets = empty_call()
        linker.update(dets, frame)

    # survivor reappears on this call, having been absent for exactly
    # `death_threshold` frames (not more) — coasting, not a death+return.
    # doomed stays absent on this same call, crossing "more than
    # death_threshold" consecutive absences, so it genuinely dies here.
    rows = [{"tracker_id": raw_survivor_id, "box": BOX_A, "bgr": survivor_bgr}]
    frame, dets = build_call(rows)
    out = linker.update(dets, frame)
    assert output_tracker_id(out, rows, raw_survivor_id) == survivor_id, (
        "reappearance at exactly the absence threshold must be treated as coasting, not a death"
    )

    survivor_gallery_calls_after = len(build_gallery_calls_for_marker(matcher, survivor_marker))
    assert survivor_gallery_calls_after == survivor_gallery_calls_before, (
        "no death embed must ever fire for an identity that only coasted through the threshold"
    )

    # doomed is now genuinely Missing; a fresh pending track's evidence
    # window closes against the complete roster comparison dictionary.
    query_bgr = (5, 15, 25)
    marker = rgb_marker_for_bgr(query_bgr)
    matcher.program(marker, [({doomed_id: 0.9, survivor_id: 0.1}, doomed_id)])
    rows = [{"tracker_id": 701, "box": BOX_C, "bgr": query_bgr}]
    frame, dets = build_call(rows)
    linker.update(dets, frame)

    matching_calls = score_calls_for_marker(matcher, marker)
    assert matching_calls, "expected the new pending track's evidence window to close and trigger scoring"
    assert matching_calls[0]["gallery_keys"] == frozenset({survivor_id, doomed_id})


def test_b6_3_quality_gate_blocks_edge_touching_and_tiny_mask_crops():
    fps = 4.0
    enrolment_window_s = 0.25
    window = round(enrolment_window_s * fps)
    evidence_window_s = 0.75
    min_mask_area_px = 150
    matcher = FakeMatcher()
    linker = SessionLinker(
        matcher,
        fps=fps,
        unknown_id_offset=UNKNOWN_OFFSET,
        enrolment_window_s=enrolment_window_s,
        evidence_window_s=evidence_window_s,
        evidence_frames=3,  # never reached: neither row below ever buffers a quality crop
        absent_death_s=10.0,
        min_mask_area_px=min_mask_area_px,
    )
    raw_roster_id = 5001
    for _ in range(window):
        rows = [{"tracker_id": raw_roster_id, "box": BOX_A, "bgr": (200, 200, 200)}]
        frame, dets = build_call(rows)
        linker.update(dets, frame)
    assert linker.roster == frozenset({1})

    evidence_window = round(evidence_window_s * fps)
    edge_id, tiny_id = 5101, 5102
    edge_box = (0, 4, 20, 24)  # touches the left frame edge -> fails the edge check
    tiny_box = (30, 30, 34, 34)  # 4x4 = 16 px^2, well below min_mask_area_px -> fails the area check
    edge_bgr = (1, 111, 201)
    tiny_bgr = (2, 112, 202)
    edge_marker = rgb_marker_for_bgr(edge_bgr)
    tiny_marker = rgb_marker_for_bgr(tiny_bgr)

    for _ in range(evidence_window):
        rows = [
            {"tracker_id": raw_roster_id, "box": BOX_A, "bgr": (200, 200, 200)},
            {"tracker_id": edge_id, "box": edge_box, "bgr": edge_bgr},
            {"tracker_id": tiny_id, "box": tiny_box, "bgr": tiny_bgr},
        ]
        frame, dets = build_call(rows)
        out = linker.update(dets, frame)
        assert output_tracker_id(out, rows, edge_id) == edge_id + UNKNOWN_OFFSET
        assert output_tracker_id(out, rows, tiny_id) == tiny_id + UNKNOWN_OFFSET

    all_build_gallery_crop_markers = set()
    for call in matcher.build_gallery_calls:
        all_build_gallery_crop_markers |= crop_markers(call["crops"])
    all_score_crop_markers = set()
    for call in matcher.score_calls:
        all_score_crop_markers |= crop_markers(call["crops"])

    assert edge_marker not in all_build_gallery_crop_markers, "an edge-touching crop must never reach build_gallery()"
    assert edge_marker not in all_score_crop_markers, "an edge-touching crop must never reach score()"
    assert tiny_marker not in all_build_gallery_crop_markers, "a below-threshold-area crop must never reach build_gallery()"
    assert tiny_marker not in all_score_crop_markers, "a below-threshold-area crop must never reach score()"


def test_b3_2_last_seen_views_are_replaced_not_appended_across_deaths():
    fps = 4.0
    enrolment_window_s = 0.25
    window = round(enrolment_window_s * fps)
    absent_death_s = 0.5
    death_threshold = round(absent_death_s * fps)
    evidence_frames = 2  # first quality appearance is Pending, second decides (mirrors test_b1_4)
    matcher = FakeMatcher()
    linker = SessionLinker(
        matcher,
        fps=fps,
        unknown_id_offset=UNKNOWN_OFFSET,
        enrolment_window_s=enrolment_window_s,
        evidence_window_s=10.0,
        evidence_frames=evidence_frames,
        absent_death_s=absent_death_s,
        min_mask_area_px=100,
    )
    raw_roster_id = 3001
    session_id = 1
    for _ in range(window):
        rows = [{"tracker_id": raw_roster_id, "box": BOX_A, "bgr": (90, 30, 30)}]
        frame, dets = build_call(rows)
        linker.update(dets, frame)
    assert linker.roster == frozenset({session_id})

    # One post-freeze presence frame so the identity's rolling last-seen
    # buffer holds "crops A": enrolment crops feed only the Start gallery,
    # and a death with an empty buffer legally skips its embed.
    rows = [{"tracker_id": raw_roster_id, "box": BOX_A, "bgr": (90, 30, 30)}]
    frame, dets = build_call(rows)
    linker.update(dets, frame)

    # Death 1: buffers "crops A" as last-seen views.
    for _ in range(death_threshold + 1):
        frame, dets = empty_call()
        linker.update(dets, frame)

    # Return and link: evidence_frames=2, so the first quality appearance is
    # Pending (no decision), the second closes the window and decides.
    return_bgr = (10, 20, 30)
    return_marker = rgb_marker_for_bgr(return_bgr)
    matcher.program(return_marker, [({session_id: 0.9}, session_id)])
    return_raw_id = 3101

    rows = [{"tracker_id": return_raw_id, "box": BOX_B, "bgr": return_bgr}]
    frame, dets = build_call(rows)
    out = linker.update(dets, frame)
    assert output_tracker_id(out, rows, return_raw_id) == return_raw_id + UNKNOWN_OFFSET, (
        "still Pending on the first appearance"
    )
    assert not matcher.score_calls, "must not decide before the window closes"

    rows = [{"tracker_id": return_raw_id, "box": BOX_B, "bgr": return_bgr}]
    frame, dets = build_call(rows)
    out = linker.update(dets, frame)
    assert output_tracker_id(out, rows, return_raw_id) == session_id, "decides and links on the second appearance"

    # A few more presence frames to firmly (re-)populate the now-Active
    # identity's rolling last-seen buffer with "crops B" before it dies again.
    for _ in range(evidence_frames):
        rows = [{"tracker_id": return_raw_id, "box": BOX_B, "bgr": return_bgr}]
        frame, dets = build_call(rows)
        linker.update(dets, frame)

    # Death 2: buffers "crops B" as the new last-seen views, which must
    # REPLACE (not append to) death 1's. Padded well past the bare "more
    # than threshold" minimum so this is unambiguously a genuine death, not
    # a boundary case (test_b3_3 covers the exact boundary separately).
    for _ in range(death_threshold + 3):
        frame, dets = empty_call()
        linker.update(dets, frame)

    # A fresh pending track triggers a scoring call that offers the identity's
    # aggregate Missing gallery (Start + only the most recent death's views).
    # It needs the same evidence_frames=2 quality appearances to decide.
    probe_bgr = (44, 44, 44)
    probe_marker = rgb_marker_for_bgr(probe_bgr)
    matcher.program(probe_marker, [({session_id: 0.5}, REJECT), ({session_id: 0.4}, REJECT)])
    probe_raw_id = 3201

    rows = [{"tracker_id": probe_raw_id, "box": BOX_C, "bgr": probe_bgr}]
    frame, dets = build_call(rows)
    linker.update(dets, frame)
    assert not score_calls_for_marker(matcher, probe_marker), "must not decide before the probe's window closes"

    rows = [{"tracker_id": probe_raw_id, "box": BOX_C, "bgr": probe_bgr}]
    frame, dets = build_call(rows)
    linker.update(dets, frame)

    probe_calls = score_calls_for_marker(matcher, probe_marker)
    assert probe_calls, "expected the probe track's evidence window to close and trigger scoring"
    offered_size_after_second_death = probe_calls[0]["gallery_view_counts"][session_id]

    assert len(matcher.build_gallery_calls) == 3, (
        "expected exactly 3 build_gallery calls in this scenario: Start gallery, death-1 embed, death-2 embed"
    )
    start_call, death1_call, death2_call = matcher.build_gallery_calls
    start_view_count = len(start_call["crops"]) or 1  # FakeGallery.views uses max(len(crops), 1)
    death1_view_count = len(death1_call["crops"]) or 1
    death2_view_count = len(death2_call["crops"]) or 1

    replace_size = start_view_count + death2_view_count
    unbounded_append_size = start_view_count + death1_view_count + death2_view_count

    assert offered_size_after_second_death == replace_size, (
        f"expected the post-death-2 gallery offered to score() to hold Start + only death 2's views "
        f"({replace_size}), got {offered_size_after_second_death}"
    )
    assert offered_size_after_second_death < unbounded_append_size, (
        "the offered gallery must not grow unboundedly by accumulating every past death's last-seen views"
    )


def test_b4_2_evidence_window_times_out_without_enough_quality_crops():
    fps = 4.0
    enrolment_window_s = 0.25
    window = round(enrolment_window_s * fps)
    absent_death_s = 0.5
    death_threshold = round(absent_death_s * fps)
    evidence_window_s = 1.0
    matcher = FakeMatcher()
    linker = SessionLinker(
        matcher,
        fps=fps,
        unknown_id_offset=UNKNOWN_OFFSET,
        enrolment_window_s=enrolment_window_s,
        evidence_window_s=evidence_window_s,
        evidence_frames=3,  # never reached: only 1 quality crop is ever buffered
        absent_death_s=absent_death_s,
        min_mask_area_px=100,
    )
    raw_roster_id = 4001
    session_id = 1
    for _ in range(window):
        rows = [{"tracker_id": raw_roster_id, "box": BOX_A, "bgr": (60, 60, 60)}]
        frame, dets = build_call(rows)
        linker.update(dets, frame)
    assert linker.roster == frozenset({session_id})

    # Kill the identity so a genuinely Missing one exists for the pending
    # track below to be scored against once its window times out.
    for _ in range(death_threshold + 1):
        frame, dets = empty_call()
        linker.update(dets, frame)

    evidence_window = round(evidence_window_s * fps)
    assert evidence_window >= 2, "test needs room for one quality frame followed by low-quality filler frames"

    pending_bgr = (17, 27, 37)
    marker = rgb_marker_for_bgr(pending_bgr)
    matcher.program(marker, [({session_id: 0.9}, session_id)])
    pending_raw_id = 4101
    tiny_box = (30, 30, 34, 34)  # 4x4 = 16 px^2, fails min_mask_area_px=100

    # First appearance: a single quality crop -> starts the Pending clock
    # and buffers exactly 1 quality crop.
    rows = [{"tracker_id": pending_raw_id, "box": BOX_B, "bgr": pending_bgr}]
    frame, dets = build_call(rows)
    out = linker.update(dets, frame)
    assert output_tracker_id(out, rows, pending_raw_id) == pending_raw_id + UNKNOWN_OFFSET
    assert not matcher.score_calls, "must not decide before the window closes"

    # Remaining calls up to (but not including) the window-closing call:
    # low-quality appearances only, so the buffer never reaches evidence_frames.
    for _ in range(evidence_window - 2):
        rows = [{"tracker_id": pending_raw_id, "box": tiny_box, "bgr": pending_bgr}]
        frame, dets = build_call(rows)
        out = linker.update(dets, frame)
        assert output_tracker_id(out, rows, pending_raw_id) == pending_raw_id + UNKNOWN_OFFSET
        assert not matcher.score_calls, "must not decide before the window closes"

    # The evidence_window-th call since first appearance: still only 1
    # quality crop ever buffered (well under evidence_frames=3), but the
    # elapsed-time criterion fires the decision anyway.
    rows = [{"tracker_id": pending_raw_id, "box": tiny_box, "bgr": pending_bgr}]
    frame, dets = build_call(rows)
    out = linker.update(dets, frame)

    matching_calls = score_calls_for_marker(matcher, marker)
    assert matching_calls, "expected the evidence window to time out on elapsed frames and trigger a decision"
    assert output_tracker_id(out, rows, pending_raw_id) == session_id, (
        "the scripted matcher accepted the link; the decision must land on this call"
    )


# ==========================================================================
# T08 — SPEC §B-B: binding at the enrolment freeze
# ==========================================================================


def bind_linker(matcher, galleries, *, bind_tau, bind_margin, fps=4.0, **kwargs):
    """A linker whose enrolment window closes on its very first call."""
    options = dict(
        enrolment_window_s=0.25,
        evidence_window_s=10.0,
        evidence_frames=1,
        absent_death_s=0.5,
        min_mask_area_px=100,
    )
    options.update(kwargs)
    return SessionLinker(
        matcher,
        fps=fps,
        persistent_galleries=galleries,
        bind_tau=bind_tau,
        bind_margin=bind_margin,
        unknown_id_offset=UNKNOWN_OFFSET,
        **options,
    )


def test_b_b1_no_binding_is_attempted_without_persistent_galleries():
    matcher = FakeMatcher()
    linker = bind_linker(matcher, {}, bind_tau=0.5, bind_margin=0.1)

    rows = [
        {"tracker_id": 301, "box": BOX_A, "bgr": (10, 20, 30)},
        {"tracker_id": 302, "box": BOX_B, "bgr": (40, 50, 60)},
    ]
    frame, dets = build_call(rows)
    linker.update(dets, frame)

    assert not matcher.score_calls, "with no persistent galleries the freeze must not score anything"
    assert linker.roster == frozenset({1, 2}), "everyone is session-only, numbered from 1 (base = 0)"


def test_b_b1_identity_without_usable_start_crops_never_binds():
    matcher = FakeMatcher()
    good_bgr, tiny_bgr = (10, 20, 30), (40, 50, 60)
    good_marker, tiny_marker = rgb_marker_for_bgr(good_bgr), rgb_marker_for_bgr(tiny_bgr)
    matcher.program(good_marker, [({1: 0.9}, None)])
    # tiny_marker is deliberately NOT programmed: FakeMatcher.score() asserts
    # if it is ever called, so an attempt to bind the crop-less identity is
    # caught rather than silently tolerated.
    linker = bind_linker(
        matcher, {1: persistent_gallery(1)}, bind_tau=0.5, bind_margin=0.1, min_mask_area_px=150
    )

    raw_good, raw_tiny = 401, 402
    tiny_box = (30, 30, 34, 34)  # 16 px^2 mask -> never a quality crop
    rows = [
        {"tracker_id": raw_good, "box": BOX_A, "bgr": good_bgr},
        {"tracker_id": raw_tiny, "box": tiny_box, "bgr": tiny_bgr},
    ]
    frame, dets = build_call(rows)
    linker.update(dets, frame)

    assert not score_calls_for_marker(matcher, tiny_marker), (
        "an identity with no usable Start crops (start_views is None) must never be scored for binding"
    )
    # raw_good binds to specimen 1; raw_tiny is session-only at base(=1) + 1.
    assert linker.roster == frozenset({1, 2})


def test_b_b2_binding_scores_start_crops_against_the_complete_gallery_dict():
    matcher = FakeMatcher()
    galleries = {1: persistent_gallery(1), 2: persistent_gallery(2), 5: persistent_gallery(5)}
    bgr_a, bgr_b = (10, 20, 30), (40, 50, 60)
    marker_a, marker_b = rgb_marker_for_bgr(bgr_a), rgb_marker_for_bgr(bgr_b)
    matcher.program(marker_a, [({1: 0.9, 2: 0.1, 5: 0.05}, None)])
    matcher.program(marker_b, [({1: 0.1, 2: 0.9, 5: 0.05}, None)])
    linker = bind_linker(matcher, galleries, bind_tau=0.5, bind_margin=0.1)

    raw_a, raw_b = 301, 302
    rows = [
        # Row order deliberately puts the HIGHER raw id first (cf.
        # test_b_b4_ties_are_broken_by_ascending_raw_id). Fed in ascending raw-id
        # order the bind-order assertion below would pass for an implementation
        # that merely iterates rows, which is not what B-B2 asks for.
        {"tracker_id": raw_b, "box": BOX_B, "bgr": bgr_b},
        {"tracker_id": raw_a, "box": BOX_A, "bgr": bgr_a},
    ]
    frame, dets = build_call(rows)
    linker.update(dets, frame)

    bind_calls = [c for c in matcher.score_calls if c["marker"] in (marker_a, marker_b)]
    assert len(bind_calls) == 2, "every enrolled identity with Start views is scored once for binding"
    for call in bind_calls:
        assert call["gallery_keys"] == frozenset({1, 2, 5}), (
            "binding must score against the COMPLETE loaded gallery dict — never a shrunken "
            "dictionary (Trap 2 applies to binding too)"
        )
    assert [c["marker"] for c in bind_calls] == [marker_a, marker_b], (
        "identities are bound in ascending raw-id order"
    )


def test_b_b3_bind_requires_the_best_score_to_reach_bind_tau():
    matcher = FakeMatcher()
    galleries = {1: persistent_gallery(1), 2: persistent_gallery(2)}
    bgr = (10, 20, 30)
    marker = rgb_marker_for_bgr(bgr)
    # Best 0.4 is under tau=0.5 even though its margin over second is huge.
    matcher.program(marker, [({1: 0.4, 2: 0.01}, None)])
    linker = bind_linker(matcher, galleries, bind_tau=0.5, bind_margin=0.05)

    rows = [{"tracker_id": 501, "box": BOX_A, "bgr": bgr}]
    frame, dets = build_call(rows)
    linker.update(dets, frame)

    assert linker.roster == frozenset({3}), (
        "below tau -> session-only at base(=2) + 1; a specimen number must never be worn "
        "by an identity that did not confidently bind"
    )


def test_b_b3_bind_requires_the_margin_over_the_runner_up():
    matcher = FakeMatcher()
    galleries = {1: persistent_gallery(1), 2: persistent_gallery(2)}
    bgr = (10, 20, 30)
    marker = rgb_marker_for_bgr(bgr)
    # Best 0.9 clears tau easily, but 0.9 - 0.8 = 0.1 is under margin=0.2.
    matcher.program(marker, [({1: 0.9, 2: 0.8}, None)])
    linker = bind_linker(matcher, galleries, bind_tau=0.5, bind_margin=0.2)

    rows = [{"tracker_id": 501, "box": BOX_A, "bgr": bgr}]
    frame, dets = build_call(rows)
    linker.update(dets, frame)

    assert linker.roster == frozenset({3}), "an ambiguous best-vs-second falls back to session-only"


def test_b_b3_with_a_single_gallery_the_margin_is_vacuous():
    matcher = FakeMatcher()
    bgr = (10, 20, 30)
    marker = rgb_marker_for_bgr(bgr)
    matcher.program(marker, [({4: 0.6}, None)])
    # An impossible margin: with one gallery there is no second score, so the
    # margin cannot be evaluated and must not block the bind.
    linker = bind_linker(matcher, {4: persistent_gallery(4)}, bind_tau=0.5, bind_margin=0.9)

    rows = [{"tracker_id": 501, "box": BOX_A, "bgr": bgr}]
    frame, dets = build_call(rows)
    linker.update(dets, frame)

    assert linker.roster == frozenset({4}), (
        "len(galleries) == 1 -> the margin is vacuous and tau alone decides"
    )


def test_b_b4_greedy_one_to_one_sends_the_loser_to_session_only_not_its_second_choice():
    matcher = FakeMatcher()
    galleries = {1: persistent_gallery(1), 2: persistent_gallery(2)}
    bgr_a, bgr_b = (10, 20, 30), (40, 50, 60)
    marker_a, marker_b = rgb_marker_for_bgr(bgr_a), rgb_marker_for_bgr(bgr_b)
    # Both propose specimen 1 (each clears tau and margin). A scores higher.
    matcher.program(marker_a, [({1: 0.9, 2: 0.05}, None)])
    matcher.program(marker_b, [({1: 0.8, 2: 0.04}, None)])
    linker = bind_linker(matcher, galleries, bind_tau=0.5, bind_margin=0.1)

    raw_a, raw_b = 501, 502
    rows = [
        {"tracker_id": raw_a, "box": BOX_A, "bgr": bgr_a},
        {"tracker_id": raw_b, "box": BOX_B, "bgr": bgr_b},
    ]
    frame, dets = build_call(rows)
    linker.update(dets, frame)

    # base = max({1, 2}) = 2, so the sole session-only identity becomes 3.
    assert linker.roster == frozenset({1, 3}), (
        "specimen 2 must stay unworn: a losing identity falls back to session-only, it is "
        "never pushed onto its second-choice specimen"
    )

    frame, dets = build_call(rows)
    out = linker.update(dets, frame)
    assert output_tracker_id(out, rows, raw_a) == 1, "the higher scorer takes the contested specimen"
    assert output_tracker_id(out, rows, raw_b) == 3, "the loser is session-only, not specimen 2"


def test_b_b4_ties_are_broken_by_ascending_raw_id():
    matcher = FakeMatcher()
    galleries = {1: persistent_gallery(1), 2: persistent_gallery(2)}
    bgr_a, bgr_b = (10, 20, 30), (40, 50, 60)
    marker_a, marker_b = rgb_marker_for_bgr(bgr_a), rgb_marker_for_bgr(bgr_b)
    # Identical best score on specimen 1 -> the tie-break decides.
    matcher.program(marker_a, [({1: 0.7, 2: 0.05}, None)])
    matcher.program(marker_b, [({1: 0.7, 2: 0.04}, None)])
    linker = bind_linker(matcher, galleries, bind_tau=0.5, bind_margin=0.1)

    raw_lower, raw_higher = 601, 602
    rows = [
        # Row order deliberately puts the higher raw id first, so a tie broken
        # by row position rather than by raw id is caught.
        {"tracker_id": raw_higher, "box": BOX_B, "bgr": bgr_b},
        {"tracker_id": raw_lower, "box": BOX_A, "bgr": bgr_a},
    ]
    frame, dets = build_call(rows)
    linker.update(dets, frame)

    frame, dets = build_call(rows)
    out = linker.update(dets, frame)
    assert output_tracker_id(out, rows, raw_lower) == 1, "an exact score tie is awarded to the lower raw id"
    assert output_tracker_id(out, rows, raw_higher) == 3, "the tie-break loser is session-only"


def test_b_b5_bound_identity_is_compared_as_persistent_union_start():
    matcher = FakeMatcher()
    persistent_views = 7
    galleries = {1: persistent_gallery(1, n_views=persistent_views)}
    bgr_a, bgr_b = (10, 20, 30), (40, 50, 60)
    marker_a, marker_b = rgb_marker_for_bgr(bgr_a), rgb_marker_for_bgr(bgr_b)
    matcher.program(marker_a, [({1: 0.9}, None)])  # binds
    matcher.program(marker_b, [({1: 0.2}, None)])  # under tau -> session-only
    linker = bind_linker(matcher, galleries, bind_tau=0.5, bind_margin=0.1)

    raw_bound, raw_unbound = 701, 702
    rows = [
        {"tracker_id": raw_bound, "box": BOX_A, "bgr": bgr_a},
        {"tracker_id": raw_unbound, "box": BOX_B, "bgr": bgr_b},
    ]
    frame, dets = build_call(rows)
    linker.update(dets, frame)
    bound_id, unbound_id = 1, 2  # base = max({1}) = 1 -> session-only starts at 2
    assert linker.roster == frozenset({bound_id, unbound_id})

    # Kill both. Their crop buffers are empty (no post-freeze presence), so no
    # death embed fires and last-seen views stay absent — leaving exactly
    # `persistent ∪ Start` vs `Start` to compare.
    death_threshold = round(0.5 * 4.0)
    for _ in range(death_threshold + 1):
        frame, dets = empty_call()
        linker.update(dets, frame)

    probe_bgr = (44, 55, 66)
    probe_marker = rgb_marker_for_bgr(probe_bgr)
    matcher.program(probe_marker, [({bound_id: 0.11, unbound_id: 0.12}, REJECT)])
    rows = [{"tracker_id": 801, "box": BOX_C, "bgr": probe_bgr}]
    frame, dets = build_call(rows)
    linker.update(dets, frame)

    probe_calls = score_calls_for_marker(matcher, probe_marker)
    assert probe_calls, "expected the probe's evidence window to close and trigger scoring"
    counts = probe_calls[0]["gallery_view_counts"]

    bound_start_views = len(build_gallery_calls_for_marker(matcher, marker_a)[0]["crops"])
    unbound_start_views = len(build_gallery_calls_for_marker(matcher, marker_b)[0]["crops"])

    assert counts[bound_id] == persistent_views + bound_start_views, (
        "a bound identity's comparison gallery is its persistent views concatenated with its Start views"
    )
    assert counts[unbound_id] == unbound_start_views, (
        "an unbound identity keeps session views only — it must not inherit any persistent views"
    )
    assert counts[bound_id] > counts[unbound_id], (
        "sanity: the mixed-bind atom asymmetry Trap 4 predicts is observable here"
    )


def test_b_b6_binding_is_never_revisited_after_the_freeze():
    matcher = FakeMatcher()
    galleries = {1: persistent_gallery(1), 2: persistent_gallery(2)}
    start_bgr = (10, 20, 30)
    start_marker = rgb_marker_for_bgr(start_bgr)
    matcher.program(start_marker, [({1: 0.9, 2: 0.05}, None)])
    linker = bind_linker(matcher, galleries, bind_tau=0.5, bind_margin=0.1)

    raw_id = 801
    rows = [{"tracker_id": raw_id, "box": BOX_A, "bgr": start_bgr}]
    frame, dets = build_call(rows)
    linker.update(dets, frame)
    assert linker.roster == frozenset({1})

    death_threshold = round(0.5 * 4.0)
    for _ in range(death_threshold + 1):
        frame, dets = empty_call()
        linker.update(dets, frame)

    # The identity returns under a new raw id and a different paint, so its
    # decision score() call is distinguishable from its binding one.
    return_bgr = (60, 70, 80)
    return_marker = rgb_marker_for_bgr(return_bgr)
    matcher.program(return_marker, [({1: 0.9}, 1)])
    return_raw_id = 802
    rows = [{"tracker_id": return_raw_id, "box": BOX_B, "bgr": return_bgr}]
    frame, dets = build_call(rows)
    out = linker.update(dets, frame)

    assert output_tracker_id(out, rows, return_raw_id) == 1, "the identity keeps the specimen it bound at the freeze"
    assert linker.roster == frozenset({1}), "the roster is frozen; a link never renumbers it"
    assert len(score_calls_for_marker(matcher, start_marker)) == 1, (
        "binding happens exactly once, at the freeze — a later link/death/return must never re-bind"
    )


# ==========================================================================
# T08 — SPEC §B-N: session-id assignment & the emitted id space
# ==========================================================================


def test_b_n1_a_bound_identity_takes_its_specimen_number_as_its_session_id():
    matcher = FakeMatcher()
    bgr = (10, 20, 30)
    marker = rgb_marker_for_bgr(bgr)
    matcher.program(marker, [({3: 0.9, 8: 0.05}, None)])
    linker = bind_linker(
        matcher,
        {3: persistent_gallery(3), 8: persistent_gallery(8)},
        bind_tau=0.5,
        bind_margin=0.1,
    )

    raw_id = 901
    rows = [{"tracker_id": raw_id, "box": BOX_A, "bgr": bgr}]
    frame, dets = build_call(rows)
    linker.update(dets, frame)

    assert linker.roster == frozenset({3}), "session_id == specimen_number, not a 1-based counter"

    frame, dets = build_call(rows)
    out = linker.update(dets, frame)
    assert output_tracker_id(out, rows, raw_id) == 3


def test_b_n2_session_only_ids_start_from_one_without_persistent_galleries():
    matcher = FakeMatcher()
    linker = bind_linker(matcher, None, bind_tau=0.5, bind_margin=0.1)

    # Row order is deliberately NOT ascending by raw id: the numbering must
    # follow raw-id order, not the order rows happen to arrive in.
    rows = [
        {"tracker_id": 4207, "box": BOX_A, "bgr": (10, 20, 30)},
        {"tracker_id": 91, "box": BOX_B, "bgr": (40, 50, 60)},
        {"tracker_id": 350, "box": BOX_C, "bgr": (70, 80, 90)},
    ]
    frame, dets = build_call(rows)
    linker.update(dets, frame)

    assert linker.roster == frozenset({1, 2, 3}), "no galleries loaded -> base = 0 -> contiguous {1..N}"

    frame, dets = build_call(rows)
    out = linker.update(dets, frame)
    assert output_tracker_id(out, rows, 91) == 1
    assert output_tracker_id(out, rows, 350) == 2
    assert output_tracker_id(out, rows, 4207) == 3


def test_b_n2_session_only_ids_are_reserved_above_the_highest_loaded_specimen():
    matcher = FakeMatcher()
    galleries = {1: persistent_gallery(1), 2: persistent_gallery(2), 5: persistent_gallery(5)}
    bgrs = [(10, 20, 30), (40, 50, 60), (70, 80, 90)]
    for bgr in bgrs:
        matcher.program(rgb_marker_for_bgr(bgr), [({1: 0.1, 2: 0.1, 5: 0.1}, None)])  # all under tau
    linker = bind_linker(matcher, galleries, bind_tau=0.5, bind_margin=0.1)

    raw_ids = [901, 902, 903]
    rows = [
        {"tracker_id": raw_id, "box": box, "bgr": bgr}
        for raw_id, box, bgr in zip(raw_ids, (BOX_A, BOX_B, BOX_C), bgrs)
    ]
    frame, dets = build_call(rows)
    linker.update(dets, frame)

    assert linker.roster == frozenset({6, 7, 8}), (
        "galleries loaded -> base = max({1, 2, 5}) = 5 -> session-only ids start at 6, so no "
        "specimen number is ever worn by an identity that did not bind"
    )

    frame, dets = build_call(rows)
    out = linker.update(dets, frame)
    assert [output_tracker_id(out, rows, raw_id) for raw_id in raw_ids] == [6, 7, 8], (
        "k = 1, 2, ... in ascending raw-id order"
    )


def test_b_n2_a_partial_bind_leaves_the_unclaimed_specimen_numbers_unworn():
    matcher = FakeMatcher()
    galleries = {1: persistent_gallery(1), 2: persistent_gallery(2), 5: persistent_gallery(5)}
    bound_bgr, lonely_bgr = (10, 20, 30), (40, 50, 60)
    matcher.program(rgb_marker_for_bgr(bound_bgr), [({1: 0.05, 2: 0.9, 5: 0.05}, None)])
    matcher.program(rgb_marker_for_bgr(lonely_bgr), [({1: 0.1, 2: 0.1, 5: 0.1}, None)])
    linker = bind_linker(matcher, galleries, bind_tau=0.5, bind_margin=0.1)

    raw_bound, raw_lonely = 901, 902
    rows = [
        {"tracker_id": raw_bound, "box": BOX_A, "bgr": bound_bgr},
        {"tracker_id": raw_lonely, "box": BOX_B, "bgr": lonely_bgr},
    ]
    frame, dets = build_call(rows)
    linker.update(dets, frame)

    assert linker.roster == frozenset({2, 6}), (
        "the mixed roster the SPEC predicts: specimen 2 is worn by its bound identity, "
        "specimens 1 and 5 stay unworn, and the session-only identity sits at base(=5) + 1"
    )

    frame, dets = build_call(rows)
    out = linker.update(dets, frame)
    assert output_tracker_id(out, rows, raw_bound) == 2
    assert output_tracker_id(out, rows, raw_lonely) == 6


def test_b_n4_roster_ids_are_emitted_unchanged_while_unknown_ids_are_offset():
    fps = 4.0
    matcher = FakeMatcher()
    linker = SessionLinker(
        matcher,
        fps=fps,
        unknown_id_offset=UNKNOWN_OFFSET,
        enrolment_window_s=0.25,
        evidence_window_s=100.0,
        evidence_frames=5,  # the newcomer stays Pending for the whole test
        absent_death_s=100.0,
        min_mask_area_px=100,
    )
    raw_roster_id, newcomer_id = 1001, 1002
    rows = [{"tracker_id": raw_roster_id, "box": BOX_A, "bgr": (10, 20, 30)}]
    frame, dets = build_call(rows)
    linker.update(dets, frame)
    assert linker.roster == frozenset({1})

    rows = [
        {"tracker_id": raw_roster_id, "box": BOX_A, "bgr": (10, 20, 30)},
        {"tracker_id": newcomer_id, "box": BOX_B, "bgr": (40, 50, 60)},
    ]
    frame, dets = build_call(rows)
    out = linker.update(dets, frame)

    assert output_tracker_id(out, rows, raw_roster_id) == 1, "roster session ids are emitted unchanged"
    emitted_unknown = output_tracker_id(out, rows, newcomer_id)
    assert emitted_unknown == newcomer_id + UNKNOWN_OFFSET
    assert emitted_unknown not in linker.roster, (
        "the roster and the unknown id range must stay disjoint forever — this is what makes the "
        "seam's 'Unknown = not in roster' test safe"
    )


def test_b_n4_the_offset_is_taken_from_the_constructor_not_a_default():
    """Two linkers, identical but for `unknown_id_offset`, must emit different
    unknown ids for the same track — proving the constructor value is honoured
    (SPEC "Tunable parameters": never assert DEFAULT_UNKNOWN_ID_OFFSET)."""
    emissions = {}
    for offset in (137, 24_000):
        matcher = FakeMatcher()
        linker = SessionLinker(
            matcher,
            fps=4.0,
            unknown_id_offset=offset,
            enrolment_window_s=0.25,
            evidence_window_s=100.0,
            evidence_frames=5,
            absent_death_s=100.0,
            min_mask_area_px=100,
        )
        rows = [{"tracker_id": 1001, "box": BOX_A, "bgr": (10, 20, 30)}]
        frame, dets = build_call(rows)
        linker.update(dets, frame)

        newcomer_id = 1002
        rows = [
            {"tracker_id": 1001, "box": BOX_A, "bgr": (10, 20, 30)},
            {"tracker_id": newcomer_id, "box": BOX_B, "bgr": (40, 50, 60)},
        ]
        frame, dets = build_call(rows)
        out = linker.update(dets, frame)
        emissions[offset] = output_tracker_id(out, rows, newcomer_id)

    assert emissions == {137: 1002 + 137, 24_000: 1002 + 24_000}


def test_b_n4_settled_unknown_tracks_keep_emitting_their_offset_id():
    fps = 4.0
    matcher = FakeMatcher()
    linker = SessionLinker(
        matcher,
        fps=fps,
        unknown_id_offset=UNKNOWN_OFFSET,
        enrolment_window_s=0.25,
        evidence_window_s=10.0,
        evidence_frames=1,  # the newcomer's window closes on its first appearance
        absent_death_s=100.0,  # nothing is Missing -> it settles Unknown outright
        min_mask_area_px=100,
    )
    rows = [{"tracker_id": 1001, "box": BOX_A, "bgr": (10, 20, 30)}]
    frame, dets = build_call(rows)
    linker.update(dets, frame)

    newcomer_id = 1002
    for _ in range(3):
        rows = [
            {"tracker_id": 1001, "box": BOX_A, "bgr": (10, 20, 30)},
            {"tracker_id": newcomer_id, "box": BOX_B, "bgr": (40, 50, 60)},
        ]
        frame, dets = build_call(rows)
        out = linker.update(dets, frame)
        assert output_tracker_id(out, rows, newcomer_id) == newcomer_id + UNKNOWN_OFFSET


def test_b_n5_the_freeze_frame_itself_emits_session_ids_consistent_with_the_roster():
    """B-N5 (clarified 2026-07-15): the flip happens on the very call in which
    the roster freezes, not on the following one.

    The raw ids here (2, 3) deliberately OVERLAP the session-id space the freeze
    publishes ({1, 2}) — that overlap is what gives the same-tick consistency
    assertion its teeth. The app samples `tracker.roster` and the emitted ids in
    the SAME tick and intersects them (T10 B-C1/B-S1), so a freeze frame that
    emitted raw ids would intersect {2, 3} against roster {1, 2} down to {2}:
    presence booked for session 2 while the track actually carrying raw 2 is
    session 1 — the wrong instrument — and session 2's real track dropped.
    """
    fps = 4.0
    enrolment_window_s = 0.75
    window = round(enrolment_window_s * fps)  # window == 3: calls 1-2 pre-freeze, call 3 IS the freeze
    matcher = FakeMatcher()
    linker = SessionLinker(
        matcher,
        fps=fps,
        unknown_id_offset=UNKNOWN_OFFSET,
        enrolment_window_s=enrolment_window_s,
        evidence_window_s=100.0,
        evidence_frames=5,  # the interloper stays Pending — never decided in this test
        absent_death_s=100.0,
        min_mask_area_px=100,
    )
    # Present for the whole window -> both enrol -> sessions 1, 2 (ascending raw-id order).
    raw_first, raw_second = 2, 3
    # Present on the freeze frame ONLY -> 1 of 3 is no majority -> never a roster identity.
    raw_interloper = 4

    enrolling_rows = [
        {"tracker_id": raw_first, "box": BOX_A, "bgr": (10, 20, 30)},
        {"tracker_id": raw_second, "box": BOX_B, "bgr": (40, 50, 60)},
    ]

    # Frames STRICTLY BEFORE the freeze still pass their raw ids through, as today.
    for i in range(window - 1):
        frame, dets = build_call(enrolling_rows)
        out = linker.update(dets, frame)
        for raw in (raw_first, raw_second):
            assert output_tracker_id(out, enrolling_rows, raw) == raw, (
                f"call {i + 1}/{window} is strictly before the freeze: ids still pass through raw"
            )
        assert linker.roster == frozenset(), "the roster is not published until the window closes"

    # ---- the freeze frame itself ----
    rows = enrolling_rows + [{"tracker_id": raw_interloper, "box": BOX_C, "bgr": (70, 80, 90)}]
    frame, dets = build_call(rows)
    out = linker.update(dets, frame)
    roster = linker.roster  # sampled in the SAME tick, exactly as the app does

    emitted = {raw: output_tracker_id(out, rows, raw) for raw in (raw_first, raw_second, raw_interloper)}

    assert emitted[raw_first] == 1, "the freeze frame itself emits session ids, not raw ids"
    assert emitted[raw_second] == 2
    assert emitted[raw_interloper] == raw_interloper + UNKNOWN_OFFSET, (
        "a track present on the freeze frame that is not a roster identity emits in the offset space (B-N4)"
    )

    # Same-tick consistency — the whole point of the strict ruling: every id the
    # freeze frame emitted must already sit on the correct side of the roster
    # intersection the app performs against the roster this same call published.
    assert roster == frozenset({1, 2})
    assert {emitted[raw_first], emitted[raw_second]} <= roster, (
        "the freeze frame's enrolled emissions must survive the app's roster intersection"
    )
    assert emitted[raw_interloper] not in roster, "roster and unknown id ranges stay disjoint on the freeze frame too"


def test_b_n6_a_session_id_reaching_the_unknown_offset_warns_but_never_raises(caplog):
    matcher = FakeMatcher()
    tiny_offset = 2  # session ids 1, 2, 3 -> two of them reach it
    linker = SessionLinker(
        matcher,
        fps=4.0,
        unknown_id_offset=tiny_offset,
        enrolment_window_s=0.25,
        evidence_window_s=10.0,
        evidence_frames=1,
        absent_death_s=100.0,
        min_mask_area_px=100,
    )
    # Raw ids deliberately carry no bare "2"/"3" token, so the message search
    # below cannot be satisfied by an echoed raw id.
    rows = [
        {"tracker_id": 101, "box": BOX_A, "bgr": (10, 20, 30)},
        {"tracker_id": 104, "box": BOX_B, "bgr": (40, 50, 60)},
        {"tracker_id": 105, "box": BOX_C, "bgr": (70, 80, 90)},
    ]
    frame, dets = build_call(rows)

    with caplog.at_level(logging.WARNING):
        linker.update(dets, frame)  # must not raise

    assert linker.roster == frozenset({1, 2, 3}), "the roster is assigned regardless — this is a warning, not a gate"

    # The warning must be ABOUT this: naming either an offending session id
    # (>= the offset) or the offset value itself. Accepting any record at
    # WARNING+ would be satisfied by an unrelated warning on another path.
    offenders = sorted({tiny_offset} | {sid for sid in linker.roster if sid >= tiny_offset})
    token = re.compile(r"\b(" + "|".join(str(n) for n in offenders) + r")\b")
    assert [r for r in caplog.records if r.levelno >= logging.WARNING and token.search(r.getMessage())], (
        "an assigned session id reaching unknown_id_offset must be logged as a warning that names "
        f"the offending id or the offset (looked for {offenders} in the WARNING records)"
    )


# ==========================================================================
# T08 — SPEC §B-R: reset & composition
# ==========================================================================


def test_b_r1_reset_keeps_the_persistent_galleries_for_the_next_enrolment():
    matcher = FakeMatcher()
    bgr = (10, 20, 30)
    marker = rgb_marker_for_bgr(bgr)
    matcher.program(marker, [({3: 0.9, 8: 0.05}, None)])
    linker = bind_linker(
        matcher,
        {3: persistent_gallery(3), 8: persistent_gallery(8)},
        bind_tau=0.5,
        bind_margin=0.1,
    )

    raw_id = 1101
    rows = [{"tracker_id": raw_id, "box": BOX_A, "bgr": bgr}]
    frame, dets = build_call(rows)
    linker.update(dets, frame)
    assert linker.roster == frozenset({3})

    linker.reset()
    assert linker.roster == frozenset()

    # A second recording binds against the SAME already-embedded galleries.
    frame, dets = build_call(rows)
    linker.update(dets, frame)
    assert linker.roster == frozenset({3}), (
        "reset() clears session state but keeps the constructor's persistent galleries, so the "
        "next enrolment can still bind"
    )
    assert len(score_calls_for_marker(matcher, marker)) == 2, "one bind per enrolment, no more"


def test_b_r1_the_second_enrolment_builds_exactly_one_start_gallery():
    """Re-enrolment costs exactly one gallery build: the new Start gallery.

    Honest scope note: this is NOT the Trap-5 (never-re-embed-the-persistent-
    views) guard it was originally named for, and it cannot fail for that
    reason — the linker is HANDED already-embedded galleries by its constructor
    and holds no crops for them, so it has nothing to re-embed even if it tried.
    What it does pin is that reset() + a second enrolment does not multiply
    gallery builds. The real Trap-5 coverage lives at the seam that actually
    owns the embedding, in
    test_tracking.py::test_b_r2_load_tracker_embeds_the_persistent_galleries_before_returning.
    """
    matcher = FakeMatcher()
    bgr = (10, 20, 30)
    marker = rgb_marker_for_bgr(bgr)
    matcher.program(marker, [({3: 0.9}, None)])
    linker = bind_linker(matcher, {3: persistent_gallery(3)}, bind_tau=0.5, bind_margin=0.1)

    rows = [{"tracker_id": 1101, "box": BOX_A, "bgr": bgr}]
    frame, dets = build_call(rows)
    linker.update(dets, frame)
    after_first_enrolment = len(matcher.build_gallery_calls)

    linker.reset()
    frame, dets = build_call(rows)
    linker.update(dets, frame)

    assert linker.roster == frozenset({3}), "sanity: the second enrolment still binds"
    # Exactly one further build_gallery call — the second recording's Start
    # gallery, and nothing besides.
    assert len(matcher.build_gallery_calls) == after_first_enrolment + 1, (
        "a second enrolment must build exactly one gallery: its own Start gallery"
    )


# ==========================================================================
# T08 — SPEC §B-O: observability
#
# These two tests pin the log PAYLOAD, never the log FORMAT. The freeze and
# batch-decision lines are the seams the T08 acceptance measurements read
# (Trap-4's atom asymmetry, Trap-5's build_ms), so the values below are a
# contract — but the wording that carries them is the implementation's to
# choose and re-word freely. Every assertion therefore searches the RENDERED
# message for a value; none matches a whole string. Scripted scores/specimen
# numbers are deliberately distinctive so a hit is evidence rather than a
# coincidence with some other number on the line.
# ==========================================================================


def test_b_o1_the_freeze_log_exposes_the_binding_it_just_solved(caplog):
    matcher = FakeMatcher()
    # Specimen numbers far from every other number this freeze could render
    # (raw ids, view counts, a sub-millisecond build_ms).
    galleries = {47: persistent_gallery(47), 62: persistent_gallery(62)}
    bound_bgr, session_only_bgr = (10, 20, 30), (40, 50, 60)
    bound_marker = rgb_marker_for_bgr(bound_bgr)
    session_only_marker = rgb_marker_for_bgr(session_only_bgr)
    # A MIXED bind: one identity takes a specimen, one falls back to
    # session-only. best/second for the binder are scripted to two values that
    # cannot coincide with anything else on the line.
    best, second = 0.91, 0.07
    matcher.program(bound_marker, [({47: best, 62: second}, None)])
    matcher.program(session_only_marker, [({47: 0.22, 62: 0.13}, None)])  # both under tau
    linker = bind_linker(matcher, galleries, bind_tau=0.5, bind_margin=0.1)

    rows = [
        {"tracker_id": 1201, "box": BOX_A, "bgr": bound_bgr},
        {"tracker_id": 1202, "box": BOX_B, "bgr": session_only_bgr},
    ]
    frame, dets = build_call(rows)
    with caplog.at_level(logging.INFO):
        linker.update(dets, frame)

    bound_id, session_only_id = 47, 63  # base = max({47, 62}) = 62 -> +1
    assert linker.roster == frozenset({bound_id, session_only_id}), (
        "sanity: this scenario must produce a mixed bind for the log to describe"
    )

    freeze_records = [r for r in caplog.records if r.levelno == logging.INFO and "freeze" in r.getMessage().lower()]
    assert freeze_records, (
        "the enrolment freeze must log an INFO line — the acceptance gate reads its payload back out"
    )
    message = " ".join(r.getMessage() for r in freeze_records)

    assert re.search(rf"\b{bound_id}\b", message), (
        f"the freeze log must expose the bound mapping (session id -> specimen); {bound_id} is absent "
        f"from {message!r}"
    )
    assert re.search(rf"\b{session_only_id}\b", message), (
        f"the freeze log must expose the session-only ids; {session_only_id} is absent from {message!r}"
    )
    # B-O1 fix: the log must expose the RAW tracker id that claimed the specimen,
    # not just the session id (which equals the specimen for every bound identity
    # and so reveals nothing about which track grabbed which photo set).
    assert re.search(r"\b1201\b", message), (
        f"the freeze log must expose the RAW tracker id that bound the specimen (B-O1); "
        f"1201 is absent from {message!r}"
    )
    assert str(best) in message, (
        f"the freeze log must expose the bound identity's BEST bind score ({best}); {message!r}"
    )
    assert str(second) in message, (
        f"the freeze log must expose the bound identity's SECOND bind score ({second}) — best alone "
        f"cannot show how close the bind was; {message!r}"
    )
    assert "build_ms" in message, (
        f"the freeze log must carry its total build_ms (now including the binding solves); {message!r}"
    )


def test_the_freeze_log_carries_a_structured_orc_event_for_the_debug_console(caplog):
    """The app's optional --debug console (backend.debug) renders the freeze from
    a structured `record.orc` payload rather than parsing the message string. Pin
    that contract here, at the producer, so a model change can't silently break
    the console: event=freeze, a per-identity roster of
    {session_id, raw_id, specimen, score}, bind_tau, and build_ms."""
    matcher = FakeMatcher()
    galleries = {47: persistent_gallery(47), 62: persistent_gallery(62)}
    bound_bgr, session_only_bgr = (10, 20, 30), (40, 50, 60)
    bound_marker = rgb_marker_for_bgr(bound_bgr)
    session_only_marker = rgb_marker_for_bgr(session_only_bgr)
    best, second = 0.91, 0.07
    matcher.program(bound_marker, [({47: best, 62: second}, None)])
    matcher.program(session_only_marker, [({47: 0.22, 62: 0.13}, None)])  # under tau
    linker = bind_linker(matcher, galleries, bind_tau=0.5, bind_margin=0.1)

    rows = [
        {"tracker_id": 1201, "box": BOX_A, "bgr": bound_bgr},
        {"tracker_id": 1202, "box": BOX_B, "bgr": session_only_bgr},
    ]
    frame, dets = build_call(rows)
    with caplog.at_level(logging.INFO):
        linker.update(dets, frame)

    freeze = next(
        (getattr(r, "orc", None) for r in caplog.records
         if getattr(r, "orc", None) and r.orc.get("event") == "freeze"),
        None,
    )
    assert freeze is not None, "the freeze record must carry a structured orc={'event':'freeze',...}"
    assert freeze["bind_tau"] == 0.5
    assert isinstance(freeze["build_ms"], float)
    by_raw = {e["raw_id"]: e for e in freeze["roster"]}
    assert set(by_raw) == {1201, 1202}, "every enrolled raw id must appear in the structured roster"
    assert by_raw[1201]["session_id"] == 47, "each roster entry must carry its session_id (the console labels by it)"
    assert by_raw[1201]["specimen"] == 47, "the bound identity's raw->specimen must be exposed structurally"
    assert by_raw[1201]["score"] == best
    assert by_raw[1202]["session_id"] == 63, "the session-only identity must still carry its session_id"
    assert by_raw[1202]["specimen"] is None, "the session-only identity must carry specimen=None"


def test_decision_and_death_logs_carry_structured_orc_events_for_the_debug_console(caplog):
    """Pin the producer side of the console contract for the OTHER two events.
    A track death and a batch decision must each emit `record.orc` with the right
    shape, so a model-side regression (a dropped/renamed `extra`) fails here rather
    than silently blanking the live --debug narrative. Drives real linker events;
    mirrors test_b3_death_after_absence_threshold_marks_identity_missing."""
    fps = 4.0
    enrolment_window_s = 0.25
    window = round(enrolment_window_s * fps)
    absent_death_s = 0.5
    death_threshold = round(absent_death_s * fps)
    matcher = FakeMatcher()
    linker = SessionLinker(
        matcher, fps=fps, unknown_id_offset=UNKNOWN_OFFSET,
        enrolment_window_s=enrolment_window_s, evidence_window_s=10.0,
        evidence_frames=1, absent_death_s=absent_death_s, min_mask_area_px=100,
    )
    session_id = 1
    for _ in range(window):
        frame, dets = build_call([{"tracker_id": 401, "box": BOX_A, "bgr": (70, 70, 70)}])
        linker.update(dets, frame)
    assert linker.roster == frozenset({session_id})

    with caplog.at_level(logging.INFO):
        for _ in range(death_threshold + 1):  # cross the absence threshold -> death event
            frame, dets = empty_call()
            linker.update(dets, frame)
        return_bgr = (5, 15, 200)
        marker = rgb_marker_for_bgr(return_bgr)
        matcher.program(marker, [({session_id: 0.9}, session_id)])
        frame, dets = build_call([{"tracker_id": 402, "box": BOX_B, "bgr": return_bgr}])
        linker.update(dets, frame)

    orcs = [getattr(r, "orc", None) for r in caplog.records if getattr(r, "orc", None)]
    death = next((o for o in orcs if o.get("event") == "death"), None)
    assert death is not None and death.get("session_id") == session_id, (
        "the death log must carry orc={'event':'death','session_id':...} for the console"
    )
    decision = next((o for o in orcs if o.get("event") == "decision"), None)
    assert decision is not None, "the batch-decision log must carry orc={'event':'decision',...}"
    assert decision["outcomes"].get(402) == f"linked:{session_id}", (
        "the decision payload must expose each raw id's outcome (raw 402 -> linked:1)"
    )


def test_b_o2_the_batch_decision_log_exposes_each_candidates_atom_count(caplog):
    matcher = FakeMatcher()
    # The mixed-bind scenario of test_b_b5_bound_identity_is_compared_as_
    # persistent_union_start, re-numbered so every count below lands on a
    # distinctive token. That test pins what score() RECEIVES; this one pins
    # what the linker LOGS — different surfaces, both load-bearing.
    persistent_views = 17
    galleries = {41: persistent_gallery(41, n_views=persistent_views)}
    bgr_a, bgr_b = (10, 20, 30), (40, 50, 60)
    marker_a, marker_b = rgb_marker_for_bgr(bgr_a), rgb_marker_for_bgr(bgr_b)
    matcher.program(marker_a, [({41: 0.9}, None)])  # binds
    matcher.program(marker_b, [({41: 0.2}, None)])  # under tau -> session-only
    # A multi-call enrolment window, so each identity carries SEVERAL Start
    # views. A one-call window would leave the unbound candidate on a single
    # atom, and a bare "1" is a token any incidental count on the line (a row
    # count, a candidate count) could supply — the search below would then pass
    # without the atom count ever being logged.
    enrolment_window_s = 0.75
    fps = 4.0
    window = round(enrolment_window_s * fps)
    linker = bind_linker(
        matcher, galleries, bind_tau=0.5, bind_margin=0.1, fps=fps, enrolment_window_s=enrolment_window_s
    )

    raw_bound, raw_unbound = 701, 702
    rows = [
        {"tracker_id": raw_bound, "box": BOX_A, "bgr": bgr_a},
        {"tracker_id": raw_unbound, "box": BOX_B, "bgr": bgr_b},
    ]
    with caplog.at_level(logging.INFO):
        for _ in range(window):
            frame, dets = build_call(rows)
            linker.update(dets, frame)
        bound_id, unbound_id = 41, 42  # base = max({41}) = 41 -> session-only starts at 42
        assert linker.roster == frozenset({bound_id, unbound_id})

        # Kill both, leaving exactly `persistent ∪ Start` vs `Start` to compare.
        death_threshold = round(0.5 * 4.0)
        for _ in range(death_threshold + 1):
            frame, dets = empty_call()
            linker.update(dets, frame)

        # Only the batch decision's own logs, so no keyword guess is needed to
        # find the record and no format gets pinned.
        caplog.clear()
        probe_bgr = (44, 55, 66)
        probe_marker = rgb_marker_for_bgr(probe_bgr)
        matcher.program(probe_marker, [({bound_id: 0.11, unbound_id: 0.12}, REJECT)])
        rows = [{"tracker_id": 801, "box": BOX_C, "bgr": probe_bgr}]
        frame, dets = build_call(rows)
        linker.update(dets, frame)

    probe_calls = score_calls_for_marker(matcher, probe_marker)
    assert probe_calls, "expected the probe's evidence window to close and trigger the batched decision"

    # Derived, never hardcoded: the atom count each candidate's comparison
    # gallery actually carries.
    bound_atoms = persistent_views + len(build_gallery_calls_for_marker(matcher, marker_a)[0]["crops"])
    unbound_atoms = len(build_gallery_calls_for_marker(matcher, marker_b)[0]["crops"])
    assert bound_atoms != unbound_atoms, "sanity: this scenario exists to produce the asymmetry"
    assert probe_calls[0]["gallery_view_counts"] == {bound_id: bound_atoms, unbound_id: unbound_atoms}, (
        "sanity (B-B5's surface): the decision must actually see the bound-vs-unbound asymmetry "
        "before its log can report it"
    )

    decision_records = [r for r in caplog.records if r.levelno == logging.INFO]
    assert decision_records, "the batched decision must log an INFO line — replay traces read it back out"
    message = " ".join(r.getMessage() for r in decision_records)

    assert re.search(rf"\b{bound_atoms}\b", message), (
        f"the batch-decision log must expose the per-candidate atom count (views.shape[0]); the bound "
        f"candidate's {bound_atoms} is absent from {message!r}"
    )
    assert re.search(rf"\b{unbound_atoms}\b", message), (
        f"the batch-decision log must expose the per-candidate atom count for EVERY candidate; the "
        f"unbound candidate's {unbound_atoms} is absent from {message!r}"
    )


# --------------------------------------------------------------------------
# data["resolving"] — the wait-state flag (seam widening, grilled 2026-07-16)
#
# The offset (not-in-roster) id range covers three states the consumer draws
# differently. `data["resolving"]` splits them: True while a track is still
# being decided (pending in its evidence window, OR deferred behind a coasting
# active id), False once it has settled — whether it settled into a roster id
# (linked) or into a settled Unknown (rejected). A settled Unknown is NOT
# permanent: it can re-arm back to Pending (resolving True again) on a material
# change or target expansion — see the B-UR section. Roster ids and enrolment
# are covered too. This is the truth the overlay spinner gates on, so it must
# track the linker's ACTUAL decision, never a clock.
# --------------------------------------------------------------------------


def resolving_of(out: sv.Detections, rows, raw_id: int) -> bool:
    """The OUTPUT resolving flag at the row `raw_id` was placed at."""
    idx = next(i for i, r in enumerate(rows) if r["tracker_id"] == raw_id)
    assert "resolving" in out.data, "update() must emit a per-detection data['resolving'] flag"
    return bool(out.data["resolving"][idx])


def test_resolving_flag_marks_enrolment_then_clears_for_a_roster_id_on_freeze():
    fps = 4.0
    enrolment_window_s = 0.5
    window = round(enrolment_window_s * fps)  # 2: one pre-freeze call, then the freeze
    matcher = FakeMatcher()
    linker = SessionLinker(
        matcher,
        fps=fps,
        unknown_id_offset=UNKNOWN_OFFSET,
        enrolment_window_s=enrolment_window_s,
        evidence_window_s=10.0,
        evidence_frames=1,
        absent_death_s=10.0,
        min_mask_area_px=100,
    )
    raw_id = 301

    # Pre-freeze: nothing has a settled identity yet, so every track resolves.
    rows = [{"tracker_id": raw_id, "box": BOX_A, "bgr": (10, 10, 10)}]
    frame, dets = build_call(rows)
    out = linker.update(dets, frame)
    assert linker.roster == frozenset(), "sanity: window not closed on the first call"
    assert resolving_of(out, rows, raw_id) is True

    # Freeze frame: the roster publishes and its member is settled at once.
    rows = [{"tracker_id": raw_id, "box": BOX_A, "bgr": (10, 10, 10)}]
    frame, dets = build_call(rows)
    out = linker.update(dets, frame)
    assert linker.roster == frozenset({1})
    assert resolving_of(out, rows, raw_id) is False


def test_resolving_flag_true_while_pending_then_false_once_linked_and_active():
    fps = 4.0
    enrolment_window_s = 0.25
    window = round(enrolment_window_s * fps)
    absent_death_s = 0.5
    death_threshold = round(absent_death_s * fps)
    matcher = FakeMatcher()
    linker = SessionLinker(
        matcher,
        fps=fps,
        unknown_id_offset=UNKNOWN_OFFSET,
        enrolment_window_s=enrolment_window_s,
        evidence_window_s=10.0,
        evidence_frames=2,  # two crops needed, so we can observe Pending vs decided
        absent_death_s=absent_death_s,
        min_mask_area_px=100,
    )
    raw_roster_id = 601
    session_id = 1
    for _ in range(window):
        rows = [{"tracker_id": raw_roster_id, "box": BOX_A, "bgr": (80, 80, 80)}]
        frame, dets = build_call(rows)
        linker.update(dets, frame)
    assert linker.roster == frozenset({session_id})

    for _ in range(death_threshold + 1):  # let it die so the returner is a link target
        frame, dets = empty_call()
        linker.update(dets, frame)

    return_bgr = (11, 22, 33)
    marker = rgb_marker_for_bgr(return_bgr)
    matcher.program(marker, [({session_id: 0.9}, session_id)])
    return_raw_id = 602

    # First appearance: evidence window still open -> Pending -> resolving.
    rows = [{"tracker_id": return_raw_id, "box": BOX_B, "bgr": return_bgr}]
    frame, dets = build_call(rows)
    out = linker.update(dets, frame)
    assert output_tracker_id(out, rows, return_raw_id) == return_raw_id + UNKNOWN_OFFSET
    assert resolving_of(out, rows, return_raw_id) is True

    # Second appearance: the window closes, the batch links it THIS call, and
    # the flag clears in the same frame the id flips to the session id.
    rows = [{"tracker_id": return_raw_id, "box": BOX_B, "bgr": return_bgr}]
    frame, dets = build_call(rows)
    out = linker.update(dets, frame)
    assert output_tracker_id(out, rows, return_raw_id) == session_id
    assert resolving_of(out, rows, return_raw_id) is False

    # And an Active, mapped identity never resolves on later frames.
    rows = [{"tracker_id": return_raw_id, "box": BOX_B, "bgr": return_bgr}]
    frame, dets = build_call(rows)
    out = linker.update(dets, frame)
    assert resolving_of(out, rows, return_raw_id) is False


def test_resolving_flag_stays_true_while_a_track_is_deferred_then_clears_on_link():
    # The wait-state bug this whole change fixes: a returning track deferred
    # behind a still-coasting active id must keep resolving (spinner), not read
    # as settled Unknown, until it actually links. Mirrors
    # test_b5_coasting_identity_handoff_waits_for_missing_then_revalidates.
    fps = 4.0
    absent_death_s = 0.5
    matcher = FakeMatcher()
    linker = SessionLinker(
        matcher,
        fps=fps,
        unknown_id_offset=UNKNOWN_OFFSET,
        enrolment_window_s=0.25,
        evidence_window_s=10.0,
        evidence_frames=1,
        absent_death_s=absent_death_s,
        min_mask_area_px=100,
    )
    raw_roster_id = 1501
    session_id = 1
    rows = [{"tracker_id": raw_roster_id, "box": BOX_A, "bgr": (33, 33, 33)}]
    frame, dets = build_call(rows)
    linker.update(dets, frame)
    assert linker.roster == frozenset({session_id})

    frame, dets = empty_call()  # old raw id absent one frame, not yet dead (coasting)
    linker.update(dets, frame)

    return_bgr = (9, 19, 209)
    marker = rgb_marker_for_bgr(return_bgr)
    matcher.program(marker, [({session_id: 0.95}, session_id), ({session_id: 0.95}, session_id)])
    return_raw_id = 1601
    rows = [{"tracker_id": return_raw_id, "box": BOX_B, "bgr": return_bgr}]

    # Deferred (accepted id is still Active-coasting): offset id AND resolving.
    frame, dets = build_call(rows)
    provisional = linker.update(dets, frame)
    assert output_tracker_id(provisional, rows, return_raw_id) == return_raw_id + UNKNOWN_OFFSET
    assert resolving_of(provisional, rows, return_raw_id) is True, (
        "a deferred track is still being decided — it must resolve (spinner), not read as settled"
    )

    # One frame later the old id is Missing, the deferred row links, flag clears.
    frame, dets = build_call(rows)
    linked = linker.update(dets, frame)
    assert output_tracker_id(linked, rows, return_raw_id) == session_id
    assert resolving_of(linked, rows, return_raw_id) is False


def test_resolving_flag_false_once_a_track_settles_unknown():
    fps = 4.0
    absent_death_s = 0.5
    death_threshold = round(absent_death_s * fps)
    matcher = FakeMatcher()
    linker = SessionLinker(
        matcher,
        fps=fps,
        unknown_id_offset=UNKNOWN_OFFSET,
        enrolment_window_s=0.25,
        evidence_window_s=10.0,
        evidence_frames=2,  # first appearance Pending, second closes -> rejected
        absent_death_s=absent_death_s,
        min_mask_area_px=100,
    )
    raw_roster_id = 1701
    session_id = 1
    rows = [{"tracker_id": raw_roster_id, "box": BOX_A, "bgr": (33, 33, 33)}]
    frame, dets = build_call(rows)
    linker.update(dets, frame)
    assert linker.roster == frozenset({session_id})

    for _ in range(death_threshold + 1):  # kill it so the foreign track has a target to reject
        frame, dets = empty_call()
        linker.update(dets, frame)

    foreign_bgr = (3, 4, 220)
    marker = rgb_marker_for_bgr(foreign_bgr)
    matcher.program(marker, [({session_id: 0.1}, REJECT)])
    raw_id = 1801

    # First appearance: Pending -> resolving.
    rows = [{"tracker_id": raw_id, "box": BOX_C, "bgr": foreign_bgr}]
    frame, dets = build_call(rows)
    out = linker.update(dets, frame)
    assert resolving_of(out, rows, raw_id) is True

    # Second appearance: window closes, matcher REJECTs -> settled Unknown; the
    # flag clears even though the id stays in the offset range.
    rows = [{"tracker_id": raw_id, "box": BOX_C, "bgr": foreign_bgr}]
    frame, dets = build_call(rows)
    out = linker.update(dets, frame)
    assert output_tracker_id(out, rows, raw_id) == raw_id + UNKNOWN_OFFSET
    assert resolving_of(out, rows, raw_id) is False

    # And with the same (unchanged) crop it stays settled — not resolving — on
    # the next frame too. (It is no longer *permanently* settled: a material
    # change or target expansion could re-arm it; see the B-UR section.)
    rows = [{"tracker_id": raw_id, "box": BOX_C, "bgr": foreign_bgr}]
    frame, dets = build_call(rows)
    out = linker.update(dets, frame)
    assert output_tracker_id(out, rows, raw_id) == raw_id + UNKNOWN_OFFSET
    assert resolving_of(out, rows, raw_id) is False


# ==========================================================================
# B-UR — Recoverable Unknown via change-triggered re-identification
# (model/docs/unknown-recovery-SPEC.md). A settled Unknown is no longer
# terminal for the lifetime of its raw id: while the raw track is present, the
# linker cheaply watches its crop and candidate target set, and re-arms a fresh
# Pending window when the crop changes materially OR a new link target appears.
# The tests below pass EXPLICIT recheck thresholds and never assert the defaults.
# ==========================================================================


def recovery_linker(
    matcher,
    *,
    fps=4.0,
    cooldown_s=0.0,
    appearance_delta=0.15,
    mask_iou=0.75,
    quality_gain=0.25,
    **kwargs,
):
    """A linker wired for the recovery tests. `cooldown_s=0.0` (the default here)
    makes a re-arm permissible on the very next frame after a decision, so tests
    that are not about the cooldown never trip over it; test_ur9 sets it."""
    options = dict(
        enrolment_window_s=0.25,
        evidence_window_s=10.0,
        evidence_frames=1,
        absent_death_s=0.5,
        min_mask_area_px=100,
    )
    options.update(kwargs)
    return SessionLinker(
        matcher,
        fps=fps,
        unknown_id_offset=UNKNOWN_OFFSET,
        unknown_recheck_cooldown_s=cooldown_s,
        unknown_recheck_appearance_delta=appearance_delta,
        unknown_recheck_mask_iou=mask_iou,
        unknown_recheck_quality_gain=quality_gain,
        **options,
    )


def _enrol(linker, rows_by_frame, *, window):
    """Drive an enrolment window from a list of per-frame row lists (or a single
    row list repeated for `window` frames)."""
    if isinstance(rows_by_frame, list) and rows_by_frame and isinstance(rows_by_frame[0], dict):
        rows_by_frame = [rows_by_frame] * window
    for rows in rows_by_frame:
        frame, dets = build_call(rows)
        linker.update(dets, frame)


def _kill(linker, *, death_threshold):
    for _ in range(death_threshold + 1):
        frame, dets = empty_call()
        linker.update(dets, frame)


def test_ur1_hand_occlusion_recovery_reidentifies_original_session_id():
    """PRIMARY REGRESSION (SPEC TDD #1, definition-of-done). An occluded window
    rejects; unchanged occluded crops do not retry; a materially different
    unobstructed crop opens a fresh window; the matcher accepts and the SAME raw
    id emits the original session id."""
    fps = 4.0
    window = round(0.25 * fps)
    death_threshold = round(0.5 * fps)
    matcher = FakeMatcher()
    linker = recovery_linker(matcher, fps=fps, cooldown_s=0.0, evidence_frames=1)

    session_id = 1
    _enrol(linker, [{"tracker_id": 601, "box": BOX_A, "bgr": (80, 80, 80)}], window=window)
    assert linker.roster == frozenset({session_id})
    _kill(linker, death_threshold=death_threshold)  # identity 1 -> Missing

    occluded_bgr = (10, 10, 10)    # hand-covered crop -> rejected
    clean_bgr = (60, 60, 200)      # hand gone: materially different -> accepted
    occluded_marker = rgb_marker_for_bgr(occluded_bgr)
    clean_marker = rgb_marker_for_bgr(clean_bgr)
    matcher.program(occluded_marker, [({session_id: 0.1}, REJECT)])
    matcher.program(clean_marker, [({session_id: 0.9}, session_id)])

    recovering_raw_id = 602  # a NEW raw id, born while the hand covered the object

    # Initial evidence window is occluded -> rejected -> settled Unknown.
    rows = [{"tracker_id": recovering_raw_id, "box": BOX_B, "bgr": occluded_bgr}]
    frame, dets = build_call(rows)
    out = linker.update(dets, frame)
    assert output_tracker_id(out, rows, recovering_raw_id) == recovering_raw_id + UNKNOWN_OFFSET
    assert resolving_of(out, rows, recovering_raw_id) is False, "a rejected track has settled, not resolving"
    assert len(score_calls_for_marker(matcher, occluded_marker)) == 1

    # Hand still covering: unchanged occluded crops must NOT re-invoke the matcher.
    for _ in range(4):
        rows = [{"tracker_id": recovering_raw_id, "box": BOX_B, "bgr": occluded_bgr}]
        frame, dets = build_call(rows)
        out = linker.update(dets, frame)
        assert output_tracker_id(out, rows, recovering_raw_id) == recovering_raw_id + UNKNOWN_OFFSET
        assert resolving_of(out, rows, recovering_raw_id) is False
    assert len(score_calls_for_marker(matcher, occluded_marker)) == 1, (
        "an unchanged settled Unknown must not retry the matcher"
    )

    # Hand leaves: the crop changes materially -> a fresh Pending window opens,
    # the matcher scores it once, accepts, and the SAME raw id recovers session 1.
    rows = [{"tracker_id": recovering_raw_id, "box": BOX_B, "bgr": clean_bgr}]
    frame, dets = build_call(rows)
    out = linker.update(dets, frame)
    assert len(score_calls_for_marker(matcher, clean_marker)) == 1, "the material change must re-arm and score once"
    assert output_tracker_id(out, rows, recovering_raw_id) == session_id, (
        "recovery links the same raw id back to its original session id"
    )
    assert resolving_of(out, rows, recovering_raw_id) is False, "linked and Active -> settled"


def test_ur2_static_foreign_object_does_no_repeated_matcher_work():
    """SPEC TDD #2 / B-UR5. After rejection, many unchanged frames cause no
    additional score() calls and stay resolving=False."""
    fps = 4.0
    window = round(0.25 * fps)
    death_threshold = round(0.5 * fps)
    matcher = FakeMatcher()
    linker = recovery_linker(matcher, fps=fps, cooldown_s=0.0, evidence_frames=1)

    session_id = 1
    _enrol(linker, [{"tracker_id": 701, "box": BOX_A, "bgr": (80, 80, 80)}], window=window)
    _kill(linker, death_threshold=death_threshold)  # identity 1 -> Missing

    foreign_bgr = (44, 55, 66)
    marker = rgb_marker_for_bgr(foreign_bgr)
    matcher.program(marker, [({session_id: 0.1}, REJECT)])
    foreign_raw_id = 801
    rows = [{"tracker_id": foreign_raw_id, "box": BOX_B, "bgr": foreign_bgr}]
    frame, dets = build_call(rows)
    out = linker.update(dets, frame)
    assert output_tracker_id(out, rows, foreign_raw_id) == foreign_raw_id + UNKNOWN_OFFSET
    calls_after_settle = len(matcher.score_calls)
    assert calls_after_settle == 1

    for _ in range(12):
        rows = [{"tracker_id": foreign_raw_id, "box": BOX_B, "bgr": foreign_bgr}]
        frame, dets = build_call(rows)
        out = linker.update(dets, frame)
        assert output_tracker_id(out, rows, foreign_raw_id) == foreign_raw_id + UNKNOWN_OFFSET
        assert resolving_of(out, rows, foreign_raw_id) is False
    assert len(matcher.score_calls) == calls_after_settle, (
        "a static Unknown must not add a single matcher call after settling"
    )


def test_ur3_recovery_window_holds_only_fresh_crops_not_rejected_ones():
    """SPEC TDD #3 / B-UR2. The recovery matcher's crop markers contain the new
    unobstructed window and none from the original rejected window."""
    fps = 4.0
    window = round(0.25 * fps)
    death_threshold = round(0.5 * fps)
    matcher = FakeMatcher()
    # evidence_frames=2 so each window spans two crops: enough to prove the fresh
    # window carries only clean crops.
    linker = recovery_linker(matcher, fps=fps, cooldown_s=0.0, evidence_frames=2)

    session_id = 1
    _enrol(linker, [{"tracker_id": 901, "box": BOX_A, "bgr": (80, 80, 80)}], window=window)
    _kill(linker, death_threshold=death_threshold)

    occluded_bgr = (10, 10, 10)
    clean_bgr = (60, 60, 200)
    occluded_marker = rgb_marker_for_bgr(occluded_bgr)
    clean_marker = rgb_marker_for_bgr(clean_bgr)
    matcher.program(occluded_marker, [({session_id: 0.1}, REJECT)])
    matcher.program(clean_marker, [({session_id: 0.9}, session_id)])
    raw_id = 902

    # Two occluded crops close the initial window -> rejected.
    for _ in range(2):
        rows = [{"tracker_id": raw_id, "box": BOX_B, "bgr": occluded_bgr}]
        frame, dets = build_call(rows)
        linker.update(dets, frame)
    assert len(score_calls_for_marker(matcher, occluded_marker)) == 1

    # Two clean crops: first re-arms + seeds the fresh window, second closes it.
    for _ in range(2):
        rows = [{"tracker_id": raw_id, "box": BOX_B, "bgr": clean_bgr}]
        frame, dets = build_call(rows)
        out = linker.update(dets, frame)
    assert output_tracker_id(out, rows, raw_id) == session_id

    clean_calls = score_calls_for_marker(matcher, clean_marker)
    assert len(clean_calls) == 1, "exactly one recovery decision"
    markers = crop_markers(clean_calls[0]["crops"])
    assert markers == {clean_marker}, "the recovery window must contain only fresh (clean) crops"
    assert occluded_marker not in markers, "no crop from the rejected window may leak into the recovery window"


def test_ur4_subthreshold_jitter_does_not_rearm():
    """SPEC TDD #4 / B-UR5. Below-threshold appearance, mask, and quality changes
    do not re-arm (no spinner/matcher loop)."""
    fps = 4.0
    window = round(0.25 * fps)
    death_threshold = round(0.5 * fps)
    matcher = FakeMatcher()
    linker = recovery_linker(
        matcher, fps=fps, cooldown_s=0.0, evidence_frames=1, appearance_delta=0.15
    )

    session_id = 1
    _enrol(linker, [{"tracker_id": 1001, "box": BOX_A, "bgr": (80, 80, 80)}], window=window)
    _kill(linker, death_threshold=death_threshold)

    baseline_bgr = (100, 100, 100)
    marker = rgb_marker_for_bgr(baseline_bgr)
    matcher.program(marker, [({session_id: 0.1}, REJECT)])
    raw_id = 1002
    rows = [{"tracker_id": raw_id, "box": BOX_B, "bgr": baseline_bgr}]
    frame, dets = build_call(rows)
    linker.update(dets, frame)
    calls_after_settle = len(matcher.score_calls)
    assert calls_after_settle == 1

    # Small per-channel colour jitter (mean |Δ| ≈ 8/255 ≈ 0.03 « 0.15) AND a small
    # confidence bump (0.9 -> 0.99 ≈ +10% quality « 25%), same box so mask IoU
    # stays 1.0. Every signal is below its gate. (The synthetic harness always
    # paints full-box masks, so mask IoU is structurally 1.0 here and the mask
    # gate is exercised only in the negative — a partial-mask fixture would be
    # needed to drive it below threshold.) These colours are deliberately NOT
    # programmed: a wrong re-arm would call score() on an unprogrammed marker and
    # raise, so this asserts the gate holds.
    jitter = [
        ((108, 104, 96), 0.99),
        ((94, 106, 108), 0.86),
        ((105, 95, 103), 0.95),
    ]
    for jitter_bgr, conf in jitter:
        rows = [{"tracker_id": raw_id, "box": BOX_B, "bgr": jitter_bgr, "confidence": conf}]
        frame, dets = build_call(rows)
        out = linker.update(dets, frame)
        assert output_tracker_id(out, rows, raw_id) == raw_id + UNKNOWN_OFFSET
        assert resolving_of(out, rows, raw_id) is False
    assert len(matcher.score_calls) == calls_after_settle, (
        "sub-threshold appearance/quality jitter must not re-arm"
    )


def test_ur5_rejected_recovery_advances_the_baseline():
    """SPEC TDD #5 / B-UR6. A changed crop triggers exactly one retry; rejection
    updates the baseline; repeated identical crops do not trigger again."""
    fps = 4.0
    window = round(0.25 * fps)
    death_threshold = round(0.5 * fps)
    matcher = FakeMatcher()
    linker = recovery_linker(matcher, fps=fps, cooldown_s=0.0, evidence_frames=1)

    session_id = 1
    _enrol(linker, [{"tracker_id": 1101, "box": BOX_A, "bgr": (80, 80, 80)}], window=window)
    _kill(linker, death_threshold=death_threshold)

    occluded_bgr = (10, 10, 10)
    changed_bgr = (60, 60, 200)
    occluded_marker = rgb_marker_for_bgr(occluded_bgr)
    changed_marker = rgb_marker_for_bgr(changed_bgr)
    matcher.program(occluded_marker, [({session_id: 0.1}, REJECT)])
    # Exactly ONE programmed recovery response: a second retry on the same changed
    # crop would consume a second (absent) response and raise.
    matcher.program(changed_marker, [({session_id: 0.2}, REJECT)])
    raw_id = 1102

    rows = [{"tracker_id": raw_id, "box": BOX_B, "bgr": occluded_bgr}]
    frame, dets = build_call(rows)
    linker.update(dets, frame)

    # First changed crop: re-arm -> one retry -> rejected -> new baseline = changed.
    rows = [{"tracker_id": raw_id, "box": BOX_B, "bgr": changed_bgr}]
    frame, dets = build_call(rows)
    out = linker.update(dets, frame)
    assert output_tracker_id(out, rows, raw_id) == raw_id + UNKNOWN_OFFSET
    assert len(score_calls_for_marker(matcher, changed_marker)) == 1

    # The SAME changed crop is now the baseline: it must not re-arm again.
    for _ in range(4):
        rows = [{"tracker_id": raw_id, "box": BOX_B, "bgr": changed_bgr}]
        frame, dets = build_call(rows)
        out = linker.update(dets, frame)
        assert output_tracker_id(out, rows, raw_id) == raw_id + UNKNOWN_OFFSET
    assert len(score_calls_for_marker(matcher, changed_marker)) == 1, (
        "the rejected recovery advanced the baseline; the same crop must not retry again"
    )


def test_ur6_target_set_expansion_rearms_with_unchanged_crop():
    """SPEC TDD #6 / B-UR3. An Unknown first decided with no Missing/deferable
    identity retries once when the original identity becomes Missing, even if the
    crop is unchanged."""
    fps = 4.0
    matcher = FakeMatcher()
    # death_threshold = 0 so the roster identity dies (Missing) on its first
    # absent frame with no coasting-candidate window in between — keeping the
    # recovery a single, undeferred decision.
    linker = recovery_linker(
        matcher, fps=fps, cooldown_s=0.0, evidence_frames=1,
        enrolment_window_s=0.25, absent_death_s=0.1,
    )
    death_threshold = round(0.1 * fps)
    assert death_threshold == 0
    window = round(0.25 * fps)

    session_id = 1
    _enrol(linker, [{"tracker_id": 501, "box": BOX_A, "bgr": (80, 80, 80)}], window=window)
    assert linker.roster == frozenset({session_id})

    foreign_bgr = (44, 55, 66)
    marker = rgb_marker_for_bgr(foreign_bgr)
    # Only consumed at the recovery decision (the initial settle has no candidate,
    # so the matcher is never called there).
    matcher.program(marker, [({session_id: 0.1}, REJECT)])
    foreign_raw_id = 601

    # identity 1 Active-present, foreign present: no candidate target exists, so
    # the foreign settles Unknown WITHOUT any matcher call (candidate set empty).
    rows = [
        {"tracker_id": 501, "box": BOX_A, "bgr": (80, 80, 80)},
        {"tracker_id": foreign_raw_id, "box": BOX_B, "bgr": foreign_bgr},
    ]
    frame, dets = build_call(rows)
    linker.update(dets, frame)
    assert not matcher.score_calls, "no candidate target -> settle Unknown with no matcher call"

    # 501 leaves -> identity 1 becomes Missing. The foreign's crop is UNCHANGED,
    # but the candidate target set expanded from {} to {1}: re-arm once.
    rows = [{"tracker_id": foreign_raw_id, "box": BOX_B, "bgr": foreign_bgr}]
    frame, dets = build_call(rows)
    out = linker.update(dets, frame)
    assert output_tracker_id(out, rows, foreign_raw_id) == foreign_raw_id + UNKNOWN_OFFSET
    assert len(score_calls_for_marker(matcher, marker)) == 1, "target-set expansion re-armed the Unknown"

    # New baseline saved the expanded set: an unchanged crop must not retry again.
    for _ in range(4):
        rows = [{"tracker_id": foreign_raw_id, "box": BOX_B, "bgr": foreign_bgr}]
        frame, dets = build_call(rows)
        linker.update(dets, frame)
    assert len(score_calls_for_marker(matcher, marker)) == 1, "the expansion retry happens at most once per expansion"


def test_ur7_recovery_never_links_a_fully_present_active_identity():
    """SPEC TDD #7 / safety invariant #3. A matcher result naming a fully present
    Active identity never links it, even on a recovery attempt."""
    fps = 4.0
    window = round(0.25 * fps)
    death_threshold = round(0.5 * fps)
    matcher = FakeMatcher()
    linker = recovery_linker(matcher, fps=fps, cooldown_s=0.0, evidence_frames=1)

    missing_id, active_id = 1, 2
    # Enrol two identities; both present for the whole window.
    _enrol(
        linker,
        [
            {"tracker_id": 501, "box": BOX_A, "bgr": (80, 80, 80)},
            {"tracker_id": 502, "box": BOX_B, "bgr": (120, 120, 120)},
        ],
        window=window,
    )
    assert linker.roster == frozenset({missing_id, active_id})

    # Kill only identity 1 (raw 501); keep identity 2 (raw 502) present.
    for _ in range(death_threshold + 1):
        rows = [{"tracker_id": 502, "box": BOX_B, "bgr": (120, 120, 120)}]
        frame, dets = build_call(rows)
        linker.update(dets, frame)

    occluded_bgr = (10, 10, 10)
    clean_bgr = (60, 60, 200)
    occluded_marker = rgb_marker_for_bgr(occluded_bgr)
    clean_marker = rgb_marker_for_bgr(clean_bgr)
    matcher.program(occluded_marker, [({missing_id: 0.1, active_id: 0.1}, REJECT)])
    # The recovery names the fully-present Active identity 2 -> must never link it.
    matcher.program(clean_marker, [({missing_id: 0.4, active_id: 0.95}, active_id)])
    foreign_raw_id = 601

    rows = [
        {"tracker_id": 502, "box": BOX_B, "bgr": (120, 120, 120)},
        {"tracker_id": foreign_raw_id, "box": BOX_C, "bgr": occluded_bgr},
    ]
    frame, dets = build_call(rows)
    linker.update(dets, frame)

    rows = [
        {"tracker_id": 502, "box": BOX_B, "bgr": (120, 120, 120)},
        {"tracker_id": foreign_raw_id, "box": BOX_C, "bgr": clean_bgr},
    ]
    frame, dets = build_call(rows)
    out = linker.update(dets, frame)
    assert len(score_calls_for_marker(matcher, clean_marker)) == 1, "the change re-armed and scored"
    assert output_tracker_id(out, rows, foreign_raw_id) == foreign_raw_id + UNKNOWN_OFFSET, (
        "a fully present Active identity is never an eligible link target"
    )
    assert output_tracker_id(out, rows, 502) == active_id, "the present Active identity keeps its own session id"


def test_ur8_no_evidence_unknown_recovers_after_first_usable_crop():
    """SPEC TDD #8 / B-UR7. A no-crop rejection stores a None fingerprint; the
    first usable crop that later appears (with a candidate target) is sufficient
    evidence change to open one fresh Pending window."""
    fps = 4.0
    window = round(0.25 * fps)
    death_threshold = round(0.5 * fps)
    matcher = FakeMatcher()
    # evidence_frames=1 with a 2-frame evidence window: an all-unusable window
    # times out on elapsed frames with an empty buffer -> no-crop settle.
    linker = recovery_linker(
        matcher, fps=fps, cooldown_s=0.0, evidence_frames=1, evidence_window_s=0.5
    )
    evidence_window = round(0.5 * fps)
    assert evidence_window == 2

    session_id = 1
    _enrol(linker, [{"tracker_id": 1201, "box": BOX_A, "bgr": (80, 80, 80)}], window=window)
    _kill(linker, death_threshold=death_threshold)

    good_bgr = (60, 60, 200)
    good_marker = rgb_marker_for_bgr(good_bgr)
    matcher.program(good_marker, [({session_id: 0.9}, session_id)])
    raw_id = 1202
    tiny_box = (30, 30, 34, 34)  # 16 px^2 -> never a usable crop

    # Two unusable (tiny) crops -> the window times out with no usable crop ->
    # no-crop settle (fingerprint None), no matcher call.
    for _ in range(evidence_window):
        rows = [{"tracker_id": raw_id, "box": tiny_box, "bgr": good_bgr}]
        frame, dets = build_call(rows)
        out = linker.update(dets, frame)
        assert output_tracker_id(out, rows, raw_id) == raw_id + UNKNOWN_OFFSET
    assert not matcher.score_calls, "a no-crop window settles Unknown without any matcher call"

    # A usable crop finally appears -> that is new evidence for a None fingerprint
    # -> re-arm -> recovery decision links it back to session 1.
    rows = [{"tracker_id": raw_id, "box": BOX_B, "bgr": good_bgr}]
    frame, dets = build_call(rows)
    out = linker.update(dets, frame)
    assert len(score_calls_for_marker(matcher, good_marker)) == 1, "the first usable crop re-armed a no-evidence Unknown"
    assert output_tracker_id(out, rows, raw_id) == session_id


def test_ur9_cooldown_blocks_additional_decisions_then_allows_one():
    """SPEC TDD #9 / cooldown. Repeated above-threshold changes inside the
    configured cooldown do not create additional decisions; once it expires, the
    next changed crop re-arms."""
    fps = 4.0
    window = round(0.25 * fps)
    death_threshold = round(0.5 * fps)
    cooldown_s = 0.75
    cooldown_frames = round(cooldown_s * fps)
    assert cooldown_frames == 3
    matcher = FakeMatcher()
    linker = recovery_linker(matcher, fps=fps, cooldown_s=cooldown_s, evidence_frames=1)

    session_id = 1
    _enrol(linker, [{"tracker_id": 1301, "box": BOX_A, "bgr": (80, 80, 80)}], window=window)
    _kill(linker, death_threshold=death_threshold)

    occluded_bgr = (10, 10, 10)
    changed_bgr = (60, 60, 200)
    occluded_marker = rgb_marker_for_bgr(occluded_bgr)
    changed_marker = rgb_marker_for_bgr(changed_bgr)
    matcher.program(occluded_marker, [({session_id: 0.1}, REJECT)])
    matcher.program(changed_marker, [({session_id: 0.9}, session_id)])
    raw_id = 1302

    # Settle at frame F (last_decision_frame = F).
    rows = [{"tracker_id": raw_id, "box": BOX_B, "bgr": occluded_bgr}]
    frame, dets = build_call(rows)
    linker.update(dets, frame)
    assert len(matcher.score_calls) == 1

    # Frames F+1, F+2 are inside the cooldown (elapsed 1, 2 < 3): even a big,
    # above-threshold change must not create a decision.
    for _ in range(cooldown_frames - 1):
        rows = [{"tracker_id": raw_id, "box": BOX_B, "bgr": changed_bgr}]
        frame, dets = build_call(rows)
        out = linker.update(dets, frame)
        assert output_tracker_id(out, rows, raw_id) == raw_id + UNKNOWN_OFFSET
    assert len(score_calls_for_marker(matcher, changed_marker)) == 0, "no decision may fire inside the cooldown"

    # Frame F+3: cooldown expired (elapsed 3 >= 3) -> the change now re-arms.
    rows = [{"tracker_id": raw_id, "box": BOX_B, "bgr": changed_bgr}]
    frame, dets = build_call(rows)
    out = linker.update(dets, frame)
    assert len(score_calls_for_marker(matcher, changed_marker)) == 1, "the cooldown gates timing, it is not permanent"
    assert output_tracker_id(out, rows, raw_id) == session_id


def test_ur10_settled_unknown_is_aged_out_and_reoccurrence_is_new_pending():
    """SPEC TDD #10 / B-UR8. Settled Unknown state is discarded after the death
    threshold; a later occurrence of the raw id is treated as a new Pending
    track (so an unchanged crop is decided afresh, not gated by a stale
    fingerprint)."""
    fps = 4.0
    window = round(0.25 * fps)
    death_threshold = round(0.5 * fps)
    matcher = FakeMatcher()
    linker = recovery_linker(matcher, fps=fps, cooldown_s=0.0, evidence_frames=1)

    session_id = 1
    _enrol(linker, [{"tracker_id": 1401, "box": BOX_A, "bgr": (80, 80, 80)}], window=window)
    _kill(linker, death_threshold=death_threshold)

    foreign_bgr = (44, 55, 66)
    marker = rgb_marker_for_bgr(foreign_bgr)
    # Two REJECTs: one at the first settle, one when the aged-out id is decided
    # as a fresh Pending. A stale settled state would suppress the second call.
    matcher.program(marker, [({session_id: 0.1}, REJECT), ({session_id: 0.1}, REJECT)])
    raw_id = 1501

    rows = [{"tracker_id": raw_id, "box": BOX_B, "bgr": foreign_bgr}]
    frame, dets = build_call(rows)
    linker.update(dets, frame)
    assert len(score_calls_for_marker(matcher, marker)) == 1

    # Absent long enough to age the settled Unknown out entirely.
    for _ in range(death_threshold + 1):
        frame, dets = empty_call()
        linker.update(dets, frame)

    # Reappears with the SAME crop: because the settled state was discarded this
    # is a brand-new Pending track and gets its own fresh decision.
    rows = [{"tracker_id": raw_id, "box": BOX_B, "bgr": foreign_bgr}]
    frame, dets = build_call(rows)
    out = linker.update(dets, frame)
    assert output_tracker_id(out, rows, raw_id) == raw_id + UNKNOWN_OFFSET
    assert len(score_calls_for_marker(matcher, marker)) == 2, (
        "an aged-out raw id is a new Pending track, not a suppressed settled Unknown"
    )


def test_ur11_simultaneous_recovery_collision_keeps_higher_score_wins():
    """SPEC TDD #11 / safety invariants #1,#2. Two rechecks claiming the same
    Missing identity preserve the higher-score-wins rule; the loser settles
    Unknown with its new baseline and does not immediately retry."""
    fps = 4.0
    window = round(0.25 * fps)
    death_threshold = round(0.5 * fps)
    matcher = FakeMatcher()
    linker = recovery_linker(matcher, fps=fps, cooldown_s=0.0, evidence_frames=1)

    session_id = 1
    _enrol(linker, [{"tracker_id": 1601, "box": BOX_A, "bgr": (80, 80, 80)}], window=window)
    _kill(linker, death_threshold=death_threshold)

    occ_a, occ_b = (10, 10, 10), (12, 12, 12)
    clean_a, clean_b = (60, 60, 200), (200, 60, 60)
    m_occ_a, m_occ_b = rgb_marker_for_bgr(occ_a), rgb_marker_for_bgr(occ_b)
    m_clean_a, m_clean_b = rgb_marker_for_bgr(clean_a), rgb_marker_for_bgr(clean_b)
    matcher.program(m_occ_a, [({session_id: 0.1}, REJECT)])
    matcher.program(m_occ_b, [({session_id: 0.1}, REJECT)])
    # Both recover-claim identity 1; B scores higher and must win.
    matcher.program(m_clean_a, [({session_id: 0.6}, session_id)])
    matcher.program(m_clean_b, [({session_id: 0.95}, session_id)])
    raw_a, raw_b = 1701, 1702

    # Both settle Unknown first (rejected against the Missing identity).
    rows = [
        {"tracker_id": raw_a, "box": BOX_B, "bgr": occ_a},
        {"tracker_id": raw_b, "box": BOX_C, "bgr": occ_b},
    ]
    frame, dets = build_call(rows)
    linker.update(dets, frame)

    # Both change on the same tick -> both re-arm -> both recover-claim identity 1
    # in one batch. Higher scorer (B) wins; A settles Unknown.
    rows = [
        {"tracker_id": raw_a, "box": BOX_B, "bgr": clean_a},
        {"tracker_id": raw_b, "box": BOX_C, "bgr": clean_b},
    ]
    frame, dets = build_call(rows)
    out = linker.update(dets, frame)
    assert output_tracker_id(out, rows, raw_b) == session_id, "the higher recovery scorer wins the identity"
    assert output_tracker_id(out, rows, raw_a) == raw_a + UNKNOWN_OFFSET, "the collision loser settles Unknown"
    assert resolving_of(out, rows, raw_a) is False

    # The loser settled with a fresh baseline. On the next frame it does not
    # retry: identity 1 is now Active-present, so the loser has no candidate
    # target and `_observe_settled_unknown` returns before any fingerprint work.
    # (Baseline-advance-with-a-still-available-candidate is pinned separately by
    # test_ur5 and test_ur15; this only asserts the collision loser is quiescent.)
    calls_before = len(matcher.score_calls)
    rows = [{"tracker_id": raw_a, "box": BOX_B, "bgr": clean_a}]
    frame, dets = build_call(rows)
    out = linker.update(dets, frame)
    assert output_tracker_id(out, rows, raw_a) == raw_a + UNKNOWN_OFFSET
    assert len(matcher.score_calls) == calls_before, "the loser does not immediately retry"


def test_ur12_reset_clears_all_recovery_state():
    """SPEC TDD #12 / B-UR9. reset() clears settled Unknown/recovery state,
    Pending state, fingerprints, cooldowns, and candidate snapshots."""
    fps = 4.0
    window = round(0.25 * fps)
    death_threshold = round(0.5 * fps)
    matcher = FakeMatcher()
    linker = recovery_linker(matcher, fps=fps, cooldown_s=0.0, evidence_frames=1)

    session_id = 1
    _enrol(linker, [{"tracker_id": 1801, "box": BOX_A, "bgr": (80, 80, 80)}], window=window)
    _kill(linker, death_threshold=death_threshold)

    foreign_bgr = (44, 55, 66)
    marker = rgb_marker_for_bgr(foreign_bgr)
    matcher.program(marker, [({session_id: 0.1}, REJECT)])
    rows = [{"tracker_id": 1901, "box": BOX_B, "bgr": foreign_bgr}]
    frame, dets = build_call(rows)
    linker.update(dets, frame)
    assert linker._settled_unknown, "sanity: a settled Unknown recovery baseline now exists"

    linker.reset()
    assert linker._settled_unknown == {}, "reset() must drop every settled Unknown recovery baseline"
    assert linker._pending == {}, "reset() must drop every Pending track"
    assert linker.roster == frozenset()


def test_ur13_recovery_transitions_stay_row_aligned_with_correct_resolving():
    """SPEC TDD #13 / seam. Every transition (settled -> re-armed Pending ->
    linked) keeps all detection fields row-aligned and emits the correct
    resolving value, alongside an unrelated Active roster row."""
    fps = 4.0
    window = round(0.25 * fps)
    death_threshold = round(0.5 * fps)
    matcher = FakeMatcher()
    linker = recovery_linker(matcher, fps=fps, cooldown_s=0.0, evidence_frames=2)

    missing_id, active_id = 1, 2
    _enrol(
        linker,
        [
            {"tracker_id": 501, "box": BOX_A, "bgr": (80, 80, 80)},
            {"tracker_id": 502, "box": BOX_D, "bgr": (120, 120, 120)},
        ],
        window=window,
    )
    assert linker.roster == frozenset({missing_id, active_id})
    for _ in range(death_threshold + 1):  # kill identity 1, keep identity 2 present
        rows = [{"tracker_id": 502, "box": BOX_D, "bgr": (120, 120, 120)}]
        frame, dets = build_call(rows)
        linker.update(dets, frame)

    occluded_bgr = (10, 10, 10)
    clean_bgr = (60, 60, 200)
    occluded_marker = rgb_marker_for_bgr(occluded_bgr)
    clean_marker = rgb_marker_for_bgr(clean_bgr)
    matcher.program(occluded_marker, [({missing_id: 0.1, active_id: 0.1}, REJECT)])
    matcher.program(clean_marker, [({missing_id: 0.9, active_id: 0.1}, missing_id)])
    foreign_raw_id = 601

    def two_rows(bgr):
        return [
            {"tracker_id": 502, "box": BOX_D, "bgr": (120, 120, 120)},
            {"tracker_id": foreign_raw_id, "box": BOX_B, "bgr": bgr},
        ]

    def assert_row_aligned(out, dets, rows):
        assert len(out) == len(rows)
        assert np.array_equal(out.xyxy, dets.xyxy)
        assert np.array_equal(out.mask, dets.mask)
        assert np.array_equal(out.confidence, dets.confidence)
        assert np.array_equal(out.class_id, dets.class_id)
        assert "resolving" in out.data and len(out.data["resolving"]) == len(rows)

    # Occluded frame 1: foreign is Pending (window open) -> resolving True.
    rows = two_rows(occluded_bgr)
    frame, dets = build_call(rows)
    out = linker.update(dets, frame)
    assert_row_aligned(out, dets, rows)
    assert resolving_of(out, rows, foreign_raw_id) is True
    assert resolving_of(out, rows, 502) is False and output_tracker_id(out, rows, 502) == active_id

    # Occluded frame 2: window closes -> REJECT -> settled Unknown -> resolving False.
    rows = two_rows(occluded_bgr)
    frame, dets = build_call(rows)
    out = linker.update(dets, frame)
    assert_row_aligned(out, dets, rows)
    assert output_tracker_id(out, rows, foreign_raw_id) == foreign_raw_id + UNKNOWN_OFFSET
    assert resolving_of(out, rows, foreign_raw_id) is False

    # Clean frame 1: material change -> re-armed to a fresh Pending -> resolving True.
    rows = two_rows(clean_bgr)
    frame, dets = build_call(rows)
    out = linker.update(dets, frame)
    assert_row_aligned(out, dets, rows)
    assert output_tracker_id(out, rows, foreign_raw_id) == foreign_raw_id + UNKNOWN_OFFSET
    assert resolving_of(out, rows, foreign_raw_id) is True, "a re-armed track is resolving again (spinner)"

    # Clean frame 2: window closes -> linked -> original session id -> resolving False.
    rows = two_rows(clean_bgr)
    frame, dets = build_call(rows)
    out = linker.update(dets, frame)
    assert_row_aligned(out, dets, rows)
    assert output_tracker_id(out, rows, foreign_raw_id) == missing_id
    assert resolving_of(out, rows, foreign_raw_id) is False
    assert output_tracker_id(out, rows, 502) == active_id


def test_ur14_present_active_identity_is_not_a_candidate_regardless_of_row_order():
    """Regression (Codex review, invariant #10). A settled-Unknown row processed
    BEFORE a returning present Active row must NOT see that identity as a coasting
    candidate. The absence streak is reset in the age pass (not deferred to the
    identity's own row), so candidate classification is row-order independent — a
    static foreign object is never re-scored just because a legitimately-absent
    identity came back on the same frame."""
    fps = 4.0
    window = round(0.25 * fps)
    death_threshold = round(0.5 * fps)
    assert death_threshold == 2  # so identity 1 survives a single absent frame
    matcher = FakeMatcher()
    linker = recovery_linker(matcher, fps=fps, cooldown_s=0.0, evidence_frames=1)

    keeper_id, missing_id = 1, 2  # renumbered ascending: raw 501 -> 1, raw 502 -> 2
    _enrol(
        linker,
        [
            {"tracker_id": 501, "box": BOX_A, "bgr": (80, 80, 80)},
            {"tracker_id": 502, "box": BOX_D, "bgr": (120, 120, 120)},
        ],
        window=window,
    )
    assert linker.roster == frozenset({keeper_id, missing_id})

    # Kill identity 2 while identity 1 (raw 501) stays present -> identity 2 Missing.
    for _ in range(death_threshold + 1):
        rows = [{"tracker_id": 501, "box": BOX_A, "bgr": (80, 80, 80)}]
        frame, dets = build_call(rows)
        linker.update(dets, frame)

    foreign_bgr = (44, 55, 66)
    marker = rgb_marker_for_bgr(foreign_bgr)
    # Exactly ONE reject: a spurious re-arm would call score() a second time
    # against an empty queue and raise.
    matcher.program(marker, [({missing_id: 0.1}, REJECT)])
    foreign_raw_id = 601

    # Foreign settles Unknown while identity 1 is present (candidate set = {2}).
    rows = [
        {"tracker_id": 501, "box": BOX_A, "bgr": (80, 80, 80)},
        {"tracker_id": foreign_raw_id, "box": BOX_B, "bgr": foreign_bgr},
    ]
    frame, dets = build_call(rows)
    linker.update(dets, frame)
    assert len(score_calls_for_marker(matcher, marker)) == 1

    # One frame where identity 1 (raw 501) is absent -> it coasts (streak 1, still
    # alive), but the foreign is also absent this frame so it is not observed.
    frame, dets = empty_call()
    linker.update(dets, frame)

    # Identity 1 RETURNS present, on the SAME frame as the (unchanged) foreign,
    # with the foreign row placed FIRST. Its stale coasting streak must have been
    # cleared in the age pass, so it is NOT a candidate and the foreign does not
    # re-arm. A regression here would re-score the static foreign object.
    rows = [
        {"tracker_id": foreign_raw_id, "box": BOX_B, "bgr": foreign_bgr},
        {"tracker_id": 501, "box": BOX_A, "bgr": (80, 80, 80)},
    ]
    frame, dets = build_call(rows)
    out = linker.update(dets, frame)
    assert output_tracker_id(out, rows, foreign_raw_id) == foreign_raw_id + UNKNOWN_OFFSET
    assert resolving_of(out, rows, foreign_raw_id) is False
    assert len(score_calls_for_marker(matcher, marker)) == 1, (
        "a present Active identity must never be counted as a coasting candidate — no spurious re-score"
    )


def test_ur15_deferred_rejection_fingerprints_the_final_window_not_an_evicted_crop():
    """Regression (Codex review, B-UR6 / invariant #10). A track that is deferred,
    keeps collecting crops (evicting an early high-quality one from its bounded
    buffer), then rejects on revalidation must fingerprint the highest-quality
    crop STILL IN the final window — not the since-evicted historical best."""
    fps = 4.0
    death_threshold = round(0.75 * fps)
    assert death_threshold == 3
    matcher = FakeMatcher()
    linker = recovery_linker(
        matcher, fps=fps, cooldown_s=0.0, evidence_frames=2,
        enrolment_window_s=0.25, absent_death_s=0.75,
    )
    session_id = 1
    _enrol(linker, [{"tracker_id": 501, "box": BOX_A, "bgr": (80, 80, 80)}], window=round(0.25 * fps))
    assert linker.roster == frozenset({session_id})

    colorx = (60, 60, 200)
    marker = rgb_marker_for_bgr(colorx)
    # F3 accepts (-> deferred behind coasting identity 1); F5 rejects (-> settle).
    matcher.program(marker, [({session_id: 0.9}, session_id), ({session_id: 0.1}, REJECT)])
    r = 601
    box_area = (24 - 4) * (60 - 40)  # BOX_B = 20x20 = 400 px

    # F2: raw 501 absent -> identity 1 coasts (streak 1). R appears, HIGH quality.
    frame, dets = build_call([{"tracker_id": r, "box": BOX_B, "bgr": colorx, "confidence": 0.95}])
    linker.update(dets, frame)
    # F3: R's second crop closes the window; accepted for coasting identity 1 -> DEFERRED.
    frame, dets = build_call([{"tracker_id": r, "box": BOX_B, "bgr": colorx, "confidence": 0.6}])
    linker.update(dets, frame)
    assert len(score_calls_for_marker(matcher, marker)) == 1, "sanity: deferral scored once (accepted)"

    # F4, F5: lower-quality crops arrive while deferred; the bounded (maxlen=2)
    # buffer evicts the high-quality F2 crop. Identity 1 finally dies on F5.
    frame, dets = build_call([{"tracker_id": r, "box": BOX_B, "bgr": colorx, "confidence": 0.5}])
    linker.update(dets, frame)
    frame, dets = build_call([{"tracker_id": r, "box": BOX_B, "bgr": colorx, "confidence": 0.5}])
    out = linker.update(dets, frame)

    assert len(score_calls_for_marker(matcher, marker)) == 2, "the deferred window revalidated and rejected"
    assert output_tracker_id(out, rows=[{"tracker_id": r}], raw_id=r) == r + UNKNOWN_OFFSET
    assert r in linker._settled_unknown, "the deferred track settled Unknown after rejection"
    settled = linker._settled_unknown[r]
    # The final window held only conf=0.5 crops (q = 400*0.5 = 200). The evicted
    # F2 crop (q = 400*0.95 = 380) must NOT be the baseline.
    assert settled.decision_quality == box_area * 0.5, (
        "the settled baseline must come from the highest-quality crop in the FINAL window"
    )
    assert settled.decision_quality != box_area * 0.95, "the evicted historical-best crop must not leak into the baseline"


def test_ur16_quality_gain_alone_rearms_a_settled_unknown():
    """A quality gain at/above the gate re-arms even with an unchanged colour and
    mask — pinning the quality signal positively (SPEC § "Material-change rule")."""
    fps = 4.0
    window = round(0.25 * fps)
    death_threshold = round(0.5 * fps)
    matcher = FakeMatcher()
    linker = recovery_linker(
        matcher, fps=fps, cooldown_s=0.0, evidence_frames=1,
        appearance_delta=0.9,   # effectively disable the appearance signal
        mask_iou=0.0,           # effectively disable the mask signal
        quality_gain=0.25,      # only a >=25% quality gain may re-arm
    )
    session_id = 1
    _enrol(linker, [{"tracker_id": 701, "box": BOX_A, "bgr": (80, 80, 80)}], window=window)
    _kill(linker, death_threshold=death_threshold)

    bgr = (60, 60, 200)
    marker = rgb_marker_for_bgr(bgr)
    matcher.program(marker, [({session_id: 0.1}, REJECT), ({session_id: 0.9}, session_id)])
    raw_id = 702

    # Settle at low confidence (quality = 400 * 0.5 = 200).
    frame, dets = build_call([{"tracker_id": raw_id, "box": BOX_B, "bgr": bgr, "confidence": 0.5}])
    out = linker.update(dets, frame)
    assert output_tracker_id(out, [{"tracker_id": raw_id}], raw_id) == raw_id + UNKNOWN_OFFSET
    assert len(score_calls_for_marker(matcher, marker)) == 1

    # Same colour and box, but confidence 0.7 -> quality 280 -> +40% gain >= 25%.
    frame, dets = build_call([{"tracker_id": raw_id, "box": BOX_B, "bgr": bgr, "confidence": 0.7}])
    out = linker.update(dets, frame)
    assert len(score_calls_for_marker(matcher, marker)) == 2, "a >=25% quality gain must re-arm"
    assert output_tracker_id(out, [{"tracker_id": raw_id}], raw_id) == session_id


def test_ur17_rearm_emits_a_structured_orc_event(caplog):
    """B-UR10 observability. A re-arm logs one structured `orc` event carrying the
    raw id, trigger, measured metrics, old/new candidate ids, and frames since the
    previous decision — the payload the --debug console reads."""
    fps = 4.0
    window = round(0.25 * fps)
    death_threshold = round(0.5 * fps)
    matcher = FakeMatcher()
    linker = recovery_linker(matcher, fps=fps, cooldown_s=0.0, evidence_frames=1)

    session_id = 1
    _enrol(linker, [{"tracker_id": 801, "box": BOX_A, "bgr": (80, 80, 80)}], window=window)
    _kill(linker, death_threshold=death_threshold)

    occluded_bgr, clean_bgr = (10, 10, 10), (60, 60, 200)
    matcher.program(rgb_marker_for_bgr(occluded_bgr), [({session_id: 0.1}, REJECT)])
    matcher.program(rgb_marker_for_bgr(clean_bgr), [({session_id: 0.9}, session_id)])
    raw_id = 802

    frame, dets = build_call([{"tracker_id": raw_id, "box": BOX_B, "bgr": occluded_bgr}])
    linker.update(dets, frame)

    with caplog.at_level(logging.INFO):
        frame, dets = build_call([{"tracker_id": raw_id, "box": BOX_B, "bgr": clean_bgr}])
        linker.update(dets, frame)

    rearm = next(
        (getattr(r, "orc", None) for r in caplog.records
         if getattr(r, "orc", None) and r.orc.get("event") == "unknown_rearm"),
        None,
    )
    assert rearm is not None, "a re-arm must log a structured orc={'event':'unknown_rearm',...}"
    assert rearm["raw_id"] == raw_id
    assert rearm["trigger"] == "appearance"
    assert rearm["appearance_delta"] is not None and rearm["appearance_delta"] >= 0.15
    # All three measured signals are present in the payload even though appearance
    # is what tripped the gate (same box/confidence -> mask IoU 1.0, quality gain 0).
    assert rearm["mask_iou"] == 1.0
    assert rearm["quality_gain"] == 0.0
    assert rearm["old_candidates"] == [session_id] and rearm["new_candidates"] == [session_id]
    assert isinstance(rearm["frames_since_decision"], int) and rearm["frames_since_decision"] >= 1


# ==========================================================================
# T11 SPEC-M — catalog-only enrolment (safe setup)
#
# Contract: the SPEC-M-phase1-model spec for ticket T11
# (docs/wayfinder/session-linker/tickets/T11-safe-setup-enrolment.md, M1-M4).
# In catalog-only mode ONLY a raw track that confidently binds to a loaded
# persistent specimen may join the roster; every other majority-present track
# (foreign object, below-threshold, contested loser) stays in the offset
# (Unknown) range and is NEVER renumbered into a session-only identity. The
# `catalog` is the fixed set of loaded specimen ids, known from construction
# and constant across reset(); in this mode `roster` is always a subset of it.
# Legacy (catalog_only_enrolment=False) behaviour is preserved unchanged.
# ==========================================================================


def catalog_linker(matcher, galleries, *, bind_tau, bind_margin, fps=4.0, **kwargs):
    """A `bind_linker` (freeze on the first call) run in catalog-only mode."""
    return bind_linker(
        matcher,
        galleries,
        bind_tau=bind_tau,
        bind_margin=bind_margin,
        fps=fps,
        catalog_only_enrolment=True,
        **kwargs,
    )


def _grid_boxes(n):
    """`n` non-overlapping quality boxes on the 64x64 frame (14x14 = 196 px^2,
    inset from every edge), laid out on a 3-wide grid. Each keeps clear of the
    frame edge so build_call's quality gate accepts it."""
    boxes = []
    for k in range(n):
        col, row = k % 3, k // 3
        x0 = 4 + 20 * col
        y0 = 4 + 20 * row
        boxes.append((x0, y0, x0 + 14, y0 + 14))
    return boxes


def _distinct_bgr(k):
    """A distinct, non-background paint per crop so each row carries its own
    FakeMatcher marker (see FakeMatcher.score)."""
    return (5 + 2 * k, 40 + 3 * k, 90 + 5 * k)


def _run_catalog_only_freeze(
    n_known, *, include_foreign=True, bind_tau=0.5, bind_margin=0.1, fps=4.0
):
    """Drive one catalog-only enrolment freeze with `n_known` crops each
    programmed to confidently bind specimen 1..n_known, plus (optionally) one
    foreign crop scored below `bind_tau`. Returns
    (linker, matcher, out, rows, known_raw_ids, foreign_raw)."""
    matcher = FakeMatcher()
    galleries = {s: persistent_gallery(s) for s in range(1, n_known + 1)}
    n_rows = n_known + (1 if include_foreign else 0)
    boxes = _grid_boxes(n_rows)
    rows = []
    known_raw_ids = []
    for i in range(n_known):
        specimen = i + 1
        bgr = _distinct_bgr(i)
        # A clean best on this crop's own specimen, everything else far below.
        scores = {s: (0.9 if s == specimen else 0.05) for s in range(1, n_known + 1)}
        matcher.program(rgb_marker_for_bgr(bgr), [(scores, None)])
        raw_id = 201 + i
        known_raw_ids.append(raw_id)
        rows.append({"tracker_id": raw_id, "box": boxes[i], "bgr": bgr})
    foreign_raw = None
    if include_foreign:
        foreign_raw = 201 + n_known
        bgr = _distinct_bgr(n_known)
        below = {s: 0.1 for s in range(1, n_known + 1)}  # best < bind_tau -> unbound
        matcher.program(rgb_marker_for_bgr(bgr), [(below, None)])
        rows.append({"tracker_id": foreign_raw, "box": boxes[n_known], "bgr": bgr})
    linker = catalog_linker(
        matcher, galleries, bind_tau=bind_tau, bind_margin=bind_margin, fps=fps
    )
    frame, dets = build_call(rows)
    out = linker.update(dets, frame)
    return linker, matcher, out, rows, known_raw_ids, foreign_raw


# SPEC-M test 1 — catalog constant + survives reset (SessionLinker half).
def test_catalog_is_constant_before_freeze_after_freeze_and_after_reset():
    matcher = FakeMatcher()
    bgr = (10, 20, 30)
    matcher.program(rgb_marker_for_bgr(bgr), [({1: 0.9, 2: 0.05}, None)])
    galleries = {1: persistent_gallery(1), 2: persistent_gallery(2)}
    linker = catalog_linker(matcher, galleries, bind_tau=0.5, bind_margin=0.1)

    catalog = frozenset({1, 2})
    assert linker.catalog == catalog, "catalog is known from construction, before any enrolment freeze"
    assert isinstance(linker.catalog, frozenset)
    assert linker.roster == frozenset(), "roster is empty until the freeze; catalog is not"

    raw_id = 301
    rows = [{"tracker_id": raw_id, "box": BOX_A, "bgr": bgr}]
    frame, dets = build_call(rows)
    linker.update(dets, frame)  # freezes on the first call
    assert linker.roster == frozenset({1}), "sanity: the freeze bound specimen 1"
    assert linker.catalog == catalog, "the enrolment freeze must not change the catalog"

    linker.reset()
    assert linker.catalog == catalog, "reset() preserves the catalog"
    assert linker.roster == frozenset(), "reset() empties the roster"


# SPEC-M test 2 — catalog-only freeze admits a confidently bound known specimen.
def test_catalog_only_freeze_admits_a_confidently_bound_specimen():
    matcher = FakeMatcher()
    bgr = (10, 20, 30)
    # >= bind_tau; with a single gallery the margin is vacuous (cf. B-B3).
    matcher.program(rgb_marker_for_bgr(bgr), [({1: 0.9}, None)])
    linker = catalog_linker(matcher, {1: persistent_gallery(1)}, bind_tau=0.5, bind_margin=0.1)

    raw_id = 401
    rows = [{"tracker_id": raw_id, "box": BOX_A, "bgr": bgr}]
    frame, dets = build_call(rows)
    out = linker.update(dets, frame)

    assert linker.roster == frozenset({1}), "a confidently bound known specimen joins the catalog-only roster"
    assert 1 in linker.catalog
    assert output_tracker_id(out, rows, raw_id) == 1, "the bound row emits its specimen id on the freeze frame"
    assert 1 in linker.roster
    assert not resolving_of(out, rows, raw_id), "a bound roster id is settled on the freeze frame"


# SPEC-M test 3 — below-threshold setup object emits offset + absent from roster.
def test_catalog_only_below_threshold_object_emits_offset_and_is_absent_from_roster():
    matcher = FakeMatcher()
    bgr = (10, 20, 30)
    matcher.program(rgb_marker_for_bgr(bgr), [({1: 0.1}, None)])  # < bind_tau=0.5 -> never binds
    linker = catalog_linker(matcher, {1: persistent_gallery(1)}, bind_tau=0.5, bind_margin=0.1)

    raw_id = 501
    rows = [{"tracker_id": raw_id, "box": BOX_A, "bgr": bgr}]
    frame, dets = build_call(rows)
    out = linker.update(dets, frame)

    assert linker.roster == frozenset(), "a below-threshold setup object never joins the roster"
    assert output_tracker_id(out, rows, raw_id) == raw_id + UNKNOWN_OFFSET, (
        "an unbound setup track emits its offset (Unknown) id, never a session-only identity"
    )
    assert resolving_of(out, rows, raw_id), "an offset id on the freeze frame is still resolving"
    assert (raw_id + UNKNOWN_OFFSET) not in linker.roster


# SPEC-M test 4 — contested binding loser is absent from roster, never takes second choice.
def test_catalog_only_contested_binding_loser_is_absent_from_roster():
    matcher = FakeMatcher()
    galleries = {1: persistent_gallery(1), 2: persistent_gallery(2)}
    bgr_winner, bgr_loser = (10, 20, 30), (40, 50, 60)
    # Both best-match specimen 1; the higher score wins the one-to-one contest.
    matcher.program(rgb_marker_for_bgr(bgr_winner), [({1: 0.9, 2: 0.05}, None)])
    matcher.program(rgb_marker_for_bgr(bgr_loser), [({1: 0.8, 2: 0.04}, None)])
    linker = catalog_linker(matcher, galleries, bind_tau=0.5, bind_margin=0.1)

    raw_winner, raw_loser = 601, 602
    rows = [
        {"tracker_id": raw_winner, "box": BOX_A, "bgr": bgr_winner},
        {"tracker_id": raw_loser, "box": BOX_B, "bgr": bgr_loser},
    ]
    frame, dets = build_call(rows)
    out = linker.update(dets, frame)

    assert linker.roster == frozenset({1}), "only the contest winner binds; the loser does not join"
    assert 2 not in linker.roster, "the loser is never pushed onto its second-choice specimen"
    assert output_tracker_id(out, rows, raw_winner) == 1
    assert output_tracker_id(out, rows, raw_loser) == raw_loser + UNKNOWN_OFFSET, (
        "a contest loser stays Unknown (offset), never renumbered into a session-only identity"
    )
    assert not resolving_of(out, rows, raw_winner)
    assert resolving_of(out, rows, raw_loser)
    assert linker.roster <= linker.catalog


# SPEC-M test 5 — eight known + rejected foreign -> roster == catalog, one non-roster detection.
def test_catalog_only_eight_known_plus_foreign_roster_equals_catalog():
    linker, _matcher, out, rows, known_raw_ids, foreign_raw = _run_catalog_only_freeze(8)
    catalog = frozenset(range(1, 9))

    assert linker.catalog == catalog
    assert linker.roster == catalog, "every confidently bound specimen joins; nothing else does"
    assert linker.roster == linker.catalog

    emitted = {output_tracker_id(out, rows, r["tracker_id"]) for r in rows}
    assert emitted == catalog | {foreign_raw + UNKNOWN_OFFSET}, (
        "emitted ids are the eight specimen numbers plus the single foreign offset id"
    )
    for i, raw_id in enumerate(known_raw_ids):
        assert output_tracker_id(out, rows, raw_id) == i + 1
        assert not resolving_of(out, rows, raw_id), "a bound specimen is settled on the freeze frame"
    assert output_tracker_id(out, rows, foreign_raw) == foreign_raw + UNKNOWN_OFFSET
    assert resolving_of(out, rows, foreign_raw), "only the foreign row is still resolving"
    assert (foreign_raw + UNKNOWN_OFFSET) not in linker.roster


# SPEC-M test 6 — catalog-only mode never creates a session-only roster id.
def test_catalog_only_never_creates_a_session_only_roster_id():
    matcher = FakeMatcher()
    galleries = {1: persistent_gallery(1), 2: persistent_gallery(2)}
    bgr_bound, bgr_unbound = (10, 20, 30), (40, 50, 60)
    matcher.program(rgb_marker_for_bgr(bgr_bound), [({1: 0.9, 2: 0.05}, None)])   # binds specimen 1
    matcher.program(rgb_marker_for_bgr(bgr_unbound), [({1: 0.2, 2: 0.1}, None)])  # < tau -> unbound
    linker = catalog_linker(matcher, galleries, bind_tau=0.5, bind_margin=0.1)

    raw_bound, raw_unbound = 701, 702
    rows = [
        {"tracker_id": raw_bound, "box": BOX_A, "bgr": bgr_bound},
        {"tracker_id": raw_unbound, "box": BOX_B, "bgr": bgr_unbound},
    ]
    frame, dets = build_call(rows)
    out = linker.update(dets, frame)

    # Legacy mode WOULD have renumbered the unbound majority track to base + 1
    # where base == max(catalog); that session-only number must never appear.
    first_session_only_id = max(linker.catalog) + 1  # == 3 for catalog {1, 2}
    assert linker.roster == frozenset({1})
    assert linker.roster <= linker.catalog, "the catalog-only roster is always a subset of the catalog"
    assert all(sid < first_session_only_id for sid in linker.roster), (
        "no roster id may reach the first session-only number legacy mode would have assigned"
    )
    assert first_session_only_id not in linker.roster
    assert output_tracker_id(out, rows, raw_unbound) == raw_unbound + UNKNOWN_OFFSET


# SPEC-M test 7 — legacy non-catalog mode retains session-only behaviour.
def test_legacy_mode_unbound_majority_track_still_gets_a_session_only_roster_id():
    matcher = FakeMatcher()
    galleries = {1: persistent_gallery(1), 2: persistent_gallery(2)}
    bgr_bound, bgr_unbound = (10, 20, 30), (40, 50, 60)
    matcher.program(rgb_marker_for_bgr(bgr_bound), [({1: 0.9, 2: 0.05}, None)])
    matcher.program(rgb_marker_for_bgr(bgr_unbound), [({1: 0.2, 2: 0.1}, None)])
    # Default catalog_only_enrolment=False: the pre-T11 behaviour must survive.
    linker = bind_linker(matcher, galleries, bind_tau=0.5, bind_margin=0.1)

    raw_bound, raw_unbound = 701, 702
    rows = [
        {"tracker_id": raw_bound, "box": BOX_A, "bgr": bgr_bound},
        {"tracker_id": raw_unbound, "box": BOX_B, "bgr": bgr_unbound},
    ]
    frame, dets = build_call(rows)
    out = linker.update(dets, frame)

    # base = max({1, 2}) = 2 -> the unbound majority track renumbers to 3 and JOINS.
    session_only_id = 3
    assert linker.roster == frozenset({1, session_only_id}), (
        "legacy mode still admits an unbound majority-present track as a session-only identity"
    )
    assert output_tracker_id(out, rows, raw_bound) == 1
    assert output_tracker_id(out, rows, raw_unbound) == session_only_id, (
        "in legacy mode the unbound track wears its session-only id, not an offset id"
    )


# SPEC-M test 9 — freeze-frame ids, roster, resolving, catalog same-tick consistent.
def test_catalog_only_freeze_frame_is_same_tick_consistent():
    # Read emitted ids, roster, resolving and catalog from the SAME update()
    # return and assert they are mutually consistent for that one tick.
    linker, _matcher, out, rows, _known_raw_ids, _foreign_raw = _run_catalog_only_freeze(3)
    roster = linker.roster
    catalog = linker.catalog

    assert roster == frozenset({1, 2, 3})
    assert roster <= catalog
    for row in rows:
        raw_id = row["tracker_id"]
        emitted = output_tracker_id(out, rows, raw_id)
        resolving = resolving_of(out, rows, raw_id)
        if emitted in roster:
            assert not resolving, "an in-roster row has settled -> resolving False"
            assert emitted in catalog, "an in-roster id is always a catalog specimen"
        else:
            assert resolving, "an offset (not-in-roster) row is still resolving on the freeze frame"
            assert emitted not in roster
