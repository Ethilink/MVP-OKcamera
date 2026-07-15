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
"""

from __future__ import annotations

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
    def __init__(self, views: np.ndarray):
        self.views = views


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


# --------------------------------------------------------------------------
# B1 — Emission contract
# --------------------------------------------------------------------------


def test_b1_2_passthrough_before_and_during_enrolment():
    fps = 4.0
    enrolment_window_s = 0.75
    window = round(enrolment_window_s * fps)  # frames the enrolment window spans
    matcher = FakeMatcher()
    linker = SessionLinker(
        matcher,
        fps=fps,
        enrolment_window_s=enrolment_window_s,
        evidence_window_s=10.0,
        evidence_frames=1,
        absent_death_s=10.0,
        min_mask_area_px=100,
    )
    raw_id = 111
    for i in range(window):
        rows = [{"tracker_id": raw_id, "box": BOX_A, "bgr": (10, 10, 10)}]
        frame, dets = build_call(rows)
        out = linker.update(dets, frame)
        assert output_tracker_id(out, rows, raw_id) == raw_id, (
            f"call {i + 1}/{window}: enrolment window still open, expected pure pass-through"
        )
        if i < window - 1:
            assert linker.roster == frozenset(), "roster must stay empty until the window closes"
    assert linker.roster == frozenset({raw_id}), "window closed on the last call; sole id had 100% presence"


def test_b1_3_empty_frames_advance_the_enrolment_window_clock():
    fps = 4.0
    enrolment_window_s = 0.75
    window = round(enrolment_window_s * fps)  # chosen so window == 3: 2-of-3 presences is a majority
    matcher = FakeMatcher()
    linker = SessionLinker(
        matcher,
        fps=fps,
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
    assert linker.roster == frozenset({raw_id})


def test_b1_1_output_is_new_object_and_input_is_never_mutated():
    fps = 4.0
    enrolment_window_s = 0.25
    window = round(enrolment_window_s * fps)
    matcher = FakeMatcher()
    linker = SessionLinker(
        matcher,
        fps=fps,
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
        enrolment_window_s=enrolment_window_s,
        evidence_window_s=10.0,
        evidence_frames=2,  # two quality crops needed, so we can observe Pending vs decided
        absent_death_s=absent_death_s,
        min_mask_area_px=100,
    )
    roster_id = 601
    for _ in range(window):
        rows = [{"tracker_id": roster_id, "box": BOX_A, "bgr": (80, 80, 80)}]
        frame, dets = build_call(rows)
        linker.update(dets, frame)

    for _ in range(death_threshold + 1):  # "more than" the threshold -> dies
        frame, dets = empty_call()
        linker.update(dets, frame)

    return_bgr = (11, 22, 33)
    marker = rgb_marker_for_bgr(return_bgr)
    matcher.program(marker, [({roster_id: 0.9}, roster_id)])
    return_raw_id = 602

    # First quality appearance: evidence window not yet closed -> Pending
    # -> pure pass-through, no decision, no score() call yet.
    rows = [{"tracker_id": return_raw_id, "box": BOX_B, "bgr": return_bgr}]
    frame, dets = build_call(rows)
    out = linker.update(dets, frame)
    assert output_tracker_id(out, rows, return_raw_id) == return_raw_id
    assert not matcher.score_calls

    # Second quality appearance: the evidence window closes THIS call, the
    # batched decision resolves synchronously, and THIS call's own output
    # already carries the original roster id, not the disposable raw id.
    rows = [{"tracker_id": return_raw_id, "box": BOX_B, "bgr": return_bgr}]
    frame, dets = build_call(rows)
    out = linker.update(dets, frame)
    assert output_tracker_id(out, rows, return_raw_id) == roster_id

    # The mapping persists for the rest of the recording.
    rows = [{"tracker_id": return_raw_id, "box": BOX_B, "bgr": return_bgr}]
    frame, dets = build_call(rows)
    out = linker.update(dets, frame)
    assert output_tracker_id(out, rows, return_raw_id) == roster_id


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

    assert linker.roster == frozenset({id_full_presence, id_late_majority})


def test_b2_roster_property_is_frozenset_and_empty_before_freeze():
    fps = 4.0
    enrolment_window_s = 0.25
    window = round(enrolment_window_s * fps)  # freezes on the very first call
    matcher = FakeMatcher()
    linker = SessionLinker(
        matcher,
        fps=fps,
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

    assert linker.roster == frozenset({raw_id})
    assert isinstance(linker.roster, frozenset)


def test_b2_post_freeze_newcomer_passes_through_its_own_raw_id():
    fps = 4.0
    enrolment_window_s = 0.25
    window = round(enrolment_window_s * fps)
    matcher = FakeMatcher()
    linker = SessionLinker(
        matcher,
        fps=fps,
        enrolment_window_s=enrolment_window_s,
        evidence_window_s=100.0,
        evidence_frames=5,  # kept Pending for the whole test, never decided
        absent_death_s=100.0,
        min_mask_area_px=100,
    )
    roster_id = 231
    for _ in range(window):
        rows = [{"tracker_id": roster_id, "box": BOX_A, "bgr": (50, 50, 50)}]
        frame, dets = build_call(rows)
        linker.update(dets, frame)
    assert linker.roster == frozenset({roster_id})

    newcomer_id = 331  # born strictly after the freeze -> never joins the roster
    for _ in range(3):
        rows = [
            {"tracker_id": roster_id, "box": BOX_A, "bgr": (50, 50, 50)},
            {"tracker_id": newcomer_id, "box": BOX_B, "bgr": (60, 60, 60)},
        ]
        frame, dets = build_call(rows)
        out = linker.update(dets, frame)
        assert output_tracker_id(out, rows, newcomer_id) == newcomer_id
    assert linker.roster == frozenset({roster_id}), "a post-freeze newcomer must never join the frozen roster"


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
        enrolment_window_s=enrolment_window_s,
        evidence_window_s=10.0,
        evidence_frames=1,
        absent_death_s=absent_death_s,
        min_mask_area_px=100,
    )
    roster_id = 401
    for _ in range(window):
        rows = [{"tracker_id": roster_id, "box": BOX_A, "bgr": (70, 70, 70)}]
        frame, dets = build_call(rows)
        linker.update(dets, frame)
    assert linker.roster == frozenset({roster_id})

    for _ in range(death_threshold + 1):  # "more than" the threshold -> dies
        frame, dets = empty_call()
        linker.update(dets, frame)

    return_bgr = (5, 15, 200)
    marker = rgb_marker_for_bgr(return_bgr)
    matcher.program(marker, [({roster_id: 0.9}, roster_id)])
    return_raw_id = 402
    rows = [{"tracker_id": return_raw_id, "box": BOX_B, "bgr": return_bgr}]
    frame, dets = build_call(rows)
    linker.update(dets, frame)

    matching_calls = [c for c in matcher.score_calls if c["marker"] == marker]
    assert matching_calls, "expected the return track's evidence window to close and trigger a scoring call"
    assert roster_id in matching_calls[0]["gallery_keys"], (
        "roster_id must be an admissible scoring candidate — proves it moved Active -> Missing after the "
        "absence threshold was crossed"
    )


def test_b3_raw_id_reactivation_after_death_must_pass_the_matcher_again():
    fps = 4.0
    absent_death_s = 0.5
    death_threshold = round(absent_death_s * fps)
    matcher = FakeMatcher()
    linker = SessionLinker(
        matcher,
        fps=fps,
        enrolment_window_s=0.25,
        evidence_window_s=10.0,
        evidence_frames=1,
        absent_death_s=absent_death_s,
        min_mask_area_px=100,
    )
    roster_id = 451
    rows = [{"tracker_id": roster_id, "box": BOX_A, "bgr": (70, 70, 70)}]
    frame, dets = build_call(rows)
    linker.update(dets, frame)

    for _ in range(death_threshold + 1):
        frame, dets = empty_call()
        linker.update(dets, frame)

    reactivated_bgr = (5, 15, 200)
    marker = rgb_marker_for_bgr(reactivated_bgr)
    matcher.program(marker, [({roster_id: 0.1}, REJECT)])
    rows = [{"tracker_id": roster_id, "box": BOX_B, "bgr": reactivated_bgr}]
    frame, dets = build_call(rows)
    linker.update(dets, frame)

    assert [call for call in matcher.score_calls if call["marker"] == marker], (
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
        enrolment_window_s=enrolment_window_s,
        evidence_window_s=10.0,
        evidence_frames=1,
        absent_death_s=100.0,  # nothing ever dies in this test
        min_mask_area_px=100,
    )
    roster_id = 821
    for _ in range(window):
        rows = [{"tracker_id": roster_id, "box": BOX_A, "bgr": (90, 90, 90)}]
        frame, dets = build_call(rows)
        linker.update(dets, frame)
    assert linker.roster == frozenset({roster_id})

    newcomer_id = 901
    rows = [
        {"tracker_id": roster_id, "box": BOX_A, "bgr": (90, 90, 90)},
        {"tracker_id": newcomer_id, "box": BOX_B, "bgr": (95, 95, 95)},
    ]
    frame, dets = build_call(rows)
    out = linker.update(dets, frame)  # evidence_frames=1 -> window closes now; missing is empty

    assert not matcher.score_calls, "no Missing identities exist; must settle Unknown without any matcher call"
    assert output_tracker_id(out, rows, newcomer_id) == newcomer_id

    rows2 = [{"tracker_id": newcomer_id, "box": BOX_B, "bgr": (95, 95, 95)}]
    frame2, dets2 = build_call(rows2)
    out2 = linker.update(dets2, frame2)
    assert output_tracker_id(out2, rows2, newcomer_id) == newcomer_id
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
        enrolment_window_s=0.25,
        evidence_window_s=10.0,
        evidence_frames=1,
        absent_death_s=absent_death_s,
        min_mask_area_px=100,
    )
    missing_id, active_id = 1101, 1102
    rows = [
        {"tracker_id": missing_id, "box": BOX_A, "bgr": (11, 11, 11)},
        {"tracker_id": active_id, "box": BOX_B, "bgr": (22, 22, 22)},
    ]
    frame, dets = build_call(rows)
    linker.update(dets, frame)

    for _ in range(death_threshold + 1):
        rows = [{"tracker_id": active_id, "box": BOX_B, "bgr": (22, 22, 22)}]
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
        enrolment_window_s=0.25,
        evidence_window_s=10.0,
        evidence_frames=1,
        absent_death_s=absent_death_s,
        min_mask_area_px=100,
    )
    missing_id, active_id = 1301, 1302
    rows = [
        {"tracker_id": missing_id, "box": BOX_A, "bgr": (33, 33, 33)},
        {"tracker_id": active_id, "box": BOX_B, "bgr": (44, 44, 44)},
    ]
    frame, dets = build_call(rows)
    linker.update(dets, frame)

    for _ in range(death_threshold + 1):
        rows = [{"tracker_id": active_id, "box": BOX_B, "bgr": (44, 44, 44)}]
        frame, dets = build_call(rows)
        linker.update(dets, frame)

    foreign_bgr = (3, 4, 220)
    marker = rgb_marker_for_bgr(foreign_bgr)
    matcher.program(marker, [({missing_id: 0.4, active_id: 0.95}, active_id)])
    raw_id = 1401
    rows = [{"tracker_id": raw_id, "box": BOX_C, "bgr": foreign_bgr}]
    frame, dets = build_call(rows)
    out = linker.update(dets, frame)

    assert output_tracker_id(out, rows, raw_id) == raw_id, (
        "an active identity may provide comparison evidence but must never be an eligible link target"
    )


def test_b5_coasting_identity_handoff_waits_for_missing_then_revalidates():
    fps = 4.0
    absent_death_s = 0.5
    matcher = FakeMatcher()
    linker = SessionLinker(
        matcher,
        fps=fps,
        enrolment_window_s=0.25,
        evidence_window_s=10.0,
        evidence_frames=1,
        absent_death_s=absent_death_s,
        min_mask_area_px=100,
    )
    roster_id = 1501
    rows = [{"tracker_id": roster_id, "box": BOX_A, "bgr": (33, 33, 33)}]
    frame, dets = build_call(rows)
    linker.update(dets, frame)

    # The old raw ID has been absent for one frame but has not crossed the
    # strict death threshold yet.
    frame, dets = empty_call()
    linker.update(dets, frame)

    return_bgr = (9, 19, 209)
    marker = rgb_marker_for_bgr(return_bgr)
    matcher.program(
        marker,
        [
            ({roster_id: 0.95}, roster_id),
            ({roster_id: 0.95}, roster_id),
        ],
    )
    return_raw_id = 1601
    rows = [{"tracker_id": return_raw_id, "box": BOX_B, "bgr": return_bgr}]

    # At exactly the death threshold the accepted roster identity is still
    # Active but its old raw ID is coasting off-screen. Do not force-link and
    # do not permanently settle this likely tracker handoff as Unknown.
    frame, dets = build_call(rows)
    provisional = linker.update(dets, frame)
    assert output_tracker_id(provisional, rows, return_raw_id) == return_raw_id

    # One frame later the old raw ID becomes Missing. The deferred row is
    # scored again against the same complete roster and can now link safely.
    frame, dets = build_call(rows)
    linked = linker.update(dets, frame)
    assert output_tracker_id(linked, rows, return_raw_id) == roster_id

    matching_calls = [call for call in matcher.score_calls if call["marker"] == marker]
    assert len(matching_calls) == 2
    assert all(call["gallery_keys"] == frozenset({roster_id}) for call in matching_calls)


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
        enrolment_window_s=enrolment_window_s,
        evidence_window_s=10.0,
        evidence_frames=1,
        absent_death_s=absent_death_s,
        min_mask_area_px=100,
    )
    id_a, id_b = 1201, 1202
    for _ in range(window):
        rows = [
            {"tracker_id": id_a, "box": BOX_A, "bgr": (11, 11, 11)},
            {"tracker_id": id_b, "box": BOX_B, "bgr": (22, 22, 22)},
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
    assert output_tracker_id(out, rows, raw_loser) == raw_loser, (
        "a collision loser must stay Unknown instead of falling through to a less likely identity"
    )

    loser_calls = [c for c in matcher.score_calls if c["marker"] == marker_loser]
    assert len(loser_calls) == 1
    assert loser_calls[0]["gallery_keys"] == frozenset({id_a, id_b})

    winner_calls = [c for c in matcher.score_calls if c["marker"] == marker_winner]
    assert len(winner_calls) == 1


def test_b5_5_reject_stays_unknown_permanently():
    fps = 4.0
    enrolment_window_s = 0.25
    window = round(enrolment_window_s * fps)
    absent_death_s = 0.5
    death_threshold = round(absent_death_s * fps)
    matcher = FakeMatcher()
    linker = SessionLinker(
        matcher,
        fps=fps,
        enrolment_window_s=enrolment_window_s,
        evidence_window_s=10.0,
        evidence_frames=1,
        absent_death_s=absent_death_s,
        min_mask_area_px=100,
    )
    roster_id = 1001
    for _ in range(window):
        rows = [{"tracker_id": roster_id, "box": BOX_A, "bgr": (100, 100, 100)}]
        frame, dets = build_call(rows)
        linker.update(dets, frame)
    for _ in range(death_threshold + 1):
        frame, dets = empty_call()
        linker.update(dets, frame)

    reject_bgr = (44, 55, 66)
    marker = rgb_marker_for_bgr(reject_bgr)
    matcher.program(marker, [({roster_id: 0.5}, REJECT)])
    unknown_raw_id = 1101
    rows = [{"tracker_id": unknown_raw_id, "box": BOX_B, "bgr": reject_bgr}]
    frame, dets = build_call(rows)
    out = linker.update(dets, frame)
    assert output_tracker_id(out, rows, unknown_raw_id) == unknown_raw_id

    calls_so_far = len([c for c in matcher.score_calls if c["marker"] == marker])
    assert calls_so_far == 1, "a rejected row is scored once against the stable comparison dictionary"

    # Reappearance after settling: never re-decided, permanent pass-through.
    rows2 = [{"tracker_id": unknown_raw_id, "box": BOX_B, "bgr": reject_bgr}]
    frame2, dets2 = build_call(rows2)
    out2 = linker.update(dets2, frame2)
    assert output_tracker_id(out2, rows2, unknown_raw_id) == unknown_raw_id
    assert len([c for c in matcher.score_calls if c["marker"] == marker]) == calls_so_far, (
        "a settled-Unknown raw id must never trigger another matcher call"
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
        enrolment_window_s=enrolment_window_s,
        evidence_window_s=10.0,
        evidence_frames=1,
        absent_death_s=absent_death_s,
        min_mask_area_px=100,
    )
    roster_id = 1401
    for _ in range(window):
        rows = [{"tracker_id": roster_id, "box": BOX_A, "bgr": (150, 150, 150)}]
        frame, dets = build_call(rows)
        linker.update(dets, frame)
    for _ in range(death_threshold + 1):
        frame, dets = empty_call()
        linker.update(dets, frame)

    bgr = (77, 88, 99)
    marker = rgb_marker_for_bgr(bgr)
    # A very high score must not matter: accept() explicitly rejects both rounds.
    matcher.program(marker, [({roster_id: 0.99}, REJECT), ({roster_id: 0.98}, REJECT)])
    raw_id = 1501
    rows = [{"tracker_id": raw_id, "box": BOX_B, "bgr": bgr}]
    frame, dets = build_call(rows)
    out = linker.update(dets, frame)

    assert output_tracker_id(out, rows, raw_id) == raw_id, (
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
        enrolment_window_s=enrolment_window_s,
        evidence_window_s=10.0,
        evidence_frames=1,
        absent_death_s=10.0,
        min_mask_area_px=100,
    )
    roster_id = 2001
    painted_bgr = (10, 20, 30)  # B=10, G=20, R=30
    for _ in range(window):
        rows = [{"tracker_id": roster_id, "box": BOX_A, "bgr": painted_bgr}]
        frame, dets = build_call(rows)
        linker.update(dets, frame)
    assert linker.roster == frozenset({roster_id})

    matching_calls = [c for c in matcher.build_gallery_calls if c["meta"].get("identity") == roster_id]
    assert matching_calls, "expected a Start-gallery build_gallery() call for the frozen roster identity"
    crops = matching_calls[0]["crops"]
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
        enrolment_window_s=enrolment_window_s,
        evidence_window_s=10.0,
        evidence_frames=1,
        absent_death_s=absent_death_s,
        min_mask_area_px=100,
    )
    roster_id = 2201
    for _ in range(window):
        rows = [{"tracker_id": roster_id, "box": BOX_A, "bgr": (33, 44, 55)}]
        frame, dets = build_call(rows)
        linker.update(dets, frame)
    for _ in range(death_threshold + 1):
        frame, dets = empty_call()
        linker.update(dets, frame)

    return_bgr = (66, 77, 88)
    marker = rgb_marker_for_bgr(return_bgr)
    matcher.program(marker, [({roster_id: 0.9}, roster_id)])
    return_raw_id = 2301
    rows = [{"tracker_id": return_raw_id, "box": BOX_B, "bgr": return_bgr}]
    frame, dets = build_call(rows)
    out = linker.update(dets, frame)
    assert output_tracker_id(out, rows, return_raw_id) == roster_id  # sanity: link happened pre-reset
    assert linker.roster == frozenset({roster_id})

    linker.reset()
    assert linker.roster == frozenset()

    # A fresh non-empty call after reset() opens a brand-new enrolment
    # window from scratch — the old raw_id -> session_id map is gone, so the
    # SAME raw id that used to resolve to roster_id now simply enrols itself.
    rows2 = [{"tracker_id": return_raw_id, "box": BOX_B, "bgr": return_bgr}]
    frame2, dets2 = build_call(rows2)
    out2 = linker.update(dets2, frame2)
    assert output_tracker_id(out2, rows2, return_raw_id) == return_raw_id
    assert linker.roster == frozenset({return_raw_id})


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
        enrolment_window_s=enrolment_window_s,
        evidence_window_s=10.0,
        evidence_frames=1,
        absent_death_s=absent_death_s,
        min_mask_area_px=100,
    )
    survivor_id, doomed_id = 601, 602
    for _ in range(window):
        rows = [
            {"tracker_id": survivor_id, "box": BOX_A, "bgr": (70, 70, 70)},
            {"tracker_id": doomed_id, "box": BOX_B, "bgr": (80, 80, 80)},
        ]
        frame, dets = build_call(rows)
        linker.update(dets, frame)
    assert linker.roster == frozenset({survivor_id, doomed_id})

    survivor_gallery_calls_before = len(
        [c for c in matcher.build_gallery_calls if c["meta"].get("identity") == survivor_id]
    )
    assert survivor_gallery_calls_before == 1, "sanity: only the Start gallery so far"

    # `death_threshold` consecutive absences for BOTH ids — exactly AT the
    # threshold, never exceeding it.
    for _ in range(death_threshold):
        frame, dets = empty_call()
        linker.update(dets, frame)

    # survivor_id reappears on this call, having been absent for exactly
    # `death_threshold` frames (not more) — coasting, not a death+return.
    # doomed_id stays absent on this same call, crossing "more than
    # death_threshold" consecutive absences, so it genuinely dies here.
    rows = [{"tracker_id": survivor_id, "box": BOX_A, "bgr": (70, 70, 70)}]
    frame, dets = build_call(rows)
    out = linker.update(dets, frame)
    assert output_tracker_id(out, rows, survivor_id) == survivor_id, (
        "reappearance at exactly the absence threshold must be treated as coasting, not a death"
    )

    survivor_gallery_calls_after = len(
        [c for c in matcher.build_gallery_calls if c["meta"].get("identity") == survivor_id]
    )
    assert survivor_gallery_calls_after == survivor_gallery_calls_before, (
        "no death embed must ever fire for an identity that only coasted through the threshold"
    )

    # doomed_id is now genuinely Missing; a fresh pending track's evidence
    # window closes against the complete roster comparison dictionary.
    query_bgr = (5, 15, 25)
    marker = rgb_marker_for_bgr(query_bgr)
    matcher.program(marker, [({doomed_id: 0.9, survivor_id: 0.1}, doomed_id)])
    rows = [{"tracker_id": 701, "box": BOX_C, "bgr": query_bgr}]
    frame, dets = build_call(rows)
    linker.update(dets, frame)

    matching_calls = [c for c in matcher.score_calls if c["marker"] == marker]
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
        enrolment_window_s=enrolment_window_s,
        evidence_window_s=evidence_window_s,
        evidence_frames=3,  # never reached: neither row below ever buffers a quality crop
        absent_death_s=10.0,
        min_mask_area_px=min_mask_area_px,
    )
    roster_id = 5001
    for _ in range(window):
        rows = [{"tracker_id": roster_id, "box": BOX_A, "bgr": (200, 200, 200)}]
        frame, dets = build_call(rows)
        linker.update(dets, frame)
    assert linker.roster == frozenset({roster_id})

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
            {"tracker_id": roster_id, "box": BOX_A, "bgr": (200, 200, 200)},
            {"tracker_id": edge_id, "box": edge_box, "bgr": edge_bgr},
            {"tracker_id": tiny_id, "box": tiny_box, "bgr": tiny_bgr},
        ]
        frame, dets = build_call(rows)
        out = linker.update(dets, frame)
        assert output_tracker_id(out, rows, edge_id) == edge_id
        assert output_tracker_id(out, rows, tiny_id) == tiny_id

    def crop_markers(crops):
        return {tuple(int(v) for v in c[0, 0]) for c in crops}

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


class _GalleryCapturingMatcher(FakeMatcher):
    """FakeMatcher variant that additionally records each score() call's
    per-identity gallery view-counts.

    The base FakeMatcher only logs `gallery_keys` (the dict's key set),
    which is enough for the existing suite but cannot distinguish a Missing
    identity's gallery *growing* across repeated deaths from it staying the
    same size — exactly what B3.2's replace-not-append semantics turns on.
    This subclass changes no inherited behavior; it only appends one extra
    field to the recorded score_calls entry.
    """

    def score(self, query_crops, query_masks, meta, galleries):
        view_counts = {key: len(gallery.views) for key, gallery in galleries.items()}
        result = super().score(query_crops, query_masks, meta, galleries)
        self.score_calls[-1]["gallery_view_counts"] = view_counts
        return result


def test_b3_2_last_seen_views_are_replaced_not_appended_across_deaths():
    fps = 4.0
    enrolment_window_s = 0.25
    window = round(enrolment_window_s * fps)
    absent_death_s = 0.5
    death_threshold = round(absent_death_s * fps)
    evidence_frames = 2  # first quality appearance is Pending, second decides (mirrors test_b1_4)
    matcher = _GalleryCapturingMatcher()
    linker = SessionLinker(
        matcher,
        fps=fps,
        enrolment_window_s=enrolment_window_s,
        evidence_window_s=10.0,
        evidence_frames=evidence_frames,
        absent_death_s=absent_death_s,
        min_mask_area_px=100,
    )
    roster_id = 3001
    for _ in range(window):
        rows = [{"tracker_id": roster_id, "box": BOX_A, "bgr": (90, 30, 30)}]
        frame, dets = build_call(rows)
        linker.update(dets, frame)
    assert linker.roster == frozenset({roster_id})

    # One post-freeze presence frame so the identity's rolling last-seen
    # buffer holds "crops A": enrolment crops feed only the Start gallery,
    # and a death with an empty buffer legally skips its embed.
    rows = [{"tracker_id": roster_id, "box": BOX_A, "bgr": (90, 30, 30)}]
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
    matcher.program(return_marker, [({roster_id: 0.9}, roster_id)])
    return_raw_id = 3101

    rows = [{"tracker_id": return_raw_id, "box": BOX_B, "bgr": return_bgr}]
    frame, dets = build_call(rows)
    out = linker.update(dets, frame)
    assert output_tracker_id(out, rows, return_raw_id) == return_raw_id, "still Pending on the first appearance"
    assert not matcher.score_calls, "must not decide before the window closes"

    rows = [{"tracker_id": return_raw_id, "box": BOX_B, "bgr": return_bgr}]
    frame, dets = build_call(rows)
    out = linker.update(dets, frame)
    assert output_tracker_id(out, rows, return_raw_id) == roster_id, "decides and links on the second appearance"

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

    # A fresh pending track triggers a scoring call that offers roster_id's
    # aggregate Missing gallery (Start + only the most recent death's views).
    # It needs the same evidence_frames=2 quality appearances to decide.
    probe_bgr = (44, 44, 44)
    probe_marker = rgb_marker_for_bgr(probe_bgr)
    matcher.program(probe_marker, [({roster_id: 0.5}, REJECT), ({roster_id: 0.4}, REJECT)])
    probe_raw_id = 3201

    rows = [{"tracker_id": probe_raw_id, "box": BOX_C, "bgr": probe_bgr}]
    frame, dets = build_call(rows)
    linker.update(dets, frame)
    assert not any(c["marker"] == probe_marker for c in matcher.score_calls), (
        "must not decide before the probe's window closes"
    )

    rows = [{"tracker_id": probe_raw_id, "box": BOX_C, "bgr": probe_bgr}]
    frame, dets = build_call(rows)
    linker.update(dets, frame)

    probe_calls = [c for c in matcher.score_calls if c["marker"] == probe_marker]
    assert probe_calls, "expected the probe track's evidence window to close and trigger scoring"
    offered_size_after_second_death = probe_calls[0]["gallery_view_counts"][roster_id]

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
        enrolment_window_s=enrolment_window_s,
        evidence_window_s=evidence_window_s,
        evidence_frames=3,  # never reached: only 1 quality crop is ever buffered
        absent_death_s=absent_death_s,
        min_mask_area_px=100,
    )
    roster_id = 4001
    for _ in range(window):
        rows = [{"tracker_id": roster_id, "box": BOX_A, "bgr": (60, 60, 60)}]
        frame, dets = build_call(rows)
        linker.update(dets, frame)
    assert linker.roster == frozenset({roster_id})

    # Kill roster_id so a genuinely Missing identity exists for the pending
    # track below to be scored against once its window times out.
    for _ in range(death_threshold + 1):
        frame, dets = empty_call()
        linker.update(dets, frame)

    evidence_window = round(evidence_window_s * fps)
    assert evidence_window >= 2, "test needs room for one quality frame followed by low-quality filler frames"

    pending_bgr = (17, 27, 37)
    marker = rgb_marker_for_bgr(pending_bgr)
    matcher.program(marker, [({roster_id: 0.9}, roster_id)])
    pending_raw_id = 4101
    tiny_box = (30, 30, 34, 34)  # 4x4 = 16 px^2, fails min_mask_area_px=100

    # First appearance: a single quality crop -> starts the Pending clock
    # and buffers exactly 1 quality crop.
    rows = [{"tracker_id": pending_raw_id, "box": BOX_B, "bgr": pending_bgr}]
    frame, dets = build_call(rows)
    out = linker.update(dets, frame)
    assert output_tracker_id(out, rows, pending_raw_id) == pending_raw_id
    assert not matcher.score_calls, "must not decide before the window closes"

    # Remaining calls up to (but not including) the window-closing call:
    # low-quality appearances only, so the buffer never reaches evidence_frames.
    for _ in range(evidence_window - 2):
        rows = [{"tracker_id": pending_raw_id, "box": tiny_box, "bgr": pending_bgr}]
        frame, dets = build_call(rows)
        out = linker.update(dets, frame)
        assert output_tracker_id(out, rows, pending_raw_id) == pending_raw_id
        assert not matcher.score_calls, "must not decide before the window closes"

    # The evidence_window-th call since first appearance: still only 1
    # quality crop ever buffered (well under evidence_frames=3), but the
    # elapsed-time criterion fires the decision anyway.
    rows = [{"tracker_id": pending_raw_id, "box": tiny_box, "bgr": pending_bgr}]
    frame, dets = build_call(rows)
    out = linker.update(dets, frame)

    matching_calls = [c for c in matcher.score_calls if c["marker"] == marker]
    assert matching_calls, "expected the evidence window to time out on elapsed frames and trigger a decision"
    assert output_tracker_id(out, rows, pending_raw_id) == roster_id, (
        "the scripted matcher accepted the link; the decision must land on this call"
    )
