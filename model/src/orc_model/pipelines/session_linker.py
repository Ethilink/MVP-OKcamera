"""SessionLinker — raw tracker ids in, session ids out.

Sits after Deep OC-SORT in `load_tracker()`'s composition. Enrols a frozen
roster at Start, binds each enrolled identity to its physical specimen, keeps
per-identity appearance galleries, embeds only at track birth/death/batch-
decision events (never per frame), and re-identifies returning instruments via
the injected SRC matcher behind an open-set gate. See
`model/docs/linker-design.md` for the current contract and rationale.

Gallery binding (linker-design.md §3, wayfinder T08) runs ONCE, inside the
enrolment freeze. Each identity's Start crops are scored with the matcher
against the COMPLETE dict of persistent specimen galleries -- pre-captured
photo sets embedded once by `persistent_gallery.load_persistent_galleries()`
and only ever read here, never re-embedded. An identity binds only when its
best score clears `bind_tau` AND beats its second-best specimen by
`bind_margin`; contested specimens are awarded greedily, best score first, and
a losing identity falls back to session-only rather than to its second-choice
specimen (a wrong bind poisons every later decision; session-only merely has
fewer views). A bound identity wears its specimen number as its session id and
matches on `persistent u Start (u last-seen)`. An unbound one is still a
first-class identity: it gets the next small number above the loaded specimen
numbers and matches on session views only. With no persistent galleries at all
the roster is still renumbered from 1 and everything runs session-only.

Out of scope for v1 (see `model/docs/linker-design.md`):
    TODO(linker-design §5): stability-gated session-refresh ring. Galleries
        are frozen at Start/death; no live crop is ever added to a gallery
        mid-recording.
    TODO(linker-design §8): rotation/mirror gallery augmentation.
        Galleries hold only the raw embedded views, no synthetic copies.
    NOT a TODO (linker-design §9, closed by measurement 2026-07-15): matcher
        calls happen synchronously inside `update()` and that is fine.
        Re-measured on the Take B cached replay (n=46 batches, n=25 deaths),
        timing BOTH halves of the synchronous path -- an earlier pass timed only
        the solve, which is the cheap half and never the reason to go async:
            solve   (`total_ms`, score+assign): median  40.1 ms / max  70.1 ms
            embed   (`build_ms` at death):      median  41.0 ms / max  67.3 ms
            embed   (`build_ms` at enrolment):  260.4 ms, ONCE, at Start
        Worst case both land on one frame: ~137 ms of linker work on top of the
        detector's ~330 ms, i.e. an occasional ~470 ms frame against a 333 ms
        nominal budget at 3 fps. That is a spike, not a stall, and it is three
        orders inside the 1.0 s resolve contract. The 260 ms enrolment freeze is
        by design (the table is still; nothing is being tracked yet).
        NOTE: this kills the map's old "~0.2-0.5 s per link event" estimate --
        that was a guess, and it was ~6x pessimistic. Embeds are ~3 crops,
        batched, on MPS.
        Reproduce: model/scripts/replay_session.py --from-cache, then read
        `build_ms` / `total_ms` out of the trace's `logs`. Absolute latency
        drifts with machine load (max solve was 142.6 ms on the uncached run
        that demo-validation.md records, 70.1 ms cached here); the batch COUNT
        (46 on Take B) is the stable, reproducible figure.
        Do not build threading for this without new evidence.
    TODO(linker-design out-of-scope): catalog identification, the
        `confidence` attribute (lives on `load_tracker`'s tracker), and `-1`
        id stripping (upstream/OC-SORT guarantees mature ids only) are not
        this module's job.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import NamedTuple

import cv2
import numpy as np
import supervision as sv

from orc_model.pipelines.matching import ChampionGallery
from orc_model.pipelines.matching.interface import REJECT  # noqa: F401


class _Crop(NamedTuple):
    rgb: np.ndarray
    mask: np.ndarray
    quality: float  # mask_area * confidence -- ranking only, never admission
    ok: bool  # quality-admitted: big enough mask, not touching the frame edge


class _CropFingerprint(NamedTuple):
    """A cheap, deterministic summary of one usable crop, used only to decide
    (without DINO/matcher) whether a settled Unknown's evidence changed enough
    to re-arm. See `unknown-recovery-SPEC.md` § "Cheap crop-change detector"."""

    rgb: np.ndarray  # (S, S, 3) float32 in [0, 1], background masked to neutral gray
    mask: np.ndarray  # (S, S) bool, nearest-neighbour resized
    quality: float  # the crop's mask_area * confidence at fingerprint time


@dataclass
class _Identity:
    """A frozen roster identity. `active_raw_id` is None while Missing.
    `bound_specimen` is the persistent specimen bound at the freeze, or None
    for a session-only identity."""

    session_id: int
    active_raw_id: int | None
    crop_buffer: deque
    start_views: np.ndarray | None = None
    last_seen_views: np.ndarray | None = None
    absence_streak: int = 0
    bound_specimen: int | None = None


@dataclass
class _PendingTrack:
    """A post-freeze raw id not yet decided (returning instrument vs foreign
    object)."""

    first_seen_frame: int
    # Each entry is `(rgb, mask, quality)` -- quality is carried alongside so a
    # rejection can fingerprint the highest-quality crop STILL IN this bounded
    # window (SPEC § "Fingerprint"), rather than a since-evicted historical best.
    crop_buffer: deque
    absence_streak: int = 0
    not_before_frame: int = 0


@dataclass
class _SettledUnknown:
    """A raw track whose last decision rejected it. Not terminal: while the raw
    track stays present the linker cheaply watches for a material crop change or
    a candidate-target expansion and re-arms a fresh Pending window (SPEC
    § "Internal state", B-UR1..B-UR8)."""

    decision_fingerprint: _CropFingerprint | None  # None == last decision had no usable crop
    decision_quality: float
    candidate_ids_at_decision: frozenset[int]
    last_decision_frame: int
    absence_streak: int = 0


DEFAULT_BIND_TAU = 0.30
DEFAULT_BIND_MARGIN = 0.02
DEFAULT_UNKNOWN_ID_OFFSET = 1000

# Recoverable-Unknown recheck gates (SPEC § "Material-change rule"). Engineering
# starting points, converted to frames via fps where relevant; tests pass
# explicit values and never assert these defaults.
DEFAULT_UNKNOWN_RECHECK_COOLDOWN_S = 0.75
DEFAULT_UNKNOWN_RECHECK_APPEARANCE_DELTA = 0.15
DEFAULT_UNKNOWN_RECHECK_MASK_IOU = 0.75
DEFAULT_UNKNOWN_RECHECK_QUALITY_GAIN = 0.25
DEFAULT_UNKNOWN_RECHECK_FINGERPRINT_PX = 32


def _round4(value: float | None) -> float | None:
    """Round a metric for the re-arm log, passing None (no measurement) through."""
    return None if value is None else round(value, 4)


class SessionLinker:
    """Remaps Deep OC-SORT's raw `tracker_id`s to stable session ids that
    survive absence. See the module docstring and `linker-design.md`."""

    def __init__(
        self,
        matcher,
        *,
        fps: float,
        persistent_galleries: dict[int, ChampionGallery] | None = None,
        bind_tau: float = DEFAULT_BIND_TAU,
        bind_margin: float = DEFAULT_BIND_MARGIN,
        unknown_id_offset: int = DEFAULT_UNKNOWN_ID_OFFSET,
        enrolment_window_s: float = 0.5,
        evidence_window_s: float = 0.5,
        evidence_frames: int = 3,
        absent_death_s: float = 1.0,
        min_mask_area_px: int = 200,
        unknown_recheck_cooldown_s: float = DEFAULT_UNKNOWN_RECHECK_COOLDOWN_S,
        unknown_recheck_appearance_delta: float = DEFAULT_UNKNOWN_RECHECK_APPEARANCE_DELTA,
        unknown_recheck_mask_iou: float = DEFAULT_UNKNOWN_RECHECK_MASK_IOU,
        unknown_recheck_quality_gain: float = DEFAULT_UNKNOWN_RECHECK_QUALITY_GAIN,
        unknown_recheck_fingerprint_px: int = DEFAULT_UNKNOWN_RECHECK_FINGERPRINT_PX,
        logger: logging.Logger | None = None,
    ) -> None:
        # Already embedded by `persistent_gallery.load_persistent_galleries()`
        # and held resident across reset() -- never re-embedded here.
        self._persistent_galleries = persistent_galleries or {}
        self._bind_tau = bind_tau
        self._bind_margin = bind_margin
        self._unknown_id_offset = unknown_id_offset
        self._matcher = matcher
        self._evidence_frames = evidence_frames
        self._min_mask_area_px = min_mask_area_px
        self._enrolment_window_frames = max(1, round(enrolment_window_s * fps))
        self._evidence_window_frames = round(evidence_window_s * fps)
        self._death_threshold_frames = round(absent_death_s * fps)
        # Recoverable-Unknown recheck gates (SPEC § "Cheap crop-change detector").
        self._unknown_recheck_cooldown_frames = round(unknown_recheck_cooldown_s * fps)
        self._unknown_recheck_appearance_delta = unknown_recheck_appearance_delta
        self._unknown_recheck_mask_iou = unknown_recheck_mask_iou
        self._unknown_recheck_quality_gain = unknown_recheck_quality_gain
        self._unknown_recheck_fingerprint_px = unknown_recheck_fingerprint_px
        self._logger = logger or logging.getLogger("orc_model.session_linker")
        self.reset()

    def reset(self) -> None:
        self._frame_count = 0
        self._enrolment_start: int | None = None
        self._enrolment_frames_seen = 0
        self._enrolment_presence: dict[int, int] = {}
        self._enrolment_crops: dict[int, list[_Crop]] = {}
        self._enrolled = False
        self._roster: frozenset[int] = frozenset()
        self._raw_to_session: dict[int, int] = {}
        self._identities: dict[int, _Identity] = {}
        self._pending: dict[int, _PendingTrack] = {}
        # Recoverable Unknown: raw id -> its last-decision baseline (fingerprint,
        # quality, candidate target set, decision frame). Replaces the old
        # terminal `set[int]`; a settled Unknown is observed, not forgotten,
        # until it re-arms or ages out (SPEC B-UR9 clears it here).
        self._settled_unknown: dict[int, _SettledUnknown] = {}

    @property
    def roster(self) -> frozenset[int]:
        return self._roster

    # -- main entry point -----------------------------------------------

    def update(self, detections: sv.Detections, frame: np.ndarray) -> sv.Detections:
        self._frame_count += 1
        is_empty = len(detections) == 0

        if not self._enrolled:
            return self._update_during_enrolment(detections, frame, is_empty)

        present = set() if is_empty else {int(r) for r in detections.tracker_id}
        self._age_out_missing_actives(present)
        self._age_out_pending(present)
        self._age_out_settled_unknown(present)

        if is_empty:
            self._run_batch_decision(self._collect_closed_pending())
            return sv.Detections.empty()

        row_by_raw_id = {int(raw_id): i for i, raw_id in enumerate(detections.tracker_id)}
        emitted = np.empty(len(detections), dtype=int)
        for raw_id, i in row_by_raw_id.items():
            emitted[i] = self._process_row(detections, frame, raw_id, i)

        resolved = self._run_batch_decision(self._collect_closed_pending())
        for raw_id, session_id in resolved.items():
            if session_id is not None and raw_id in row_by_raw_id:
                emitted[row_by_raw_id[raw_id]] = session_id

        output = detections[np.arange(len(detections))]
        output.tracker_id = emitted
        # A row is *resolving* iff, after this frame's batch decision, its raw id
        # is still an undecided pending track -- which includes a deferred one,
        # since `_run_batch_decision` puts deferred raw ids straight back into
        # `_pending`. A winner (now a roster id) and a settled Unknown have both
        # been removed from `_pending`, so both read False. This is the truthful
        # wait-state the overlay spinner gates on (grilled 2026-07-16).
        output.data["resolving"] = np.array(
            [int(raw_id) in self._pending for raw_id in detections.tracker_id], dtype=bool
        )
        return output

    def _process_row(self, detections: sv.Detections, frame: np.ndarray, raw_id: int, i: int) -> int:
        crop = self._crop_for_row(
            frame, detections.xyxy[i], detections.mask[i], detections.confidence[i]
        )

        if raw_id in self._raw_to_session:
            session_id = self._raw_to_session[raw_id]
            identity = self._identities[session_id]
            identity.absence_streak = 0
            if crop is not None and crop.ok:
                identity.crop_buffer.append((crop.rgb, crop.mask))
            return session_id

        if raw_id in self._settled_unknown:
            return self._observe_settled_unknown(raw_id, crop)

        pending = self._pending.get(raw_id)
        if pending is None:
            pending = _PendingTrack(
                first_seen_frame=self._frame_count,
                crop_buffer=deque(maxlen=self._evidence_frames),
            )
            self._pending[raw_id] = pending
        pending.absence_streak = 0
        if crop is not None and crop.ok:
            pending.crop_buffer.append((crop.rgb, crop.mask, crop.quality))
        # Not (yet) a roster identity: emit in the offset id space, so the
        # roster and the unknown/pending ranges stay disjoint forever. Internal
        # state stays keyed by the raw id.
        return raw_id + self._unknown_id_offset

    # -- B6: crop extraction ---------------------------------------------

    def _crop_for_row(
        self, frame: np.ndarray, xyxy: np.ndarray, mask: np.ndarray, confidence: float
    ) -> _Crop | None:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = (int(round(float(v))) for v in xyxy)
        touches_edge = x1 <= 0 or y1 <= 0 or x2 >= w or y2 >= h

        cx1, cy1 = max(0, x1), max(0, y1)
        cx2, cy2 = min(w, x2), min(h, y2)
        if cx2 <= cx1 or cy2 <= cy1:
            return None  # zero area after clamping -> no crop

        rgb = np.ascontiguousarray(frame[cy1:cy2, cx1:cx2, ::-1])  # BGR -> RGB, own copy
        mask_crop = np.ascontiguousarray(mask[cy1:cy2, cx1:cx2])

        mask_area = int(mask.sum())
        ok = mask_area >= self._min_mask_area_px and not touches_edge
        quality = mask_area * float(confidence)
        return _Crop(rgb=rgb, mask=mask_crop, quality=quality, ok=ok)

    # -- B2: enrolment window & roster freeze -----------------------------

    def _update_during_enrolment(
        self, detections: sv.Detections, frame: np.ndarray, is_empty: bool
    ) -> sv.Detections:
        if self._enrolment_start is None:
            if is_empty:
                return sv.Detections.empty()
            self._enrolment_start = self._frame_count

        self._enrolment_frames_seen += 1
        if not is_empty:
            for i, raw_id in enumerate(detections.tracker_id):
                raw_id = int(raw_id)
                self._enrolment_presence[raw_id] = self._enrolment_presence.get(raw_id, 0) + 1
                crop = self._crop_for_row(
                    frame, detections.xyxy[i], detections.mask[i], detections.confidence[i]
                )
                if crop is not None and crop.ok:
                    self._enrolment_crops.setdefault(raw_id, []).append(crop)

        if self._enrolment_frames_seen >= self._enrolment_window_frames:
            self._freeze_roster()

        if is_empty:
            return sv.Detections.empty()
        output = detections[np.arange(len(detections))]
        if self._enrolled:
            # B-N5: the freeze frame ITSELF emits session ids -- every frame
            # before it passed raw ids through untouched, but `_freeze_roster()`
            # has just published `self._roster`, and callers sample `roster` and
            # the emitted ids in the SAME tick. Leaving raw ids on this frame
            # would let them collide with session ids and book presence for the
            # wrong instruments. Anything not on the roster emits in the offset
            # id space, exactly like every other post-freeze frame (B-N4).
            #
            # Only the id remap runs here. `_pending` is empty entering the
            # freeze, so the pending/batch passes really are no-ops, and this
            # frame's crops are already in the enrolment buffers. The age-out
            # pass is SKIPPED rather than no-op: an identity majority-qualified
            # but absent on this final enrolment frame would be charged an
            # absence on the very frame it enrolled, which is worse than not
            # counting the frame -- and no clause requires counting it.
            output.tracker_id = np.array(
                [
                    self._raw_to_session.get(int(raw_id), int(raw_id) + self._unknown_id_offset)
                    for raw_id in detections.tracker_id
                ],
                dtype=int,
            )
        # During enrolment nothing has a settled identity yet, so every track is
        # resolving; on the freeze frame the roster members flip to settled
        # (in-roster) while any not-in-roster track stays resolving. `_roster` is
        # empty pre-freeze, so `id not in roster` is True for the whole window --
        # the same wait-state the post-freeze path emits.
        output.data["resolving"] = np.array(
            [int(session_id) not in self._roster for session_id in output.tracker_id], dtype=bool
        )
        return output

    def _freeze_roster(self) -> None:
        majority = self._enrolment_frames_seen / 2
        roster_ids = sorted(
            raw_id for raw_id, count in self._enrolment_presence.items() if count > majority
        )

        t0 = time.monotonic()
        start_crops: dict[int, list[_Crop]] = {}
        start_views: dict[int, np.ndarray] = {}
        for raw_id in roster_ids:
            best_crops = sorted(
                self._enrolment_crops.get(raw_id, []), key=lambda c: c.quality, reverse=True
            )[:3]
            if not best_crops:
                continue  # no usable Start crops -> session-only, never binds
            start_crops[raw_id] = best_crops
            gallery = self._matcher.build_gallery(
                [c.rgb for c in best_crops], [c.mask for c in best_crops],
                {"identity": raw_id},
            )
            start_views[raw_id] = gallery.views

        bound, bind_scores = self._bind_specimens(roster_ids, start_crops)
        session_ids = self._assign_session_ids(roster_ids, bound)
        build_ms = (time.monotonic() - t0) * 1000

        self._identities = {
            session_ids[raw_id]: _Identity(
                session_id=session_ids[raw_id],
                active_raw_id=raw_id,
                crop_buffer=deque(maxlen=self._evidence_frames),
                start_views=start_views.get(raw_id),
                bound_specimen=bound.get(raw_id),
            )
            for raw_id in roster_ids
        }
        self._raw_to_session = {raw_id: session_ids[raw_id] for raw_id in roster_ids}
        self._roster = frozenset(session_ids.values())
        self._enrolled = True
        # B-O1: `bound` keyed by session id is tautological (session_id == specimen
        # for every bound identity), so it reveals nothing about which RAW tracker
        # id claimed which photo set. `raw_binds` is the diagnostic a live mis-bind
        # actually needs -> raw tracker id -> specimen.
        raw_binds = {raw: specimen for raw, specimen in sorted(bound.items())}
        self._logger.info(
            "enrolment freeze: roster_size=%d views=%s bound=%s raw_binds=%s session_only=%s "
            "bind_scores=%s build_ms=%.1f",
            len(roster_ids),
            {session_ids[r]: len(start_views.get(r, ())) for r in roster_ids},
            {session_ids[r]: specimen for r, specimen in sorted(bound.items())},
            raw_binds,
            sorted(session_ids[r] for r in roster_ids if r not in bound),
            {
                session_ids[r]: (round(best, 4), round(second, 4))
                for r, (best, second) in sorted(bind_scores.items())
            },
            build_ms,
            # Structured payload for the app's optional --debug console (backend.debug).
            # Ignored by every default handler; carries no behaviour.
            extra={
                "orc": {
                    "event": "freeze",
                    "build_ms": round(build_ms, 1),
                    "bind_tau": self._bind_tau,
                    "roster": [
                        {
                            "session_id": session_ids[r],
                            "raw_id": r,
                            "specimen": bound.get(r),
                            "score": (
                                round(bind_scores[r][0], 4) if r in bind_scores else None
                            ),
                        }
                        for r in roster_ids
                    ],
                }
            },
        )

    def _bind_specimens(
        self, roster_ids: list[int], start_crops: dict[int, list[_Crop]]
    ) -> tuple[dict[int, int], dict[int, tuple[float, float]]]:
        """Bind enrolled identities one-to-one to persistent specimens.

        Greedy, not Hungarian, and deliberately so (linker-design.md §3 says
        "Hungarian"; §6.5's shipped philosophy wins): a global optimum would
        push a losing identity onto its second-choice specimen. A wrong bind
        poisons every later decision, while session-only merely means fewer
        views -- so a contested loser falls back, it never settles for second.

        Every identity is scored against the COMPLETE loaded gallery dict; the
        comparison set is never shrunk to the unclaimed specimens, and no
        second round is run (the same relative-SCI trap as
        `_comparison_galleries`).
        """
        galleries = self._persistent_galleries
        if not galleries:
            return {}, {}

        proposals: list[tuple[float, int, int]] = []  # (score, raw_id, specimen)
        bind_scores: dict[int, tuple[float, float]] = {}
        for raw_id in roster_ids:  # ascending raw-id order
            crops = start_crops.get(raw_id)
            if not crops:
                continue
            scores = self._matcher.score(
                [c.rgb for c in crops], [c.mask for c in crops], {}, galleries
            )
            if not scores:
                continue
            ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
            specimen, best = ranked[0]
            second = ranked[1][1] if len(ranked) > 1 else 0.0
            bind_scores[raw_id] = (float(best), float(second))
            if best < self._bind_tau:
                continue
            if len(galleries) > 1 and best - second < self._bind_margin:
                continue  # ambiguous -> session-only, never a coin-flip bind
            proposals.append((float(best), raw_id, specimen))

        bound: dict[int, int] = {}
        claimed: set[int] = set()
        for _score, raw_id, specimen in sorted(proposals, key=lambda p: (-p[0], p[1])):
            if specimen in claimed:
                continue  # outbid -> session-only, NOT its second choice
            claimed.add(specimen)
            bound[raw_id] = specimen
        return bound, bind_scores

    def _assign_session_ids(self, roster_ids: list[int], bound: dict[int, int]) -> dict[int, int]:
        """Bound identity -> its specimen number; session-only identity -> the
        next number above every LOADED specimen number, so a specimen number is
        never worn by an identity that didn't confidently bind."""
        base = max(self._persistent_galleries) if self._persistent_galleries else 0
        session_ids: dict[int, int] = {}
        next_session_only = base
        for raw_id in roster_ids:  # ascending raw-id order
            if raw_id in bound:
                session_ids[raw_id] = bound[raw_id]
                continue
            next_session_only += 1
            session_ids[raw_id] = next_session_only

        collisions = sorted(s for s in session_ids.values() if s >= self._unknown_id_offset)
        if collisions:
            self._logger.warning(
                "session ids %s reach unknown_id_offset=%d -- roster and unknown id "
                "ranges are no longer disjoint",
                collisions, self._unknown_id_offset,
            )
        return session_ids

    # -- B3: death (Active -> Missing) ------------------------------------

    def _age_out_missing_actives(self, present: set[int]) -> None:
        for identity in self._identities.values():
            if identity.active_raw_id is None:
                continue
            if identity.active_raw_id in present:
                # Reset the streak HERE, in the age pass, not later when this
                # identity's own row is processed. Otherwise a settled-Unknown row
                # observed EARLIER in the same frame's row loop would still read a
                # stale positive streak and wrongly count a fully-present identity
                # as a coasting candidate target -- a row-order-dependent spurious
                # re-arm (SPEC § "Candidate target set"; invariant #10). The
                # `_process_row` reset below is now redundant but harmless.
                identity.absence_streak = 0
                continue
            identity.absence_streak += 1
            if identity.absence_streak > self._death_threshold_frames:
                self._kill(identity)

    def _kill(self, identity: _Identity) -> None:
        t0 = time.monotonic()
        retired_raw_id = identity.active_raw_id
        crops = [c for c, _ in identity.crop_buffer]
        masks = [m for _, m in identity.crop_buffer]
        if crops:
            gallery = self._matcher.build_gallery(crops, masks, {"identity": identity.session_id})
            identity.last_seen_views = gallery.views
        build_ms = (time.monotonic() - t0) * 1000

        identity.active_raw_id = None
        if retired_raw_id is not None:
            self._raw_to_session.pop(retired_raw_id, None)
        identity.absence_streak = 0
        identity.crop_buffer = deque(maxlen=self._evidence_frames)
        self._logger.info(
            "death: identity=%d crops_used=%d build_ms=%.1f",
            identity.session_id, len(crops), build_ms,
            extra={"orc": {"event": "death", "session_id": identity.session_id}},
        )

    # -- B4: birth & evidence window ---------------------------------------

    def _age_out_pending(self, present: set[int]) -> None:
        for raw_id in list(self._pending):
            if raw_id in present:
                continue
            pending = self._pending[raw_id]
            pending.absence_streak += 1
            if pending.absence_streak > self._death_threshold_frames:
                del self._pending[raw_id]  # died before its window closed -> no decision

    def _collect_closed_pending(self) -> list[tuple[int, _PendingTrack]]:
        closed = []
        for raw_id in list(self._pending):
            pending = self._pending[raw_id]
            if self._frame_count < pending.not_before_frame:
                continue
            elapsed = self._frame_count - pending.first_seen_frame + 1
            if elapsed >= self._evidence_window_frames or len(pending.crop_buffer) >= self._evidence_frames:
                closed.append((raw_id, pending))
                del self._pending[raw_id]
        return closed

    # -- B-UR: recoverable Unknown (change-triggered re-identification) -------

    def _age_out_settled_unknown(self, present: set[int]) -> None:
        """SPEC B-UR8. Present -> reset the absence streak; absent -> increment it
        and delete the settled state once it exceeds the death threshold, so a
        later occurrence of the raw id is treated as a fresh Pending track and
        rejected-track state cannot grow session-long."""
        for raw_id in list(self._settled_unknown):
            settled = self._settled_unknown[raw_id]
            if raw_id in present:
                settled.absence_streak = 0
                continue
            settled.absence_streak += 1
            if settled.absence_streak > self._death_threshold_frames:
                del self._settled_unknown[raw_id]

    def _observe_settled_unknown(self, raw_id: int, crop: _Crop | None) -> int:
        """Cheaply watch a settled Unknown and re-arm it as Pending when its
        evidence materially changed or a new link target appeared. Returns the
        offset id to emit while it stays (or becomes) not-in-roster.

        This never calls the matcher (SPEC B-UR5): the only work here is the
        low-cost fingerprint comparison, and only once the cheap gates pass."""
        settled = self._settled_unknown[raw_id]
        offset_id = raw_id + self._unknown_id_offset

        if crop is None or not crop.ok:
            return offset_id  # no usable evidence this frame -> stay settled
        if self._frame_count - settled.last_decision_frame < self._unknown_recheck_cooldown_frames:
            return offset_id  # rate-limit successive decisions for one raw id
        candidate_ids = self._candidate_target_ids()
        if not candidate_ids:
            return offset_id  # nothing this track could legitimately recover into

        trigger, metrics = self._recheck_trigger(settled, crop, candidate_ids)
        if trigger is None:
            return offset_id  # unchanged evidence and no target expansion

        self._rearm_unknown(raw_id, crop, settled, candidate_ids, trigger, metrics)
        return offset_id  # still offset; the fresh Pending window decides later

    def _recheck_trigger(
        self, settled: _SettledUnknown, crop: _Crop, candidate_ids: frozenset[int]
    ) -> tuple[str | None, dict[str, float | None]]:
        """Decide whether (and why) a settled Unknown re-arms. Any one signal
        crossing its gate is sufficient (SPEC § "Material-change rule")."""
        metrics: dict[str, float | None] = {
            "appearance_delta": None,
            "mask_iou": None,
            "quality_gain": None,
        }
        expanded = bool(candidate_ids - settled.candidate_ids_at_decision)

        if settled.decision_fingerprint is None:
            # B-UR7: the last decision had no usable crop, so this first usable
            # crop is itself new evidence.
            return "no_previous_evidence", metrics

        fingerprint = self._fingerprint(crop.rgb, crop.mask, crop.quality)
        appearance_delta = float(np.mean(np.abs(fingerprint.rgb - settled.decision_fingerprint.rgb)))
        mask_iou = self._mask_iou(fingerprint.mask, settled.decision_fingerprint.mask)
        quality_gain = (crop.quality / max(settled.decision_quality, 1.0)) - 1.0
        metrics = {
            "appearance_delta": appearance_delta,
            "mask_iou": mask_iou,
            "quality_gain": quality_gain,
        }

        if appearance_delta >= self._unknown_recheck_appearance_delta:
            return "appearance", metrics
        if mask_iou <= self._unknown_recheck_mask_iou:
            return "mask", metrics
        if quality_gain >= self._unknown_recheck_quality_gain:
            return "quality", metrics
        if expanded:
            return "target_expanded", metrics
        return None, metrics

    def _rearm_unknown(
        self,
        raw_id: int,
        crop: _Crop,
        settled: _SettledUnknown,
        candidate_ids: frozenset[int],
        trigger: str,
        metrics: dict[str, float | None],
    ) -> None:
        """Move a raw id from Settled Unknown back to a fresh normal Pending
        window, seeded with only the triggering crop (SPEC B-UR2: rejected-window
        crops are never mixed in). This links nothing; the existing batch path
        makes the eventual decision."""
        del self._settled_unknown[raw_id]
        pending = _PendingTrack(
            first_seen_frame=self._frame_count,
            crop_buffer=deque(maxlen=self._evidence_frames),
        )
        pending.crop_buffer.append((crop.rgb, crop.mask, crop.quality))
        self._pending[raw_id] = pending

        frames_since = self._frame_count - settled.last_decision_frame
        old_candidates = sorted(settled.candidate_ids_at_decision)
        new_candidates = sorted(candidate_ids)
        self._logger.info(
            "unknown re-armed: raw_id=%d trigger=%s appearance_delta=%s mask_iou=%s "
            "quality_gain=%s old_candidates=%s new_candidates=%s frames_since_decision=%d",
            raw_id, trigger,
            _round4(metrics["appearance_delta"]), _round4(metrics["mask_iou"]),
            _round4(metrics["quality_gain"]), old_candidates, new_candidates, frames_since,
            # Structured payload for the app's optional --debug console (backend.debug).
            extra={
                "orc": {
                    "event": "unknown_rearm",
                    "raw_id": raw_id,
                    "trigger": trigger,
                    "appearance_delta": metrics["appearance_delta"],
                    "mask_iou": metrics["mask_iou"],
                    "quality_gain": metrics["quality_gain"],
                    "old_candidates": old_candidates,
                    "new_candidates": new_candidates,
                    "frames_since_decision": frames_since,
                }
            },
        )

    def _settle_unknown(
        self, raw_id: int, pending: _PendingTrack, candidate_ids: frozenset[int]
    ) -> None:
        """Record (or replace) a raw id's Settled Unknown baseline from the just-
        closed window: the fingerprint of the highest-quality crop STILL IN the
        window (or None if it had no usable crop), its quality, the candidate
        target set considered, and this decision frame. Deriving the crop from the
        current `crop_buffer` -- not a separately tracked all-time best -- keeps
        the baseline aligned with the crops the matcher actually scored, including
        after a deferred window evicted its earlier crops (SPEC B-UR6). Never
        appends decision history (SPEC § "Internal state")."""
        best = max(pending.crop_buffer, key=lambda entry: entry[2], default=None)
        if best is None:
            fingerprint, quality = None, 0.0
        else:
            rgb, mask, quality = best
            fingerprint = self._fingerprint(rgb, mask, quality)
        self._settled_unknown[raw_id] = _SettledUnknown(
            decision_fingerprint=fingerprint,
            decision_quality=quality,
            candidate_ids_at_decision=candidate_ids,
            last_decision_frame=self._frame_count,
        )

    def _candidate_target_ids(self) -> frozenset[int]:
        """The identities that could legitimately receive or defer a link on this
        frame AND have a usable gallery (SPEC § "Candidate target set"): every
        Missing identity, plus every Active identity currently coasting inside its
        death grace. A cheap trigger only -- matcher comparison stays against the
        complete frozen roster."""
        return frozenset(
            session_id
            for session_id, identity in self._identities.items()
            if self._has_usable_gallery(identity)
            and (identity.active_raw_id is None or identity.absence_streak > 0)
        )

    @staticmethod
    def _has_usable_gallery(identity: _Identity) -> bool:
        """Whether an identity would appear in `_comparison_galleries()` -- i.e.
        has at least one gallery source (persistent, Start, or last-seen)."""
        return (
            identity.bound_specimen is not None
            or identity.start_views is not None
            or identity.last_seen_views is not None
        )

    def _fingerprint(self, rgb: np.ndarray, mask: np.ndarray, quality: float) -> _CropFingerprint:
        """A small, deterministic RGB+mask thumbnail (SPEC § "Fingerprint"):
        neutral-gray outside the mask, bilinear-resized RGB, nearest mask."""
        size = self._unknown_recheck_fingerprint_px
        rgb = rgb.astype(np.float32)
        mask = mask.astype(bool)
        masked = np.where(mask[:, :, None], rgb, 128.0).astype(np.float32)
        rgb_small = cv2.resize(masked, (size, size), interpolation=cv2.INTER_LINEAR)
        mask_small = cv2.resize(
            mask.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST
        ).astype(bool)
        return _CropFingerprint(
            rgb=(rgb_small / 255.0).astype(np.float32),
            mask=mask_small,
            quality=quality,
        )

    @staticmethod
    def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
        intersection = int(np.logical_and(a, b).sum())
        union = int(np.logical_or(a, b).sum())
        if union == 0:
            return 1.0  # both empty -> treat as identical (no mask-change signal)
        return intersection / union

    # -- B5: batched, gated decision ----------------------------------------

    def _comparison_galleries(self) -> dict[int, ChampionGallery]:
        """Return every usable frozen-roster gallery.

        A bound identity matches on `persistent u Start (u last-seen)` -- the
        persistent views are the cached vectors from the freeze, not a re-embed.
        An unbound identity has session views only.

        Gallery membership is deliberately independent of Active/Missing
        state.  SRC's SCI gate is a relative statistic calibrated against the
        complete roster; removing Active identities would collapse the normal
        one-missing case to K=1 and destroy its open-set rejection signal.
        """
        galleries: dict[int, ChampionGallery] = {}
        for session_id, identity in self._identities.items():
            views = []
            if identity.bound_specimen is not None:
                views.append(self._persistent_galleries[identity.bound_specimen].views)
            views += [v for v in (identity.start_views, identity.last_seen_views) if v is not None]
            if not views:
                continue
            merged = views[0] if len(views) == 1 else np.concatenate(views, axis=0)
            galleries[session_id] = ChampionGallery(views=merged, identity=session_id)
        return galleries

    def _eligible_missing_ids(self, galleries: dict[int, ChampionGallery]) -> set[int]:
        """Return Missing identities that have evidence and may receive a link."""
        return {
            session_id
            for session_id, identity in self._identities.items()
            if identity.active_raw_id is None and session_id in galleries
        }

    def _run_batch_decision(self, closed: list[tuple[int, _PendingTrack]]) -> dict[int, int | None]:
        if not closed:
            return {}

        t_start = time.monotonic()
        resolved: dict[int, int | None] = {}
        rows = []
        pending_crops_by_raw_id: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
        pending_by_raw_id = dict(closed)
        # The candidate target set considered for every row settling on this tick.
        # Sampled BEFORE any link mutates identity state, so it reflects the
        # targets available when these decisions were made (SPEC § "Candidate
        # target set"; used as each Settled Unknown's re-arm baseline).
        candidate_ids = self._candidate_target_ids()
        for raw_id, pending in closed:
            crop_tuples = list(pending.crop_buffer)  # (rgb, mask, quality)
            crops = [rgb for rgb, _mask, _quality in crop_tuples]
            masks = [mask for _rgb, mask, _quality in crop_tuples]
            if not crops:
                resolved[raw_id] = None
                self._settle_unknown(raw_id, pending, candidate_ids)
                continue
            rows.append((raw_id, crops, masks))
            # `_link` transfers these into the identity's rolling buffer, which is
            # (rgb, mask) pairs -- strip the quality carried for fingerprinting.
            pending_crops_by_raw_id[raw_id] = [(rgb, mask) for rgb, mask, _quality in crop_tuples]

        comparison_galleries = self._comparison_galleries()
        eligible_missing_ids = self._eligible_missing_ids(comparison_galleries)
        deferable_active_ids = {
            session_id
            for session_id, identity in self._identities.items()
            if identity.active_raw_id is not None
            and identity.absence_streak > 0
            and session_id in comparison_galleries
        }
        winners, unresolved, deferred, score_ms, assignment_ms = self._score_and_assign(
            rows,
            comparison_galleries,
            eligible_missing_ids,
            deferable_active_ids,
        )

        for raw_id, session_id in winners.items():
            self._link(raw_id, session_id, pending_crops_by_raw_id.get(raw_id, []))
            resolved[raw_id] = session_id
        for raw_id, _crops, _masks in unresolved:
            self._settle_unknown(raw_id, pending_by_raw_id[raw_id], candidate_ids)
            resolved[raw_id] = None
        for raw_id, session_id in deferred.items():
            pending = pending_by_raw_id[raw_id]
            identity = self._identities[session_id]
            frames_until_missing = max(
                1,
                self._death_threshold_frames - identity.absence_streak + 1,
            )
            pending.not_before_frame = self._frame_count + frames_until_missing
            self._pending[raw_id] = pending

        total_ms = (time.monotonic() - t_start) * 1000
        outcomes = {
            **{
                raw_id: ("unknown" if sid is None else f"linked:{sid}")
                for raw_id, sid in resolved.items()
            },
            **{raw_id: f"deferred:{sid}" for raw_id, sid in deferred.items()},
        }
        self._logger.info(
            "batch decision: rows=%d comparison_galleries=%d eligible_missing=%d "
            "atom_counts=%s score_ms=%.1f collision_ms=%.1f total_ms=%.1f outcomes=%s",
            len(rows), len(comparison_galleries), len(eligible_missing_ids),
            # Per-candidate dictionary size: exposes the bound-vs-unbound
            # asymmetry (~18 atoms vs ~3) that biases SRC's score.
            {sid: int(g.views.shape[0]) for sid, g in sorted(comparison_galleries.items())},
            score_ms, assignment_ms, total_ms,
            outcomes,
            extra={"orc": {"event": "decision", "outcomes": outcomes}},
        )
        return resolved

    def _score_and_assign(
        self,
        rows: list[tuple[int, list, list]],
        comparison_galleries: dict[int, ChampionGallery],
        eligible_missing_ids: set[int],
        deferable_active_ids: set[int],
    ) -> tuple[
        dict[int, int],
        list[tuple[int, list, list]],
        dict[int, int],
        float,
        float,
    ]:
        if not rows or not (eligible_missing_ids or deferable_active_ids):
            return {}, rows, {}, 0.0, 0.0

        t_score = time.monotonic()
        proposals: list[tuple[int, int, float]] = []
        deferred_indices: dict[int, int] = {}
        for idx, (raw_id, crops, masks) in enumerate(rows):
            scores = self._matcher.score(crops, masks, {}, comparison_galleries)
            accepted = self._matcher.accept(scores)
            self._logger.debug(
                "match scores: raw_id=%d accepted=%r scores=%s",
                raw_id,
                accepted,
                {candidate: round(float(score), 4) for candidate, score in scores.items()},
            )
            if accepted == REJECT:
                continue
            if accepted not in scores:
                self._logger.warning("matcher accepted identity %r without returning its score", accepted)
                continue
            if accepted in eligible_missing_ids:
                proposals.append((idx, accepted, scores[accepted]))
            elif accepted in deferable_active_ids:
                deferred_indices[idx] = accepted
        score_ms = (time.monotonic() - t_score) * 1000

        if not proposals:
            unresolved = [row for idx, row in enumerate(rows) if idx not in deferred_indices]
            deferred = {rows[idx][0]: session_id for idx, session_id in deferred_indices.items()}
            return {}, unresolved, deferred, score_ms, 0.0

        t_assign = time.monotonic()
        best_by_session: dict[int, tuple[int, float]] = {}
        for idx, session_id, score in proposals:
            incumbent = best_by_session.get(session_id)
            if incumbent is None or score > incumbent[1]:
                best_by_session[session_id] = (idx, score)
        assignment_ms = (time.monotonic() - t_assign) * 1000

        resolved_idx = {idx for idx, _score in best_by_session.values()}
        winners = {
            rows[idx][0]: session_id
            for session_id, (idx, _score) in best_by_session.items()
        }

        unresolved = [
            row
            for i, row in enumerate(rows)
            if i not in resolved_idx and i not in deferred_indices
        ]
        deferred = {rows[idx][0]: session_id for idx, session_id in deferred_indices.items()}
        return winners, unresolved, deferred, score_ms, assignment_ms

    def _link(
        self,
        raw_id: int,
        session_id: int,
        pending_crops: list[tuple[np.ndarray, np.ndarray]] | None = None,
    ) -> None:
        identity = self._identities[session_id]
        identity.active_raw_id = raw_id
        identity.absence_streak = 0
        # Transfer the winning pending track's buffered crops so an identity
        # that links and then quickly disappears still has last-seen crops at
        # death; respects the identity's own maxlen.
        crop_buffer = deque(maxlen=self._evidence_frames)
        if pending_crops:
            crop_buffer.extend(pending_crops)
        identity.crop_buffer = crop_buffer
        self._raw_to_session[raw_id] = session_id
