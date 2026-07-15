"""SessionLinker — raw tracker ids in, session ids out.

Sits after Deep OC-SORT in `load_tracker()`'s composition. Enrols a frozen
roster at Start, keeps per-identity appearance galleries (session-only, v1),
embeds only at track birth/death/batch-decision events (never per frame), and
re-identifies returning instruments via the injected SRC matcher behind an
open-set gate. See `model/docs/linker-design.md` for the current contract and
rationale.

Out of scope for v1 (see `model/docs/linker-design.md`):
    TODO(linker-design §3): persistent reference galleries + Hungarian binding
        against pre-captured specimen photos. v1 galleries are session-only
        (Start crops + last-seen crops); reference photos do not exist yet.
    TODO(linker-design §5): stability-gated session-refresh ring. Galleries
        are frozen at Start/death; no live crop is ever added to a gallery
        mid-recording.
    TODO(linker-design §8): rotation/mirror gallery augmentation.
        Galleries hold only the raw embedded views, no synthetic copies.
    TODO(linker-design §9): non-blocking/async decision execution. Matcher
        calls happen synchronously inside `update()`; threading is deferred.
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

import numpy as np
import supervision as sv

from orc_model.pipelines.matching import ChampionGallery
from orc_model.pipelines.matching.interface import REJECT  # noqa: F401


class _Crop(NamedTuple):
    rgb: np.ndarray
    mask: np.ndarray
    quality: float  # mask_area * confidence -- ranking only, never admission
    ok: bool  # quality-admitted: big enough mask, not touching the frame edge


@dataclass
class _Identity:
    """A frozen roster identity. `active_raw_id` is None while Missing."""

    session_id: int
    active_raw_id: int | None
    crop_buffer: deque
    start_views: np.ndarray | None = None
    last_seen_views: np.ndarray | None = None
    absence_streak: int = 0


@dataclass
class _PendingTrack:
    """A post-freeze raw id not yet decided (returning instrument vs foreign
    object)."""

    first_seen_frame: int
    crop_buffer: deque
    absence_streak: int = 0
    not_before_frame: int = 0


class SessionLinker:
    """Remaps Deep OC-SORT's raw `tracker_id`s to stable session ids that
    survive absence. See the module docstring and `linker-design.md`."""

    def __init__(
        self,
        matcher,
        *,
        fps: float,
        enrolment_window_s: float = 0.5,
        evidence_window_s: float = 0.5,
        evidence_frames: int = 3,
        absent_death_s: float = 1.0,
        min_mask_area_px: int = 200,
        logger: logging.Logger | None = None,
    ) -> None:
        self._matcher = matcher
        self._evidence_frames = evidence_frames
        self._min_mask_area_px = min_mask_area_px
        self._enrolment_window_frames = max(1, round(enrolment_window_s * fps))
        self._evidence_window_frames = round(evidence_window_s * fps)
        self._death_threshold_frames = round(absent_death_s * fps)
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
        self._settled_unknown: set[int] = set()

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
            return raw_id

        pending = self._pending.get(raw_id)
        if pending is None:
            pending = _PendingTrack(
                first_seen_frame=self._frame_count,
                crop_buffer=deque(maxlen=self._evidence_frames),
            )
            self._pending[raw_id] = pending
        pending.absence_streak = 0
        if crop is not None and crop.ok:
            pending.crop_buffer.append((crop.rgb, crop.mask))
        return raw_id

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
        return detections[np.arange(len(detections))]

    def _freeze_roster(self) -> None:
        majority = self._enrolment_frames_seen / 2
        roster_ids = sorted(
            raw_id for raw_id, count in self._enrolment_presence.items() if count > majority
        )

        t0 = time.monotonic()
        identities: dict[int, _Identity] = {}
        view_counts: dict[int, int] = {}
        for raw_id in roster_ids:
            best_crops = sorted(
                self._enrolment_crops.get(raw_id, []), key=lambda c: c.quality, reverse=True
            )[:3]
            start_views = None
            if best_crops:
                gallery = self._matcher.build_gallery(
                    [c.rgb for c in best_crops], [c.mask for c in best_crops],
                    {"identity": raw_id},
                )
                start_views = gallery.views
            identities[raw_id] = _Identity(
                session_id=raw_id,
                active_raw_id=raw_id,
                crop_buffer=deque(maxlen=self._evidence_frames),
                start_views=start_views,
            )
            self._raw_to_session[raw_id] = raw_id
            view_counts[raw_id] = 0 if start_views is None else len(start_views)
        build_ms = (time.monotonic() - t0) * 1000

        self._identities = identities
        self._roster = frozenset(roster_ids)
        self._enrolled = True
        self._logger.info(
            "enrolment freeze: roster_size=%d views=%s build_ms=%.1f",
            len(roster_ids), view_counts, build_ms,
        )

    # -- B3: death (Active -> Missing) ------------------------------------

    def _age_out_missing_actives(self, present: set[int]) -> None:
        for identity in self._identities.values():
            if identity.active_raw_id is None or identity.active_raw_id in present:
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

    # -- B5: batched, gated decision ----------------------------------------

    def _comparison_galleries(self) -> dict[int, ChampionGallery]:
        """Return every usable frozen-roster gallery.

        Gallery membership is deliberately independent of Active/Missing
        state.  SRC's SCI gate is a relative statistic calibrated against the
        complete roster; removing Active identities would collapse the normal
        one-missing case to K=1 and destroy its open-set rejection signal.
        """
        galleries: dict[int, ChampionGallery] = {}
        for session_id, identity in self._identities.items():
            views = [v for v in (identity.start_views, identity.last_seen_views) if v is not None]
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
        for raw_id, pending in closed:
            crop_tuples = list(pending.crop_buffer)
            crops = [c for c, _ in crop_tuples]
            masks = [m for _, m in crop_tuples]
            if not crops:
                resolved[raw_id] = None
                self._settled_unknown.add(raw_id)
                continue
            rows.append((raw_id, crops, masks))
            pending_crops_by_raw_id[raw_id] = crop_tuples

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
            self._settled_unknown.add(raw_id)
            resolved[raw_id] = None
        pending_by_raw_id = dict(closed)
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
        self._logger.info(
            "batch decision: rows=%d comparison_galleries=%d eligible_missing=%d "
            "score_ms=%.1f collision_ms=%.1f total_ms=%.1f outcomes=%s",
            len(rows), len(comparison_galleries), len(eligible_missing_ids),
            score_ms, assignment_ms, total_ms,
            {
                **{
                    raw_id: ("unknown" if sid is None else f"linked:{sid}")
                    for raw_id, sid in resolved.items()
                },
                **{raw_id: f"deferred:{sid}" for raw_id, sid in deferred.items()},
            },
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
