"""The seam between the model (RF-DETR detector + tracker) and its consumers.

A *consumer* (the data-collection dashboard, later the demo backend) depends ONLY
on the `InstrumentTracker` interface below — never on the detector internals. The
producer is free to retrain the detector, swap the ONNX export, retune the tracker,
etc.; as long as `update()` returns the `sv.Detections` shape documented here,
nothing reaches the consumer.

See `docs/tracker-interface.md` for the full contract and rationale.
"""

from __future__ import annotations

import hashlib
import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
import supervision as sv

from orc_model.pipelines.persistent_gallery import (
    DEFAULT_INSTRUMENTS_DIR,
    load_persistent_galleries,
)

if TYPE_CHECKING:
    from orc_model.components.detector.detector import Detector
    from orc_model.pipelines.config import OCSortConfig, TrackerConfig
    from orc_model.pipelines.deep_ocsort.tracker import DeepOCSortTracker
    from orc_model.pipelines.session_linker import MatchDebug
    from orc_model.pipelines.session_linker import SessionLinker

_log = logging.getLogger("orc_model.tracking")

DEFAULT_PROCESSING_FPS = 3.0
DEFAULT_WORKSPACE_MAX_CENTER_Y_RATIO = 0.88


@runtime_checkable
class InstrumentTracker(Protocol):
    """Stateful, one instance per recording. Feed frames in capture order.

    `update(frame)` returns the tracked detections for THIS frame as an
    `sv.Detections` with these fields guaranteed populated:

        xyxy              float32 (N, 4)    pixel coords in the frame's OWN space
        confidence        float32 (N,)      per detection
        class_id          int     (N,)      key into `class_names`
        tracker_id        int     (N,)      stable & unique per instance this session
        mask              bool    (N, H, W) full-frame instance masks
        data["resolving"] bool    (N,)      True while the linker is still
                                            deciding this track's identity
                                            (pending in its evidence window, or
                                            deferred); False for a settled roster
                                            id and for a settled Unknown.

    `data["resolving"]` splits the offset (not-in-roster) id range into the two
    states a consumer draws differently: a *resolving* track shows the
    "resolving" spinner, a settled one the "Unknown" label. It is driven by the
    linker's actual decision, not a consumer-side timer, so the spinner can never
    disagree with when the identity is really settled. (Seam widening grilled
    with Bram 2026-07-16 — the wait-state fix; see `tracker-interface.md`.)

    Treat `frame` as read-only: do not mutate its pixels or retain a mutable
    reference. Every output field is row-aligned for this exact input frame;
    boxes contain finite, ordered coordinates in the frame's pixel space and
    may extend outside its boundary (consumers clamp before pixel indexing).

    Only detections at or above `confidence` are returned, and every returned
    detection carries a real `tracker_id` (there is no untracked/sentinel case).
    A frame with nothing to report returns `sv.Detections.empty()`. `update()`
    never returns None and never raises on a valid frame.
    """

    confidence: float
    """Read/write. The live detection threshold — set at startup, changeable
    between frames (e.g. from the dashboard's slider). Only detections >= this
    are returned. Changing it mid-recording may perturb track ids; that's fine,
    each captured frame is an independent image."""

    def update(self, frame: np.ndarray) -> sv.Detections:
        """One read-only BGR frame (H, W, 3 uint8) -> detections for THIS frame."""
        ...

    def reset(self) -> None:
        """Drop all track state. Called once when a new recording starts, so
        each recording is its own `tracker_id` namespace."""
        ...

    @property
    def class_names(self) -> dict[int, str]:
        """Maps `class_id` -> label, e.g. `{0: "surgical_instrument"}`. The
        consumer builds its category table from this."""
        ...

    @property
    def model_version(self) -> str:
        """Provenance string (weights tag/hash). The consumer stashes it with
        every saved frame so a dataset is traceable to the model that made it."""
        ...

    @property
    def roster(self) -> frozenset[int]:
        """Frozen Start roster of session ids. Empty before the enrolment
        freeze and immediately after reset(). (Seam widening grilled with
        Bram 2026-07-15 — wayfinder T10 route (b).)"""
        ...

    @property
    def catalog(self) -> frozenset[int]:
        """Persistent specimen IDs available for catalog binding.

        Constant for the tracker lifetime and preserved across reset(). Unlike
        `roster` (empty until the enrolment freeze), `catalog` is known from
        construction. In catalog-only mode `roster` is always a subset of
        `catalog`.
        """
        ...

    @property
    def match_debug(self) -> dict[int, "MatchDebug"]:
        """Experimental (feat/matching-tests): last matcher score per emitted
        id. A testing aid, not part of the model seam's real contract -- a
        tracker with no real matcher (the fake) may just return `{}`."""
        ...


class FakeInstrumentTracker:
    """A dependency-free stand-in that honours the `InstrumentTracker` contract,
    so the dashboard can be built and demoed before the real weights exist.

    Emits a few instruments that drift smoothly across the frame with
    oscillating confidence — so some flicker below the threshold (exercising the
    "instrument the model missed" case) and occasionally a frame comes back
    empty (a false negative). Deterministic: same frame count -> same output.
    """

    def __init__(self, confidence: float = 0.5, n_instruments: int = 3) -> None:
        self.confidence = confidence
        self.n_instruments = n_instruments
        self._frame = 0

    @property
    def class_names(self) -> dict[int, str]:
        return {0: "surgical_instrument"}

    @property
    def model_version(self) -> str:
        return "fake-0.1"

    @property
    def roster(self) -> frozenset[int]:
        # 1-based, matching the app fake and the real linker's specimen-number
        # session ids; the fake enrols instantly.
        return frozenset(range(1, self.n_instruments + 1))

    @property
    def catalog(self) -> frozenset[int]:
        # The fake's full identity range is its catalog: constant, independent
        # of `_frame`, and identical to `roster` (it enrols everything).
        return frozenset(range(1, self.n_instruments + 1))

    @property
    def match_debug(self) -> dict[int, "MatchDebug"]:
        # No real matcher runs here -- deliberately blank (feat/matching-tests).
        return {}

    def reset(self) -> None:
        self._frame = 0

    def update(self, frame: np.ndarray) -> sv.Detections:
        height, width = frame.shape[:2]
        t = self._frame
        self._frame += 1

        boxes: list[list[float]] = []
        masks: list[np.ndarray] = []
        confidences: list[float] = []
        tracker_ids: list[int] = []

        for i in range(self.n_instruments):
            phase = t * 0.05 + i
            cx = (0.2 + 0.6 * (i + 1) / (self.n_instruments + 1)) * width
            cx += math.sin(phase) * 0.03 * width
            cy = (0.5 + 0.2 * math.sin(phase * 0.7 + i)) * height
            box_w, box_h = 0.12 * width, 0.10 * height

            x1 = max(0.0, cx - box_w / 2)
            y1 = max(0.0, cy - box_h / 2)
            x2 = min(float(width), cx + box_w / 2)
            y2 = min(float(height), cy + box_h / 2)

            confidence = 0.55 + 0.40 * math.sin(phase * 1.3 + i * 2.0)
            confidence = min(0.99, max(0.01, confidence))
            if confidence < self.confidence:
                continue  # flickered below threshold -> simulates a miss

            mask = np.zeros((height, width), dtype=bool)
            mask[int(y1) : int(y2), int(x1) : int(x2)] = True

            boxes.append([x1, y1, x2, y2])
            masks.append(mask)
            confidences.append(confidence)
            tracker_ids.append(i + 1)  # 1-based emitted id, matching roster/catalog

        if not boxes:
            return sv.Detections.empty()

        return sv.Detections(
            xyxy=np.array(boxes, dtype=np.float32),
            mask=np.stack(masks),
            confidence=np.array(confidences, dtype=np.float32),
            class_id=np.zeros(len(boxes), dtype=int),
            tracker_id=np.array(tracker_ids, dtype=int),
            data={
                "class_name": np.array(["surgical_instrument"] * len(boxes)),
                # The fake enrols instantly and never re-identifies, so every
                # emitted track is a settled roster id -- nothing is resolving.
                "resolving": np.zeros(len(boxes), dtype=bool),
            },
        )


def load_tracker(
    weights_path: str | Path,
    confidence: float = 0.5,
    *,
    fps: float = DEFAULT_PROCESSING_FPS,
    max_age_seconds: float = 1.0,
    absent_death_s: float | None = None,
    workspace_max_center_y_ratio: float | None = DEFAULT_WORKSPACE_MAX_CENTER_Y_RATIO,
    instruments_dir: str | Path | None = DEFAULT_INSTRUMENTS_DIR,
    config: TrackerConfig | None = None,
) -> InstrumentTracker:
    """Compose the real `InstrumentTracker`: RF-DETR detector -> Deep OC-SORT
    -> SessionLinker, per `model/docs/linker-design.md` "Where it composes".
    The app passes the typed configuration loaded from ``app/mvp.toml``.

    `weights_path` is the only required argument -- every other knob has a
    workable default, so `load_tracker(weights_path)` alone still works.
    The KU Leuven app passes its complete, documented ``TrackerConfig`` from
    ``backend.mvp_settings`` instead; when ``config`` is supplied, it is the
    source of truth and the legacy individual keyword arguments are ignored.

    `fps` is the measured processing rate, not the camera capture rate. The
    final M3 Max pipeline measured 2.87-3.13 fps on the two demo recordings,
    so the live default is 3.0. NEVER let this default drift to 30 --
    `DeepOCSortTracker.frame_rate`
    defaults to 30 internally, which is exactly why `fps` is threaded through
    explicitly to both the tracker and the linker below: a
    wrong fps here silently stretches the tracker's "1.0 s" coast window to
    ~2.5 s of wall time at real fps, breaching the tracker/linker boundary
    described in linker-design.md §9.

    `max_age_seconds` is Deep OC-SORT's coasting window before it frees a raw
    id. `absent_death_s` is the linker's window before it declares a
    *session* identity Missing. These must satisfy
    `absent_death_s >= max_age_seconds` -- otherwise the linker could bury an
    identity while the tracker is still coasting its raw id, and a coasted
    reactivation would silently resurrect a "dead" identity's raw id without
    ever crossing the linker's re-id gate. By default `absent_death_s` is
    pinned to `max_age_seconds`; pass it explicitly only to widen it further,
    a narrower value raises `ValueError`.

    `workspace_max_center_y_ratio` excludes detections whose box centre falls
    below the fixed overhead workspace. The demo camera sees tripod/table
    clutter below the blue mat; 0.88 keeps that clutter out of the frozen
    roster while retaining the instrument row. Pass `None` for an uncropped
    installation or calibrate the ratio when the camera framing changes.

    `instruments_dir` holds the pre-captured specimen photos the linker binds
    enrolled identities to (`instrument{N}/images` + COCO annotations). They are
    loaded and embedded ONCE here, at startup, and cached for the process's
    life -- embedding ~15 views per specimen per enrolment would push the
    freeze from ~260 ms toward seconds. Pass `None` to disable binding
    (everything links session-only); in the default (non-catalog) mode a missing
    or empty directory logs and degrades the same way, never raises. Under
    `catalog_only_enrolment` that same missing/empty/wrong-sized/id-colliding
    catalog is instead a fatal config error and `load_tracker` raises
    (fail-closed).

    Constructs the SRC matcher (`ChampionMethod`) eagerly, which loads
    DINOv2-B from the local HF cache -- so this call fails fast at startup on
    a missing/mismatched cache instead of on the first tracked frame.
    """
    from orc_model.pipelines.config import (
        DetectorConfig,
        LinkerConfig,
        MatcherConfig,
        OCSortConfig,
        TrackerConfig,
    )

    if config is None:
        resolved_absent_death_s = (
            max_age_seconds if absent_death_s is None else absent_death_s
        )
        config = TrackerConfig(
            expected_processing_fps=fps,
            instruments_dir=instruments_dir,
            detector=DetectorConfig(
                confidence=confidence,
                workspace_max_center_y_ratio=workspace_max_center_y_ratio,
            ),
            oc_sort=OCSortConfig(max_age_seconds=max_age_seconds),
            matcher=MatcherConfig(),
            linker=LinkerConfig(absent_death_s=resolved_absent_death_s),
        )

    detector_config = config.detector
    oc_sort_config = config.oc_sort
    matcher_config = config.matcher
    linker_config = config.linker

    if linker_config.absent_death_s < oc_sort_config.max_age_seconds:
        raise ValueError(
            f"absent_death_s ({linker_config.absent_death_s}) must be >= max_age_seconds "
            f"({oc_sort_config.max_age_seconds}): the linker must never declare an identity "
            "dead before the tracker itself has given up coasting its raw id."
        )

    weights_path = Path(weights_path)
    detector = _build_detector(
        weights_path,
        detector_config.confidence,
        detector_config.top_k,
    )
    from orc_model.pipelines.matching import ChampionMethod
    from orc_model.pipelines.session_linker import SessionLinker

    matcher = ChampionMethod(
        alpha=matcher_config.alpha,
        size_alpha=matcher_config.size_alpha,
        tau=matcher_config.acceptance_threshold,
        margin=matcher_config.winner_margin,
        mask_dilate_px=matcher_config.mask_dilate_px,
        max_iter=matcher_config.max_iterations,
        model_id=matcher_config.embedding_model,
        cos_tau=matcher_config.single_gallery_cosine_threshold,
        active_features=matcher_config.active_features,
    )  # loads DINOv2-B eagerly -- fail fast, not on first frame
    galleries = None
    # In catalog-only mode a missing/empty gallery set is a fatal config error
    # (gate below), so don't emit the misleading "linking session-only" INFO --
    # we are about to refuse exactly that fallback.
    catalog_only = linker_config.catalog_only_enrolment
    if config.instruments_dir is None:
        if not catalog_only:
            _log.info("instruments_dir=None -- specimen binding disabled, linking session-only")
    else:
        # Embed the specimen photos ONCE, here, and hand the vectors to the
        # linker; reset() must never re-load or re-embed them.
        galleries = load_persistent_galleries(matcher, config.instruments_dir) or None
        if galleries is None and not catalog_only:
            _log.info(
                "no persistent galleries under %s -- linking session-only",
                config.instruments_dir,
            )
    if catalog_only:
        # Fail-closed gate: catalog-only mode is a SAFETY promise (only loaded
        # specimens may join the roster), so it must refuse both to fall back to
        # the session-only behaviour it exists to replace and to accept any
        # catalog that could corrupt the "Unknown = not in roster" test. All
        # checks need the actually-loaded galleries, hence here and not in
        # `__post_init__`.
        loaded = 0 if galleries is None else len(galleries)
        if galleries is None:
            where = (
                "instruments_dir is not configured"
                if config.instruments_dir is None
                else f"none loaded from {config.instruments_dir!r}"
            )
            raise ValueError(
                f"catalog_only_enrolment is set but no persistent galleries loaded "
                f"({where}); refusing session-only fallback"
            )
        expected = linker_config.expected_catalog_size
        if expected is not None and loaded != expected:
            raise ValueError(
                f"catalog_only_enrolment expected {expected} galleries under "
                f"{config.instruments_dir!r} but loaded {loaded}"
            )
        # The offset keeps roster ids and Unknown/offset ids disjoint, which is
        # what makes "Unknown = not in roster" exact (tracker-interface.md, "Two
        # disjoint emitted ID ranges"). A specimen id at/above the offset would
        # collide: an unbound raw id `k` emits `k + offset`, so a roster id
        # >= offset could not be told apart from a foreign track's Unknown id.
        # In the safety mode the catalog is known at startup, so reject it now.
        offset = linker_config.unknown_id_offset
        colliding = sorted(sid for sid in galleries if sid >= offset)
        if colliding:
            raise ValueError(
                f"catalog_only_enrolment specimen ids {colliding} under "
                f"{config.instruments_dir!r} reach unknown_id_offset={offset}; "
                f"roster and Unknown id ranges would overlap"
            )
    session_linker = SessionLinker(
        matcher,
        fps=config.expected_processing_fps,
        persistent_galleries=galleries,
        catalog_only_enrolment=linker_config.catalog_only_enrolment,
        bind_tau=linker_config.bind_threshold,
        bind_margin=linker_config.bind_margin,
        unknown_id_offset=linker_config.unknown_id_offset,
        enrolment_window_s=linker_config.enrolment_window_s,
        evidence_window_s=linker_config.evidence_window_s,
        evidence_frames=linker_config.evidence_frames,
        absent_death_s=linker_config.absent_death_s,
        min_mask_area_px=linker_config.min_mask_area_px,
        unknown_recheck_cooldown_s=linker_config.unknown_recheck_cooldown_s,
        unknown_recheck_appearance_delta=linker_config.unknown_recheck_appearance_delta,
        unknown_recheck_mask_iou=linker_config.unknown_recheck_mask_iou,
        unknown_recheck_quality_gain=linker_config.unknown_recheck_quality_gain,
        unknown_recheck_fingerprint_px=linker_config.unknown_recheck_fingerprint_px,
    )

    return _RealInstrumentTracker(
        detector=detector,
        confidence=detector_config.confidence,
        fps=config.expected_processing_fps,
        max_age_seconds=oc_sort_config.max_age_seconds,
        oc_sort_config=oc_sort_config,
        session_linker=session_linker,
        model_version=_model_version(weights_path),
        workspace_max_center_y_ratio=detector_config.workspace_max_center_y_ratio,
    )


def _build_detector(weights_path: Path, confidence: float, top_k: int = 300) -> Detector:
    """The detector on the CoreML EP (Apple Silicon GPU/Neural Engine), falling
    back to plain CPU wherever CoreML isn't available.

    Measured on the M3 Max, 1920x1080 in: **0.33 s/frame (3.1 fps) vs 0.84 s
    (1.2 fps)** CPU-only -- a 2.6x speedup with BIT-IDENTICAL output (box delta
    0.000, confidence delta 0.0000). Since `update()` is inference-bound, that
    factor lands directly on the whole seam's frame rate, so this is not an
    optional nicety: on CPU the tracker cannot keep up with a demo.

    This mirrors what `data-collection/dashboard/backend/main.py` has done
    since commit 89cfbbc ("perf(detector): run ORC detector on CoreML EP");
    `load_tracker()` was composed without it and silently regressed the demo
    path back to CPU. The options are load-bearing and were each paid for
    once already -- do not "simplify" them away:

    - `ModelFormat=MLProgram` is MANDATORY: the NeuralNetwork format throws
      GatherElements-out-of-range on this graph. It is also what makes CoreML
      absorb the graph -- 17 partitions instead of 119 (1166/1232 nodes), which
      IS the speedup. Plain `["CoreMLExecutionProvider"]` with default options
      is *slower than CPU* (2.8 s/frame), so a half-configured CoreML EP is
      worse than none.
    - `MLComputeUnits=ALL` picks the GPU correctly; don't force
      CPUAndNeuralEngine.
    - `RequireStaticInputShapes=1` is safe here: `preprocess()` always emits a
      fixed (1, 3, 768, 768) tensor regardless of camera resolution.
    - `ModelCacheDirectory` persists the compile cache next to the weights,
      cutting session build from ~49 s to ~10 s on warm restarts. Stale entries
      are keyed by graph hash, so a re-exported ONNX just builds a new one.
    """
    import onnxruntime

    from orc_model.components.detector.detector import Detector

    if "CoreMLExecutionProvider" not in onnxruntime.get_available_providers():
        return Detector(weights_path, confidence_threshold=confidence, top_k=top_k)

    cache_dir = weights_path.parent / ".coreml_cache"
    cache_dir.mkdir(exist_ok=True)
    return Detector(
        weights_path,
        confidence_threshold=confidence,
        top_k=top_k,
        providers=["CoreMLExecutionProvider", "CPUExecutionProvider"],
        provider_options=[
            {
                "ModelFormat": "MLProgram",
                "MLComputeUnits": "ALL",
                "RequireStaticInputShapes": "1",
                "ModelCacheDirectory": str(cache_dir),
            },
            {},
        ],
    )


def _model_version(weights_path: Path) -> str:
    """Weights filename stem + first 8 hex chars of the file's sha256 -- a
    cheap one-time provenance stamp, hashed once at load."""
    digest = hashlib.sha256(weights_path.read_bytes()).hexdigest()[:8]
    return f"{weights_path.stem}-{digest}"


class _RealInstrumentTracker:
    """The real `InstrumentTracker`: RF-DETR detector -> Deep OC-SORT ->
    SessionLinker. Built by `load_tracker()`; not meant to be constructed
    directly.

    `update()` runs the three-stage composition from
    `model/docs/linker-design.md` "Where it composes" every call. `reset()`
    re-creates the Deep OC-SORT tracker (it has no `reset()` of its own -- a
    fresh instance IS the reset) and resets the linker's session state, but
    keeps the already-loaded detector session and matcher/DINOv2-B model, so
    repeated recordings don't pay the load cost twice.
    """

    def __init__(
        self,
        *,
        detector: Detector,
        confidence: float,
        fps: float,
        max_age_seconds: float,
        oc_sort_config: OCSortConfig | None = None,
        session_linker: SessionLinker,
        model_version: str,
        workspace_max_center_y_ratio: float | None,
    ) -> None:
        self.confidence = confidence
        self._detector = detector
        self._fps = fps
        if oc_sort_config is None:
            from orc_model.pipelines.config import OCSortConfig

            oc_sort_config = OCSortConfig(max_age_seconds=max_age_seconds)
        self._oc_sort_config = oc_sort_config
        self._session_linker = session_linker
        self._model_version = model_version
        self._workspace_max_center_y_ratio = workspace_max_center_y_ratio
        self._deep_ocsort = self._new_deep_ocsort()

    def _new_deep_ocsort(self) -> DeepOCSortTracker:
        # det_thresh stays fixed at the startup confidence:
        # OC-SORT's internal gate, distinct from the per-call detector
        # threshold below, which does track `self.confidence` live.
        from orc_model.pipelines.deep_ocsort.tracker import DeepOCSortTracker

        return DeepOCSortTracker(
            det_thresh=self.confidence,
            frame_rate=self._fps,
            max_age_seconds=self._oc_sort_config.max_age_seconds,
            min_hits=self._oc_sort_config.min_hits,
            iou_threshold=self._oc_sort_config.iou_threshold,
            delta_t=self._oc_sort_config.delta_t,
            association=self._oc_sort_config.association,
            inertia=self._oc_sort_config.inertia,
            appearance_weight=self._oc_sort_config.appearance_weight,
            embedding_momentum=self._oc_sort_config.embedding_momentum,
            adaptive_weight=self._oc_sort_config.adaptive_weight,
            embedding_off=self._oc_sort_config.embedding_off,
            camera_motion_compensation_off=(
                self._oc_sort_config.camera_motion_compensation_off
            ),
            adaptive_weight_off=self._oc_sort_config.adaptive_weight_off,
            mask_crop=self._oc_sort_config.mask_crop,
        )

    @property
    def class_names(self) -> dict[int, str]:
        return {0: "surgical_instrument"}

    @property
    def model_version(self) -> str:
        return self._model_version

    @property
    def roster(self) -> frozenset[int]:
        return self._session_linker.roster

    @property
    def catalog(self) -> frozenset[int]:
        return self._session_linker.catalog

    @property
    def match_debug(self) -> dict[int, "MatchDebug"]:
        return self._session_linker.match_debug

    def reset(self) -> None:
        self._deep_ocsort = self._new_deep_ocsort()
        self._session_linker.reset()

    def update(self, frame: np.ndarray) -> sv.Detections:
        detections = self._detector.predict(frame, confidence_threshold=self.confidence)
        detections = _filter_workspace(
            detections,
            frame_height=frame.shape[0],
            max_center_y_ratio=self._workspace_max_center_y_ratio,
        )
        detections = self._deep_ocsort.update(detections, frame)
        detections = self._session_linker.update(detections, frame)
        return detections


def _filter_workspace(
    detections: sv.Detections,
    *,
    frame_height: int,
    max_center_y_ratio: float | None,
) -> sv.Detections:
    """Keep detections whose box centre lies inside the overhead workspace."""
    if max_center_y_ratio is None:
        return detections
    if not 0.0 < max_center_y_ratio <= 1.0:
        raise ValueError("workspace_max_center_y_ratio must be in (0, 1] or None")
    if len(detections) == 0:
        return detections

    center_y = (detections.xyxy[:, 1] + detections.xyxy[:, 3]) / 2.0
    return detections[center_y <= frame_height * max_center_y_ratio]
