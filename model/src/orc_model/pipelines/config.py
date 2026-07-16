"""Typed configuration values for the production detection/tracking pipeline.

The model package defines the shape and validation of each pipeline stage. The
KU Leuven application's actual choices live in ``app/mvp.toml`` so an operator
has one place to tune the complete MVP.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from orc_model.pipelines.persistent_gallery import DEFAULT_INSTRUMENTS_DIR


def _require_probability(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")


@dataclass(frozen=True)
class DetectorConfig:
    confidence: float = 0.5
    top_k: int = 300
    workspace_max_center_y_ratio: float | None = 0.88

    def __post_init__(self) -> None:
        _require_probability("confidence", self.confidence)
        if self.top_k < 1:
            raise ValueError("top_k must be at least 1")
        ratio = self.workspace_max_center_y_ratio
        if ratio is not None and not 0.0 < ratio <= 1.0:
            raise ValueError("workspace_max_center_y_ratio must be in (0, 1] or None")


@dataclass(frozen=True)
class OCSortConfig:
    max_age_seconds: float = 1.0
    min_hits: int = 3
    iou_threshold: float = 0.3
    delta_t: int = 3
    association: str = "iou"
    inertia: float = 0.2
    appearance_weight: float = 0.75
    embedding_momentum: float = 0.95
    adaptive_weight: float = 0.5
    embedding_off: bool = False
    camera_motion_compensation_off: bool = False
    adaptive_weight_off: bool = False
    mask_crop: bool = False

    def __post_init__(self) -> None:
        if self.max_age_seconds <= 0:
            raise ValueError("max_age_seconds must be positive")
        if self.min_hits < 1:
            raise ValueError("min_hits must be at least 1")
        _require_probability("iou_threshold", self.iou_threshold)
        if self.delta_t < 1:
            raise ValueError("delta_t must be at least 1")
        if self.association not in {"iou", "giou", "ciou", "diou", "ct_dist"}:
            raise ValueError("association must be iou, giou, ciou, diou, or ct_dist")


@dataclass(frozen=True)
class MatcherConfig:
    alpha: float = 0.0003
    size_alpha: float = 0.5
    acceptance_threshold: float = 0.30
    winner_margin: float = 0.02
    mask_dilate_px: int = 2
    max_iterations: int = 2000
    embedding_model: str = "facebook/dinov2-base"
    single_gallery_cosine_threshold: float = 0.60

    def __post_init__(self) -> None:
        if self.alpha <= 0:
            raise ValueError("alpha must be positive")
        if self.size_alpha < 0:
            raise ValueError("size_alpha cannot be negative")
        _require_probability("acceptance_threshold", self.acceptance_threshold)
        _require_probability("winner_margin", self.winner_margin)
        _require_probability(
            "single_gallery_cosine_threshold", self.single_gallery_cosine_threshold
        )
        if self.mask_dilate_px < 0:
            raise ValueError("mask_dilate_px cannot be negative")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")
        if not self.embedding_model:
            raise ValueError("embedding_model cannot be empty")


@dataclass(frozen=True)
class LinkerConfig:
    bind_threshold: float = 0.30
    bind_margin: float = 0.02
    unknown_id_offset: int = 1000
    enrolment_window_s: float = 0.5
    evidence_window_s: float = 0.5
    evidence_frames: int = 3
    absent_death_s: float = 1.0
    min_mask_area_px: int = 200
    unknown_recheck_cooldown_s: float = 0.75
    unknown_recheck_appearance_delta: float = 0.15
    unknown_recheck_mask_iou: float = 0.75
    unknown_recheck_quality_gain: float = 0.25
    unknown_recheck_fingerprint_px: int = 32

    def __post_init__(self) -> None:
        for name in (
            "bind_threshold",
            "bind_margin",
            "unknown_recheck_appearance_delta",
            "unknown_recheck_mask_iou",
            "unknown_recheck_quality_gain",
        ):
            _require_probability(name, getattr(self, name))
        if self.unknown_id_offset < 1:
            raise ValueError("unknown_id_offset must be positive")
        if self.enrolment_window_s <= 0 or self.evidence_window_s < 0:
            raise ValueError(
                "linker windows must be non-negative and enrolment must be positive"
            )
        if self.evidence_frames < 1:
            raise ValueError("evidence_frames must be at least 1")
        if self.absent_death_s <= 0:
            raise ValueError("absent_death_s must be positive")
        if self.min_mask_area_px < 1 or self.unknown_recheck_fingerprint_px < 1:
            raise ValueError("linker pixel sizes must be positive")
        if self.unknown_recheck_cooldown_s < 0:
            raise ValueError("unknown_recheck_cooldown_s cannot be negative")


@dataclass(frozen=True)
class TrackerConfig:
    expected_processing_fps: float = 3.0
    instruments_dir: str | Path | None = DEFAULT_INSTRUMENTS_DIR
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    oc_sort: OCSortConfig = field(default_factory=OCSortConfig)
    matcher: MatcherConfig = field(default_factory=MatcherConfig)
    linker: LinkerConfig = field(default_factory=LinkerConfig)

    def __post_init__(self) -> None:
        if self.expected_processing_fps <= 0:
            raise ValueError("expected_processing_fps must be positive")
        if self.linker.absent_death_s < self.oc_sort.max_age_seconds:
            raise ValueError(
                f"absent_death_s ({self.linker.absent_death_s}) must be >= "
                f"max_age_seconds ({self.oc_sort.max_age_seconds})"
            )
