#!/usr/bin/env python
"""
FROZEN — the fixed method interface. NEVER edit this file from a method/ variant
or a runs/ experiment. If a challenger needs the interface to change, that is a
finding for FINDINGS.md, not a code change here.

Mirrors linker-design.md §6 ("Match rule — open-set, multi-candidate") and
program.md §3:

    build_gallery(crops, masks, meta) -> Gallery
    score(query_crops, query_masks, meta, galleries) -> {candidate: score}
    accept(scores) -> decision | REJECT

Contract (the eval enforces this, methods must honour it):
  - `crops` are RGB arrays (H, W, 3) uint8, bbox-cropped, UNMASKED pixels.
    `masks` are (H, W) binary arrays, 1 = instrument pixel, aligned to `crops`.
    Masking/dilation/canonicalization strategy is the METHOD's choice — the
    frozen loader hands over raw pixels + raw masks, never a pre-masked crop.
  - `meta` passed into build_gallery MAY carry the identity being enrolled
    (that's not a leak — you always know whose gallery you're building) plus
    non-identity bookkeeping (frame ids).
  - `meta` passed into score() / the crops given to score() NEVER carry the
    true identity of the query. A method that reads a query label anywhere is
    disqualified regardless of its metric.
  - `galleries` is a dict[candidate_id -> Gallery]; a method may only score
    against what's in that dict — it must not read galleries for identities
    outside the current candidate set (Missing set), and must not read the
    locked holdout split from inside score()/accept()/build_gallery().
  - `Gallery` is opaque to the eval: whatever a method's build_gallery returns
    is only ever handed back into that same method's score(). The eval never
    inspects its internals.
  - REJECT is the sentinel string below; accept() must return either a key
    present in `scores` or exactly this sentinel.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

REJECT = "REJECT"


class Gallery:
    """Marker base class for whatever a method's build_gallery returns. Purely
    a typing aid — the eval treats every Gallery as opaque."""


@runtime_checkable
class MatcherMethod(Protocol):
    """The fixed plug every method/ variant and champion/ module must implement."""

    family: str  # one-line family name, for TRIED.md / LEADERBOARD.md bookkeeping

    def build_gallery(self, crops: list, masks: list, meta: dict) -> Gallery:
        """Build the reference gallery for ONE identity from its enrolment crops."""
        ...

    def score(self, query_crops: list, query_masks: list, meta: dict,
              galleries: dict[str, Gallery]) -> dict[str, float]:
        """Score a query event (one or more frames = a return-window) against
        every candidate gallery. Returns {candidate_id: score}. Higher = more
        confident match. Must not read any ground-truth identity."""
        ...

    def accept(self, scores: dict[str, float]) -> str:
        """Turn a score dict into a final decision: a candidate_id from
        `scores`, or REJECT. Encapsulates threshold + margin + (if the method
        does it) multi-frame-consistency logic."""
        ...


def check_decision(decision: str, scores: dict[str, float]) -> None:
    """Frozen-eval-side sanity check: a method's accept() must return either
    REJECT or a key that was actually offered to it. Raises on violation so a
    method can't silently invent a candidate id."""
    if decision != REJECT and decision not in scores:
        raise ValueError(
            f"accept() returned {decision!r}, which is not REJECT and not one "
            f"of the candidates it was given ({sorted(scores)}). This is a "
            f"contract violation, not a valid decision."
        )
