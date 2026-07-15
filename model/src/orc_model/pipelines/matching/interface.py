#!/usr/bin/env python
"""Stable matcher interface used by the session linker.

The contract mirrors `model/docs/linker-design.md`:

    build_gallery(crops, masks, meta) -> Gallery
    score(query_crops, query_masks, meta, galleries) -> {candidate: score}
    accept(scores) -> decision | REJECT

The implementation is deliberately self-contained so production code has no
dependency on a research workspace.
"""
from __future__ import annotations

REJECT = "REJECT"


class Gallery:
    """Marker base class for whatever build_gallery() returns — opaque to the
    caller, only ever passed back into this same module's score()."""
