#!/usr/bin/env python
"""
The fixed method interface this module implements — mirrors
`linker-design.md` §6 ("Match rule — open-set, multi-candidate") and
`../frozen/interface.py` in the research sandbox this champion came from:

    build_gallery(crops, masks, meta) -> Gallery
    score(query_crops, query_masks, meta, galleries) -> {candidate: score}
    accept(scores) -> decision | REJECT

This copy is standalone on purpose (no import of the experiment sandbox's
`frozen`/`method` packages) so `champion/` can be dropped straight into the
linker build per program.md §9.
"""
from __future__ import annotations

REJECT = "REJECT"


class Gallery:
    """Marker base class for whatever build_gallery() returns — opaque to the
    caller, only ever passed back into this same module's score()."""
