"""Optional ``--debug`` console — a readable, event-level pipeline narrative.

The linker already logs its meaningful moments (enrolment freeze, each matcher
decision, a track death) on the ``orc_model.session_linker`` logger, and carries
a structured ``record.orc`` payload for the ones worth pretty-printing. This
module turns those into a step-labelled console narrative for the live camera
test (wayfinder T09).

It is **off by default and behaviour-free**: enabling it only attaches a log
handler. Nothing here runs unless ``orc-demo --debug`` (or ``ORC_DEBUG=1``) is
set, and no linker/decision path changes. Records without an ``orc`` payload
(gallery load, "binding disabled", warnings) still print, via a clean fallback.
"""

from __future__ import annotations

import logging
import sys
from typing import TextIO

ORC_LOGGER = "orc_model"
_ATTACHED_FLAG = "_orc_debug_attached"
_RULE = "─" * 78
_THIN_MARGIN = 0.05  # score-over-tau below this earns a "thin bind" warning


class OrcDebugFormatter(logging.Formatter):
    """Render ``orc_model`` records as a per-event narrative.

    A record with a ``record.orc`` dict (``event`` in {freeze, decision, death})
    is rendered as a readable block; anything else falls back to
    ``● HH:MM:SS  <logger>  <message>``.
    """

    def format(self, record: logging.LogRecord) -> str:
        clock = self.formatTime(record, "%H:%M:%S")
        orc = getattr(record, "orc", None)
        if not isinstance(orc, dict):
            tail = record.name.rsplit(".", 1)[-1]
            return f"●  {clock}  {tail:<10} {record.getMessage()}"
        event = orc.get("event")
        if event == "freeze":
            return self._freeze(clock, orc)
        if event == "death":
            return f"●  {clock}  left     Instrument {orc['session_id']}"
        if event == "decision":
            return self._decision(clock, orc)
        return f"●  {clock}  {record.getMessage()}"

    def _freeze(self, clock: str, orc: dict) -> str:
        lines = [f"{'─' * 24}  ENROLMENT FREEZE  {clock}  {'─' * 24}"]
        tau = orc.get("bind_tau")
        for e in orc.get("roster", []):
            sid, raw, specimen, score = (
                e["session_id"], e["raw_id"], e.get("specimen"), e.get("score"),
            )
            if specimen is not None:
                thin = ""
                if tau is not None and score is not None and score - tau < _THIN_MARGIN:
                    thin = f"   ⚠ thin (+{score - tau:.4f} over τ)"
                score_txt = f"{score:.4f}" if score is not None else "?"
                lines.append(
                    f"     Instrument {sid:<3} ← raw track {raw:<5} specimen {specimen}"
                    f"   score {score_txt}{thin}"
                )
            else:
                lines.append(
                    f"     Instrument {sid:<3} ← raw track {raw:<5} session-only "
                    f"(no confident bind)"
                )
        build_ms = orc.get("build_ms")
        if build_ms is not None:
            lines.append(f"     build {build_ms:.0f} ms")
        lines.append(_RULE)
        return "\n".join(lines)

    def _decision(self, clock: str, orc: dict) -> str:
        lines: list[str] = []
        for raw, outcome in orc.get("outcomes", {}).items():
            if outcome == "unknown":
                lines.append(f"●  {clock}  decide   raw {raw} → Unknown          ✗ rejected")
            elif outcome.startswith("linked:"):
                sid = outcome.split(":", 1)[1]
                lines.append(f"●  {clock}  decide   raw {raw} → Instrument {sid}     ✓ linked")
            elif outcome.startswith("deferred:"):
                sid = outcome.split(":", 1)[1]
                lines.append(
                    f"●  {clock}  decide   raw {raw} → Instrument {sid}     … deferred"
                    f" (re-check pending)"
                )
            else:  # forward-compatible: render any future outcome verbatim
                lines.append(f"●  {clock}  decide   raw {raw} → {outcome}")
        return "\n".join(lines) if lines else f"●  {clock}  decide   (no rows)"


def configure_debug_logging(stream: TextIO | None = None) -> logging.Logger:
    """Attach the pretty console handler to the ``orc_model`` logger.

    Idempotent — a second call is a no-op, so repeated invocation never
    double-prints. Sets the logger to INFO and stops propagation so the pretty
    handler owns ``orc_model`` output (uvicorn's root handlers don't duplicate
    it). Returns the configured logger.
    """
    logger = logging.getLogger(ORC_LOGGER)
    if getattr(logger, _ATTACHED_FLAG, False):
        return logger
    handler = logging.StreamHandler(stream if stream is not None else sys.stderr)
    handler.setFormatter(OrcDebugFormatter())
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    setattr(logger, _ATTACHED_FLAG, True)
    return logger
