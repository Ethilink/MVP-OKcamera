"""The optional ``--debug`` console (`backend.debug`), wayfinder T09.

Two surfaces, both pinned here: `configure_debug_logging()` attaches exactly one
pretty handler to the `orc_model` logger, off-by-default and idempotent; and
`OrcDebugFormatter` renders the linker's structured `record.orc` events
(freeze / decision / death) as a readable narrative, with a clean fallback for
plain records. The formatter is exercised directly on hand-built `LogRecord`s,
so these tests need no camera, no model, and no threads.
"""

import logging

import pytest

from backend.debug import (
    ORC_LOGGER,
    OrcDebugFormatter,
    configure_debug_logging,
    env_flag_enabled,
)


@pytest.fixture
def clean_orc_logger():
    """Restore the process-global `orc_model` logger after each test."""
    logger = logging.getLogger(ORC_LOGGER)
    before = (list(logger.handlers), logger.level, logger.propagate)
    yield logger
    logger.handlers, logger.level, logger.propagate = (
        before[0], before[1], before[2],
    )
    if hasattr(logger, "_orc_debug_attached"):
        delattr(logger, "_orc_debug_attached")


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, False), ("", False), ("0", False), ("false", False), ("no", False),
        ("1", True), ("true", True), ("TRUE", True), ("yes", True), ("on", True),
        (" 1 ", True),
    ],
)
def test_env_flag_enabled_only_accepts_explicit_true_values(value, expected):
    # ORC_DEBUG=0 / "" / unset must NOT enable the console (a bare truthiness
    # check would have enabled it for any non-empty value, e.g. "0").
    assert env_flag_enabled(value) is expected


def _record(orc=None, msg="", name="orc_model.session_linker"):
    record = logging.LogRecord(name, logging.INFO, __file__, 0, msg, (), None)
    if orc is not None:
        record.orc = orc
    return record


# -- configure_debug_logging ------------------------------------------------


def test_off_by_default_nothing_is_attached_until_configured(clean_orc_logger):
    # The fixture captured a pristine logger; simply not calling configure means
    # no orc-debug handler exists. (main.py only calls it under --debug/ORC_DEBUG.)
    assert not getattr(clean_orc_logger, "_orc_debug_attached", False)


def test_configure_attaches_one_info_handler_and_stops_propagation(clean_orc_logger):
    logger = configure_debug_logging()
    assert logger is clean_orc_logger
    assert logger.level == logging.INFO
    assert logger.propagate is False, (
        "the pretty handler must own orc_model output so uvicorn's root handlers "
        "don't duplicate every line"
    )
    orc_handlers = [h for h in logger.handlers if isinstance(h.formatter, OrcDebugFormatter)]
    assert len(orc_handlers) == 1


def test_configure_is_idempotent_no_double_printing(clean_orc_logger):
    configure_debug_logging()
    n_after_first = len(clean_orc_logger.handlers)
    configure_debug_logging()
    assert len(clean_orc_logger.handlers) == n_after_first, (
        "a second configure must be a no-op — repeated calls must never stack handlers"
    )


# -- OrcDebugFormatter ------------------------------------------------------


def test_freeze_renders_raw_to_specimen_and_flags_a_thin_bind():
    fmt = OrcDebugFormatter()
    out = fmt.format(_record(orc={
        "event": "freeze",
        "build_ms": 495.0,
        "bind_tau": 0.30,
        "roster": [
            {"session_id": 1, "raw_id": 3, "specimen": 1, "score": 0.5587},
            {"session_id": 3, "raw_id": 14, "specimen": 3, "score": 0.3007},  # thin
            {"session_id": 7, "raw_id": 12, "specimen": None, "score": None},  # session-only
        ],
    }))
    assert "ENROLMENT FREEZE" in out
    assert "raw track 3" in out and "specimen 1" in out
    assert "raw track 14" in out, "the raw tracker id must be visible for live diagnosis"
    assert "⚠ thin" in out, "a bind within 0.05 of tau must be flagged thin"
    assert "session-only" in out, "an unbound identity must read session-only, not specimen"
    assert "build 495 ms" in out


def test_decision_renders_link_unknown_and_deferred():
    fmt = OrcDebugFormatter()
    out = fmt.format(_record(orc={
        "event": "decision",
        "outcomes": {63: "linked:1", 12: "unknown", 19: "deferred:7"},
    }))
    assert "raw 63 → Instrument 1" in out and "linked" in out
    assert "raw 12 → Unknown" in out and "rejected" in out
    assert "raw 19 → Instrument 7" in out and "deferred" in out


def test_death_renders_left_the_roster():
    fmt = OrcDebugFormatter()
    out = fmt.format(_record(orc={"event": "death", "session_id": 4}))
    assert "left" in out and "Instrument 4" in out


def test_plain_record_without_orc_falls_back_to_a_clean_prefixed_line():
    fmt = OrcDebugFormatter()
    out = fmt.format(_record(
        msg="persistent galleries: 8 specimen(s) loaded",
        name="orc_model.persistent_gallery",
    ))
    assert "persistent galleries: 8 specimen(s) loaded" in out
    assert "persistent_gallery" in out, "the fallback should label the step by logger tail"


def test_malformed_orc_payload_falls_back_instead_of_raising():
    """A formatter must NEVER raise — logging would dump a traceback into the very
    console being watched at T09. A malformed payload (missing key, wrong type,
    unknown event) falls back to the plain message line."""
    fmt = OrcDebugFormatter()
    malformed = [
        {"event": "death"},                       # missing session_id
        {"event": "freeze", "roster": [{}]},      # roster entry missing every key
        {"event": "decision", "outcomes": None},  # wrong type
        {"event": "totally-unknown-event"},       # unhandled event
    ]
    for bad in malformed:
        out = fmt.format(_record(orc=bad, msg="fallback message"))
        assert "fallback message" in out, (
            f"a malformed payload {bad!r} must fall back to the plain message, never raise"
        )
