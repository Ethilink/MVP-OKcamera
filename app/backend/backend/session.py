"""Pure session state machine for debounce, usage windows, and reporting.

It is fed ``(t, present_ids)`` and owns no clock or threads (DESIGN D11).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class Phase(StrEnum):
    SETUP = "setup"
    RECORDING = "recording"
    FINISHED = "finished"


@dataclass(frozen=True)
class InstrumentStatus:
    tracker_id: int
    label: str
    on_table: bool


@dataclass(frozen=True)
class UsageWindow:
    off_s: float
    on_s: float | None


@dataclass(frozen=True)
class InstrumentReport:
    tracker_id: int
    label: str
    completeness: str
    usage: tuple[UsageWindow, ...]


@dataclass(frozen=True)
class Report:
    duration_s: float
    instruments: tuple[InstrumentReport, ...]


@dataclass(frozen=True)
class SetupReadiness:
    """Immutable, clock-free verdict on whether setup is ready to start
    recording, computed purely from the latest same-tick observation (T11/B2).
    Capture health is NOT folded in here — the API combines it (D3 cond 1)."""

    detected_count: int
    expected_count: int
    recognised_count: int
    resolving_count: int
    unknown_count: int
    stable_for_s: float
    ready: bool
    blocking_reason: str | None   # "recognising" | "missing_instruments" |
                                  # "unknown_objects" | "hold_steady" | None


class InvalidPhase(RuntimeError):
    """Raised by a method call that isn't legal in the session's current
    phase."""


@dataclass
class _Track:
    """Debounce state for one `tracker_id`, from the raw per-frame presence
    signal up to the confirmed (debounced) on/off state. `confirmed` means
    the id has joined the instrument universe (entry debounce passed)."""

    confirmed: bool = False
    on_table: bool = False
    run_present: bool = False
    first_present_t: float = 0.0   # start of the current raw presence run
    last_present_t: float = 0.0    # t of the most recent raw-present frame
    usage: list[UsageWindow] = field(default_factory=list)


def _project(track: _Track, t: float, on_debounce_s: float, off_debounce_s: float) -> tuple[bool, bool]:
    """Pure projection: what (confirmed, on_table) WOULD be at `t`, given the
    track's durable state and its current raw run — without mutating it."""
    if track.run_present:
        duration = t - track.first_present_t
        if not track.confirmed:
            entered = duration > on_debounce_s
            return entered, entered
        if not track.on_table:
            return True, duration > on_debounce_s
        return True, True
    if not track.confirmed:
        return False, False
    if track.on_table:
        return True, not (t - track.last_present_t > off_debounce_s)
    return True, False


class Session:
    def __init__(
        self,
        off_debounce_s: float = 1.5,
        on_debounce_s: float = 1.0,
        setup_stable_s: float = 2.0,
    ) -> None:
        self._off_debounce_s = off_debounce_s
        self._on_debounce_s = on_debounce_s
        # D3 condition 5: the id-set must be unchanged for at least this long
        # before setup counts as stable enough to start.
        self._setup_stable_s = setup_stable_s
        self._phase = Phase.SETUP
        self._last_t: float | None = None
        self._idset: frozenset[int] | None = None
        self._idset_since_t: float = 0.0
        self._start_t: float = 0.0
        self._tracks: dict[int, _Track] = {}
        self._report: Report | None = None
        # Latest same-tick setup observation, stored so setup_readiness can be a
        # pure function of state (no tracker, no clock). None present => nothing
        # observed yet (readiness reports "recognising").
        self._setup_present: frozenset[int] | None = None
        self._setup_roster: frozenset[int] = frozenset()
        self._setup_catalog: frozenset[int] = frozenset()
        self._setup_resolving: frozenset[int] = frozenset()

    @property
    def phase(self) -> Phase:
        return self._phase

    def observe(
        self,
        t: float,
        present_ids: frozenset[int],
        roster: frozenset[int] | None = None,
        catalog: frozenset[int] | None = None,
        resolving_ids: frozenset[int] | None = None,
    ) -> None:
        """`roster` (T10/D8a) filters the RECORDING half only: an id outside it
        is not an instrument, so it never becomes a track, never confirms, and
        never reaches the report. Setup readiness deliberately judges the full
        `present_ids` against the already-enrolled roster/catalog: every object
        on the table must be classified. `roster=None` means unfiltered.

        `catalog`/`resolving_ids` (T11/B2) are stored alongside `present_ids`/
        `roster` so `setup_readiness` can compute a pure verdict from the latest
        same-tick observation without reaching into the tracker or a clock."""
        self._advance(t, strict=True)
        if present_ids != self._idset:
            self._idset = present_ids
            self._idset_since_t = t
        # Store the latest same-tick setup observation for setup_readiness. This
        # runs in every phase (cheap) — readiness itself is only legal in
        # SETUP/FINISHED, and prepare() resets these fields.
        self._setup_present = present_ids
        self._setup_roster = roster or frozenset()
        self._setup_catalog = catalog or frozenset()
        self._setup_resolving = resolving_ids or frozenset()
        if self._phase is Phase.RECORDING:
            recorded_ids = present_ids if roster is None else present_ids & roster
            self._observe_recording(t, recorded_ids)

    def setup_readiness(self, t: float) -> SetupReadiness:
        """Pure verdict on setup readiness from the latest same-tick observation.
        Legal only in SETUP/FINISHED. Priority order for the blocking reason
        (D3/B2): recognising > unknown_objects > missing_instruments >
        hold_steady. Capture health is combined by the API, not here."""
        if self._phase not in (Phase.SETUP, Phase.FINISHED):
            raise InvalidPhase(f"setup_readiness invalid in {self._phase}")
        if self._setup_present is None:
            # Nothing observed yet -> not ready, still coming up.
            return SetupReadiness(0, 0, 0, 0, 0, 0.0, False, "recognising")

        present = self._setup_present
        roster = self._setup_roster
        catalog = self._setup_catalog
        resolving_ids = self._setup_resolving

        recognised = present & roster
        non_roster = present - roster
        resolving = resolving_ids & present
        unknown = non_roster - resolving

        detected_count = len(present)
        expected_count = len(catalog)
        recognised_count = len(recognised)
        resolving_count = len(resolving)
        unknown_count = len(unknown)
        stable_for_s = max(0.0, self._effective_t(t) - self._idset_since_t)

        all_recognised = expected_count > 0 and recognised == catalog
        ready = (
            all_recognised
            and unknown_count == 0
            and resolving_count == 0
            and stable_for_s >= self._setup_stable_s
        )
        if ready:
            blocking_reason = None
        elif resolving_count > 0:
            blocking_reason = "recognising"
        elif unknown_count > 0:
            blocking_reason = "unknown_objects"
        elif not all_recognised:
            blocking_reason = "missing_instruments"
        else:
            blocking_reason = "hold_steady"
        return SetupReadiness(
            detected_count,
            expected_count,
            recognised_count,
            resolving_count,
            unknown_count,
            stable_for_s,
            ready,
            blocking_reason,
        )

    def prepare(self, t: float) -> None:
        """Begin a fresh setup pass. Legal only from SETUP or FINISHED. Clears the
        setup id-set/stability and recording tracks so the next observation starts
        at zero seconds, but PRESERVES the finished report and the current phase
        (the previous report is discarded only on a successful start())."""
        if self._phase not in (Phase.SETUP, Phase.FINISHED):
            raise InvalidPhase(f"cannot prepare from {self._phase}")
        self._advance(t, strict=False)
        self._idset = None
        self._idset_since_t = t
        self._setup_present = None
        self._setup_roster = frozenset()
        self._setup_catalog = frozenset()
        self._setup_resolving = frozenset()
        self._tracks = {}
        # Deliberately does NOT touch self._phase or self._report.

    def start(self, t: float) -> None:
        if self._phase not in (Phase.SETUP, Phase.FINISHED):
            raise InvalidPhase(f"cannot start from {self._phase}")
        self._advance(t, strict=False)
        self._phase = Phase.RECORDING
        self._start_t = t
        self._tracks = {}
        self._report = None

    def stop(self, t: float) -> Report:
        if self._phase is not Phase.RECORDING:
            raise InvalidPhase(f"cannot stop from {self._phase}")
        self._advance(t, strict=False)
        report = self._build_report(t)
        self._report = report
        self._phase = Phase.FINISHED
        return report

    def recording_status(self, t: float) -> tuple[float, tuple[InstrumentStatus, ...]]:
        if self._phase is not Phase.RECORDING:
            raise InvalidPhase(f"recording_status invalid in {self._phase}")
        effective_t = self._effective_t(t)
        statuses = []
        for tracker_id in sorted(self._tracks):
            track = self._tracks[tracker_id]
            confirmed, on_table = _project(track, effective_t, self._on_debounce_s, self._off_debounce_s)
            if not confirmed:
                continue
            statuses.append(
                InstrumentStatus(
                    tracker_id=tracker_id,
                    label=f"Instrument {tracker_id}",
                    on_table=on_table,
                )
            )
        return (effective_t - self._start_t, tuple(statuses))

    def report(self) -> Report:
        if self._phase is not Phase.FINISHED:
            raise InvalidPhase(f"report invalid in {self._phase}")
        assert self._report is not None
        return self._report

    def _effective_t(self, t: float) -> float:
        return t if self._last_t is None else max(t, self._last_t)

    def _advance(self, t: float, *, strict: bool) -> None:
        if self._last_t is not None:
            backwards = t <= self._last_t if strict else t < self._last_t
            if backwards:
                raise ValueError(f"t={t} is not monotonic (last t={self._last_t})")
        self._last_t = t

    def _observe_recording(self, t: float, present_ids: frozenset[int]) -> None:
        for tracker_id in set(self._tracks) | set(present_ids):
            track = self._tracks.setdefault(tracker_id, _Track())
            now_present = tracker_id in present_ids
            if now_present and not track.run_present:
                track.first_present_t = t
            if now_present:
                track.last_present_t = t
            track.run_present = now_present

            was_confirmed, was_on_table = track.confirmed, track.on_table
            confirmed, on_table = _project(track, t, self._on_debounce_s, self._off_debounce_s)
            if confirmed and not was_confirmed:
                track.confirmed = True
                track.on_table = True
            elif was_confirmed:
                if was_on_table and not on_table:
                    track.on_table = False
                    track.usage.append(UsageWindow(off_s=track.last_present_t - self._start_t, on_s=None))
                elif not was_on_table and on_table:
                    track.on_table = True
                    closed = track.usage[-1]
                    track.usage[-1] = UsageWindow(off_s=closed.off_s, on_s=track.first_present_t - self._start_t)

    def _build_report(self, t: float) -> Report:
        instruments = []
        for tracker_id in sorted(self._tracks):
            track = self._tracks[tracker_id]
            was_confirmed = track.confirmed
            confirmed, on_table = _project(track, t, self._on_debounce_s, self._off_debounce_s)
            if not confirmed:
                continue
            usage = list(track.usage)
            if was_confirmed:
                if track.on_table and not on_table:
                    usage.append(UsageWindow(off_s=track.last_present_t - self._start_t, on_s=None))
                elif not track.on_table and on_table:
                    closed = usage[-1]
                    usage[-1] = UsageWindow(off_s=closed.off_s, on_s=track.first_present_t - self._start_t)
            instruments.append(
                InstrumentReport(
                    tracker_id=tracker_id,
                    label=f"Instrument {tracker_id}",
                    completeness="present" if on_table else "missing",
                    usage=tuple(usage),
                )
            )
        return Report(duration_s=t - self._start_t, instruments=tuple(instruments))
