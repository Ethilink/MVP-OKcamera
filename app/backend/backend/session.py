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
    off_since_s: float | None
    pickup_count: int


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
    pickup_count: int = 0
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
    def __init__(self, off_debounce_s: float = 1.5, on_debounce_s: float = 1.0) -> None:
        self._off_debounce_s = off_debounce_s
        self._on_debounce_s = on_debounce_s
        self._phase = Phase.SETUP
        self._last_t: float | None = None
        self._idset: frozenset[int] | None = None
        self._idset_since_t: float = 0.0
        self._start_t: float = 0.0
        self._tracks: dict[int, _Track] = {}
        self._report: Report | None = None

    @property
    def phase(self) -> Phase:
        return self._phase

    def observe(
        self,
        t: float,
        present_ids: frozenset[int],
        roster: frozenset[int] | None = None,
    ) -> None:
        """`roster` (T10/D8a) filters the RECORDING half only: an id outside it
        is not an instrument, so it never becomes a track, never confirms, and
        never reaches the report. The Start gate below deliberately keeps using
        the FULL `present_ids` — it is the operator's judgment on everything
        detected, made before any roster exists. `roster=None` means unfiltered."""
        self._advance(t, strict=True)
        if present_ids != self._idset:
            self._idset = present_ids
            self._idset_since_t = t
        if self._phase is Phase.RECORDING:
            recorded_ids = present_ids if roster is None else present_ids & roster
            self._observe_recording(t, recorded_ids)

    def setup_status(self, t: float) -> tuple[int, float]:
        if self._phase not in (Phase.SETUP, Phase.FINISHED):
            raise InvalidPhase(f"setup_status invalid in {self._phase}")
        if self._idset is None:
            return (0, 0.0)
        effective_t = self._effective_t(t)
        return (len(self._idset), effective_t - self._idset_since_t)

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
            if on_table:
                off_since_s = None
            elif not track.on_table:
                off_since_s = track.usage[-1].off_s
            else:
                off_since_s = track.last_present_t - self._start_t
            pickup_count = track.pickup_count + (1 if track.on_table and not on_table else 0)
            statuses.append(
                InstrumentStatus(
                    tracker_id=tracker_id,
                    label=f"Instrument {tracker_id}",
                    on_table=on_table,
                    off_since_s=off_since_s,
                    pickup_count=pickup_count,
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
                    track.pickup_count += 1
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
