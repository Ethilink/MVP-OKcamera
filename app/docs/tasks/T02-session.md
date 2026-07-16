# T02 — Session state machine + Usage/Completeness report

status: todo
depends-on: T01
blocks: T04
spec: [DESIGN.md](../DESIGN.md) D7–D9, D11; [api-contract.md](../api-contract.md) §Phases, §/status, §/report

## Goal

The correctness heart of the demo: a **pure** state machine that turns a stream
of `(t, present_ids)` observations into the live instrument panel, the Start
gate, and the final Usage/Completeness report. No clock, no threads, no I/O —
callers supply monotonic `t` (seconds, float).

## In scope

`backend/session.py` + `tests/test_session.py`. Process: **blind-TDD**.

## Out of scope

HTTP (T04), frames/tracker (T03), wall-clock timestamps (`started_at` is the
API layer's job).

## Public interface (frozen)

```python
# backend/session.py
class Phase(StrEnum):
    SETUP = "setup"; RECORDING = "recording"; FINISHED = "finished"

@dataclass(frozen=True)
class InstrumentStatus:      # one row of the live panel (contract: /status.recording.instruments)
    tracker_id: int
    label: str               # f"Instrument {tracker_id}"
    on_table: bool

@dataclass(frozen=True)
class UsageWindow:
    off_s: float
    on_s: float | None       # None = never returned

@dataclass(frozen=True)
class InstrumentReport:
    tracker_id: int
    label: str
    completeness: str        # "present" | "missing"
    usage: tuple[UsageWindow, ...]

@dataclass(frozen=True)
class Report:
    duration_s: float
    instruments: tuple[InstrumentReport, ...]   # sorted by tracker_id

class Session:
    def __init__(self, off_debounce_s: float = 1.5, on_debounce_s: float = 1.0): ...
    @property
    def phase(self) -> Phase: ...
    def observe(self, t: float, present_ids: frozenset[int]) -> None: ...
    # SETUP/FINISHED: feeds the Start gate. RECORDING: feeds presence tracking.
    def setup_status(self, t: float) -> tuple[int, float]: ...
    # (detected_count, stable_for_s) — count of CURRENTLY present ids and how
    # long that exact ID-SET has been unchanged. Valid in SETUP and FINISHED.
    # Stability is PHASE-INDEPENDENT: it tracks the current id-set's duration
    # purely from the observe history and does NOT reset at start()/stop() — so
    # in FINISHED (D15) it reflects observes made after Stop, and if the table
    # is untouched across Stop the window simply keeps growing. Fresh session,
    # no observe yet → (0, 0.0). Post-stop observes feed ONLY this gate; they
    # never mutate the frozen report.
    def start(self, t: float) -> None: ...   # SETUP|FINISHED -> RECORDING; discards old report; else InvalidPhase
    def stop(self, t: float) -> Report: ...  # RECORDING -> FINISHED; else InvalidPhase
    def recording_status(self, t: float) -> tuple[float, tuple[InstrumentStatus, ...]]: ...
    # (elapsed_s, instruments sorted by tracker_id); InvalidPhase unless RECORDING
    def report(self) -> Report: ...          # InvalidPhase unless FINISHED

class InvalidPhase(RuntimeError): ...
```

## Semantics (normative)

- **Time convention (one origin).** Callers pass *absolute* monotonic `t` to
  `observe`/`start`/`stop`/`setup_status`/`recording_status` (any origin — the
  API layer passes `clock()`; it need not start at 0). The `Session` records
  `t` at `start()` and reports **everything session-relative**: `elapsed_s`,
  `UsageWindow.off_s`/`on_s`, and `Report.duration_s` are all
  `t − start_t`. So a pickup with the id absent for absolute `t∈(120,135)` after
  `start(100)` (present again at `t=135`) yields `off_s≈20, on_s≈35`.
- **Accessors are non-mutating projections of (state, t).** `setup_status(t)`
  and `recording_status(t)` compute what the debounced state WOULD be at `t`
  (advancing `stable_for_s`, surfacing a debounce flip whose
  threshold `t` has crossed) **without committing anything** — polling at a
  later `t` never changes durable state. Durable state advances ONLY on
  `observe`/`start`/`stop`. `stop(t)` finalizes the report by applying this SAME
  projection at its `t`: a flip a bare poll surfaced is included in the report
  iff `stop`'s `t ≥` the flip time — no intervening `observe` required. (One
  consequence: `report()` is a pure function of the observation history + the
  `stop` time; repeated accessor calls can't perturb it.)
- Instrument universe = every id **confirmed** while RECORDING: an id joins the
  universe only once it has been continuously present for `> on_debounce_s`
  (**entry debounce**, same constant), registering as on-table. A shorter-lived
  id — a spurious detection, or a provisional pre-link id
  from the tracker's track linking (`model/docs/tracker-interface.md`
  §tracker_id across absence) — never enters the universe and leaves no trace
  in live status or the report. Accepted gap: an instrument that satisfied the
  Start gate but is already gone by the first RECORDING observe never enters the
  universe — demo choreography (one pickup at a time, starting from a full
  table) makes this a non-issue; documented, not engineered around.
- **Debounce (D9):** an on-table instrument whose id is absent flips to
  off-table only once absence has lasted `> off_debounce_s`; its window's
  `off_s` is the t of the LAST frame it was seen. Symmetric for return with
  `on_debounce_s`; `on_s` = t of the first frame of the confirmed return run.
  Blips shorter than the debounce leave no trace (no usage window).
- `stop(t)`: completeness = the instrument's **debounced** on/off state at Stop
  (D9 debounce / api-contract §Completeness; Bram 2026-07-07). An instrument in
  a **confirmed** off-table window at
  Stop (absence already passed `off_debounce_s`, window still open) →
  `completeness="missing"`, window stays open (`on_s=None`). Every instrument whose
  debounced state is on-table → `"present"` — this INCLUDES an instrument absent
  only for a sub-`off_debounce_s` blip at Stop (a detector flicker on a tool
  that's physically on the table): it is NOT yet confirmed off, so it stays
  present and leaves NO phantom window. **Mirror
  case (intended, not a bug):** an instrument that RETURNED less than
  `on_debounce_s` before Stop is still debounced-off → reported `"missing"` though
  physically back. Same debounce, opposite direction; demo choreography avoids
  Stopping mid-return, so this is documented, not engineered around.
- **Monotonic `t` guard (all time-taking methods).** `observe`, `start`, and
  `stop` require `t ≥` the last `t` the session saw (`observe` strictly `>`
  previous observe; `start`/`stop` `≥` last observe). A `t` that goes backwards
  → `ValueError`. Read-only accessors with a smaller `t` return the state as of
  the last known `t` (don't rewind), they don't raise.

## Acceptance criteria

- **AC1** Fresh session: `phase==SETUP`; `start` then `stop` then `start`
  walks SETUP→RECORDING→FINISHED→RECORDING; every invalid transition and
  wrong-phase accessor raises `InvalidPhase`.
- **AC2** Start gate: after observes with id-set {1,2,3} from t=10 to t=13,
  `setup_status(13) == (3, 3.0)`; one observe with {1,2} resets stability to 0.
- **AC3** Steady presence, no absences → every instrument `"present"` with
  `usage == ()`; live status contains identity and `on_table` only.
- **AC4** Scripted pickup at a **non-zero origin** (`start(100)`, id absent for
  absolute `t∈(120,135)` — last seen at 120, present again at 135, debounce
  defaults, ~10 fps observes) → exactly one window with **session-relative**
  `off_s≈20`, `on_s≈35` (tolerance one frame interval); live status shows only
  `on_table=False` during the absence. The final window proves report times are
  `t − start_t`, not absolute.
- **AC5** Flicker: absence lasting 0.9 s (< off_debounce 1.5) leaves NO window;
  a presence blip of 0.5 s during a real absence (< on_debounce
  1.0) does NOT close the window.
- **AC6** Never-returns: id last seen at t=50, absent for all t>50 through
  stop(80) → `"missing"`, last window `(50, None)`; report invariants hold (sorted,
  non-overlapping windows, `missing ⟺ last on_s is None`).
- **AC6b** Blink-at-stop: an instrument present throughout, whose id is absent
  only for the final <`off_debounce_s` before `stop(t)` (e.g. gone t∈(79.7,80],
  debounce 1.5) → `completeness="present"`, `usage==()`, NO open window — the
  sub-debounce blip leaves no trace even though it touches Stop.
- **AC7** Id appearing mid-recording and continuously present `> on_debounce_s`
  joins the universe and can itself be picked up and reported; an id whose
  entire presence run is `≤ on_debounce_s` before vanishing (spurious detection
  / provisional pre-link id) appears in neither live status nor the report, and
  does not corrupt others' windows.
- **AC8** `start` after FINISHED discards the old report (`report()` raises
  until the next `stop`), and debounce/usage state is fully reset.
- **AC9** Non-monotonic `observe` raises `ValueError`; `start`/`stop` with
  `t <` last observe also raise; `observe` in SETUP never creates usage state.
- **AC10** Non-mutating projection: after a confirmed presence then no further
  observes, `setup_status` at a growing `t` returns a growing `stable_for_s`;
  during a recording absence, `recording_status` at a `t` past `off_debounce_s`
  RETURNS `on_table=False` with NO intervening `observe`; and repeating the call
  does not change durable state — a later `stop(t)` still yields the same report
  as if the projection had never been queried.
- **AC11** D15 finished-phase gate + report immutability: fresh session
  `setup_status(t)==(0, 0.0)` before any observe. After `stop(t)`, further
  observes update `setup_status` (a stable id-set held across the Stop keeps
  growing `stable_for_s`; changing the id-set resets it) AND `report()` returns
  a byte-identical `Report` before and after those post-stop observes (they feed
  only the gate, never the frozen report).

## Log

- (append dated one-liners here)
