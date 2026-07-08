# T01 — Scaffold backend + frontend + fakes

status: todo
depends-on: —
blocks: T02, T03, T04, T05
spec: [DESIGN.md](../DESIGN.md) §Repository layout, D10, D14

## Goal

Both project skeletons exist, run, and ship the two fakes every later task
tests against: `ScenarioTracker` (scripted pickups) and `FakeCaptureSource`
(synthetic frames). No real feature code.

## In scope

`app/backend/` (uv project: `pyproject.toml`, `backend/__init__.py`,
`backend/fakes.py`, `tests/test_fakes.py`, empty `backend/{session,capture,render,main}.py`
stubs) and `app/frontend/` (Vite react-ts, Tailwind, shadcn/ui init, Vitest +
React Testing Library + MSW configured, one passing smoke test).

Backend `pyproject.toml` **must** declare `orc_model` (the model package
providing `InstrumentTracker` + `FakeInstrumentTracker`; path or workspace dep
on `../../model`) and `supervision` (`sv.Detections`) — every later backend task
imports these. Also FastAPI + uvicorn (T04 needs them; adding here keeps one
lockfile). Set `requires-python = ">=3.11"` (T02 uses `StrEnum`).

> **`load_tracker` does NOT exist yet.** The real-model factory is Constantijn's
> and hasn't landed; `orc_model` today exports only `InstrumentTracker`
> (Protocol) and `FakeInstrumentTracker` (drift-only). Do not import a
> `load_tracker` at module top level anywhere — T04 imports it lazily inside the
> real-camera branch only, so `--fake` and the whole test suite never touch it.

## Out of scope

Any endpoint, screen, session logic, or capture thread. The TS API types
(T05's job).

## Public interface (frozen)

```python
# backend/fakes.py
@dataclass(frozen=True)
class ScenarioEvent:
    tracker_id: int
    leave_s: float
    return_s: float | None      # None = never returns ("missing")

class ScenarioTracker:
    """Implements InstrumentTracker (model/docs/tracker-interface.md).
    Instrument *positions* drift smoothly; events script pickups. Time =
    frame_count/fps (deterministic — no wall clock). `tracker_id`s are the
    integers `1..n_instruments` (1-based; DEFAULT_SCENARIO's ids 1 and 3 are
    within {1..5}). reset() restarts the script at t=0.

    Boundary-frame convention (so T02 windows land on exact seconds): an
    instrument is ABSENT from update() for frames where
    `leave_s < t < return_s` (return_s None ⇒ absent for all t > leave_s). So it
    is PRESENT at exactly t=leave_s (its last-seen frame) and PRESENT again at
    exactly t=return_s. At fps=10 that means present at t=20.0, absent from
    t=20.1 through t=34.9, present at t=35.0."""
    def __init__(self, n_instruments: int = 5, fps: float = 10.0,
                 events: Sequence[ScenarioEvent] = DEFAULT_SCENARIO,
                 confidence: float = 0.5) -> None: ...
    # + the full InstrumentTracker surface: update/reset/confidence/
    #   class_names/model_version ("scenario-0.1")

DEFAULT_SCENARIO = (
    ScenarioEvent(tracker_id=1, leave_s=20.0, return_s=35.0),
    ScenarioEvent(tracker_id=3, leave_s=50.0, return_s=None),
)

class FakeCaptureSource:
    """cv2.VideoCapture-compatible (isOpened/read/set/get/release) emitting
    1920x1080 BGR synthetic frames (frame index burned in). For --fake mode
    AND as the tests' capture source. Note: it does NOT take a camera index —
    T03's CaptureLoop calls cap_factory(camera_index), so wrap it as
    `lambda _idx: FakeCaptureSource()` rather than passing the class directly.

    **Pacing (critical for --fake correctness).** When `fps` is not None,
    `read()` blocks so frames are emitted at ~`fps` in WALL time — this makes
    scenario seconds ≈ wall seconds, so `ScenarioTracker(fps=10)`'s scripted
    t=20/35/50 line up with `Session`'s wall-clock debounce (1.5 s). Without
    pacing the loop free-runs at encode speed and the 15 s scripted pickup
    collapses below the debounce and never registers. Demo/`--fake` mode uses
    `fps=10.0` (matching the ScenarioTracker fps); UNIT TESTS pass `fps=None`
    for instant frames and drive timing by other means (injected clock / frame
    counts), never real sleeps."""
    def __init__(self, size: tuple[int, int] = (1920, 1080),
                 fps: float | None = 10.0) -> None: ...
```

## Acceptance criteria

- **AC1** `uv run pytest` passes in `app/backend/`; `npm test` and
  `npm run build` pass in `app/frontend/`.
- **AC2** `ScenarioTracker` satisfies `isinstance(t, InstrumentTracker)`
  (runtime-checkable Protocol from `orc_model.pipelines.tracking`).
- **AC3** With `DEFAULT_SCENARIO` at `fps=10`: instrument 1's id is absent from
  `update()` results for frames covering t∈(20, 35) and present again after;
  instrument 3 absent for all t>50. All other ids present throughout.
- **AC4** Determinism: two `ScenarioTracker`s fed the same frame count return
  identical `tracker_id` sets; `reset()` restores frame 0 behaviour.
- **AC5** `update()` output honours the tracker contract: `xyxy/confidence/
  class_id/tracker_id/mask` populated. The empty-table branch (→
  `sv.Detections.empty()`) is asserted with a **custom all-gone scenario**
  (every instrument has `return_s=None` and a `leave_s` in the past), since
  `DEFAULT_SCENARIO` never fully empties the table.
- **AC6** `FakeCaptureSource.read()` returns `(True, frame)` with the requested
  size; `set/get` record and echo `CAP_PROP_*` values.
- **AC7** shadcn/ui is initialised (`components.json` exists, one shadcn
  component imported in the smoke-tested page).
- **AC8** MSW is wired for tests (the smoke test mocks one fetch through MSW).

## Log

- (append dated one-liners here)
