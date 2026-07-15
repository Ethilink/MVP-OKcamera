# SPEC — T10 app-side Unknown & Pending, backend half (`app/backend/`)

**Single source of truth for all blind-TDD roles.** Distilled 2026-07-15 from
`model/docs/linker-design.md` §7, the T10 ticket, and a grilling with Bram
(same day): route **(b)** — the roster crosses the seam; unknowns are
**video-only** (no panel row); each roster instrument's `/status` entry carries
its fixed mask **colour**. The frontend half is a separate build.

> **REVISED 2026-07-16 — the wait-state fix.** The resolving-vs-settled split is
> no longer a renderer-owned `pending_s` timer + first-seen map. The model seam
> now carries a per-detection `data["resolving"]` flag (see
> `tracker-interface.md` → "The resolving flag splits the Unknown range"), and
> `OverlayRenderer` gates the spinner on it. This reverses the specific choice in
> the "Tunable parameters" note below ("`pending_s` … not the linker's decision
> instant") and retires **B-V3/B-V5**'s renderer clock, because a *deferred*
> track outlasts the timer and flickered to "Unknown" before it linked. The
> sections below are updated in place; the video-only architecture and the
> colour/roster rules are unchanged.

## Goal

Make T06's decision real: a foreign object placed on the table mid-recording
renders as a **gray mask with a small "resolving" spinner** for its first
moments, settles to **gray "Unknown"**, is never labelled `"Instrument N"`, and
never enters `/status.recording.instruments`, Usage, or Completeness. Roster
instruments get **8 distinct, stable colours** on the overlay; because the
linker re-emits original session ids, a returned instrument regains its colour.
The roster reaches the app via the widened model seam (`tracker.roster`).

Not in scope: any change under `model/` (T08 ships the `roster` property —
treat it as existing per `tracker-interface.md`; note `roster` may be **empty
for ~0.7 s after Start** until the linker's enrolment freeze), frontend code
(separate build), report-shape changes, persistence, WebSockets.

## Public interface (changes only)

### `backend/capture.py`

```python
class Latest(NamedTuple):
    # ... existing fields unchanged, plus:
    roster: frozenset[int]   # tracker.roster sampled in the SAME tick as
                             # present_ids (builtin ints)

OnFrame = Callable[[float, frozenset[int], frozenset[int]], None]
# (t, present_ids, roster) — roster sampled same-tick
```

`CaptureLoop.__init__` gains `render_fn` calls with the new signature below;
everything else (health, reset, snapshot identity pacing) unchanged.

### `backend/render.py`

```python
ROSTER_PALETTE: tuple[str, ...]   # 8 distinct hex colours, e.g. "#4285f4"

def roster_colour(roster: frozenset[int], session_id: int) -> str:
    """Fixed hex colour for a roster id: palette[index of id in sorted(roster),
    mod 8]. Pure; stable for the whole recording because the roster is frozen."""

class OverlayRenderer:
    """Stateful only in its recording flag (capture thread only). Replaces the
    plain `render` function as CaptureLoop's default render_fn. Reads the
    spinner-vs-Unknown split from dets.data["resolving"] (REVISED 2026-07-16);
    holds no `pending_s` timer or first-seen map."""
    def __init__(self) -> None: ...
    def set_recording(self, recording: bool) -> None:
        """Thread-safe flag flip; called by the start/stop HTTP handlers."""
    def __call__(self, frame, dets, roster: frozenset[int], t: float) -> np.ndarray: ...
```

### `backend/session.py`

```python
class Session:
    def observe(self, t: float, present_ids: frozenset[int],
                roster: frozenset[int] | None = None) -> None: ...
```

### `backend/main.py`

```python
class InstrumentStatusModel(BaseModel):
    # ... existing fields unchanged, plus:
    colour: str   # roster_colour(...) hex; panel swatch == overlay mask colour
```

### `backend/fakes.py`

```python
@dataclass(frozen=True)
class ForeignWindow:
    """A scripted foreign object during recording: `tracker_id` (NOT in the
    roster) is present for appear_s <= t < disappear_s."""
    tracker_id: int
    appear_s: float
    disappear_s: float

DEFAULT_FOREIGN: tuple[ForeignWindow, ...]  # one window: id 9, t=40..48 s

# ScenarioState/ScenarioTracker gain a `foreign: Sequence[ForeignWindow]`
# kwarg (default DEFAULT_FOREIGN) and ScenarioTracker gains:
@property
def roster(self) -> frozenset[int]:   # frozenset(range(1, n_instruments + 1))

# REVISED 2026-07-16: ScenarioState/ScenarioTracker also gain a
# `resolving_window_s` kwarg (default 1.0), ScenarioState gains
# `is_resolving(tracker_id) -> bool`, and ScenarioTracker.update() emits
# data["resolving"] (see B-F6).
```

## Behaviour

### B-C: capture samples the roster

- **B-C1** `_capture_tick` reads `self._tracker.roster` once per tick,
  immediately after `update()`, casts members to builtin `int`, stores it in
  `Latest.roster`, and passes it to both the render call and `on_frame`.
- **B-C2** A tracker without a `roster` attribute (defensive) → treat as
  `frozenset()` is NOT required; the seam guarantees it. Do not add fallback
  code.

### B-S: session filters recording state by roster

- **B-S1** `observe(t, present_ids, roster)`: the id-set stability tracking
  (`setup_status`) keeps using the **full** `present_ids` (the Start gate is
  the operator's judgment on everything detected). Only the RECORDING half
  changes: `_observe_recording` uses `present_ids & roster` when `roster is not
  None`, so a not-in-roster id never becomes a `_Track`, never confirms, never
  appears in `recording_status()` or the report, never affects
  `on_table_count`, Usage, or Completeness.
- **B-S2** `roster=None` (default) preserves today's unfiltered behaviour —
  existing callers/tests that don't pass a roster are untouched semantically.
- **B-S3** Session stays pure (D11): roster arrives as data per call; no clock,
  no threads, no stored tracker reference.
- **B-S4** `main._on_frame` forwards the roster from the capture callback into
  `session.observe` under the existing lock.
- **B-S5** Empty roster while recording (the ~0.7 s pre-freeze window) means no
  ids pass the filter those frames — harmless: it is shorter than the 1 s
  entry debounce, so the report is unchanged.

### B-V: overlay encoding (video-only, per grilling)

All drawing happens in `OverlayRenderer.__call__` on the capture thread; it
returns the mutated frame exactly like today's `render`.

- **B-V1 (not recording)** `set_recording(False)` state: draw exactly today's
  overlay (per-track colours + `"Instrument {id}"` labels). Setup/finished
  phases must look unchanged — the roster there is stale by design.
- **B-V2 (recording, roster id)** A detection whose `tracker_id` ∈ roster:
  mask+box in `roster_colour(roster, id)` (same hex the API reports), label
  `"Instrument {id}"`. Colour follows the id, so a returned instrument regains
  its colour with no renderer memory involved.
- **B-V3 (recording, not-in-roster, resolving)** `dets.data["resolving"]` is
  `True` for this row (pending or deferred — REVISED 2026-07-16, was "first seen
  < `pending_s` ago"): **gray** mask/box, **no** "Instrument" and **no**
  "Unknown" label, and a small spinner-like resolving indicator drawn on the box
  (rotating arc or equivalent; exact art is free, but it must visibly animate
  across frames — driven by `t` — and read as "thinking", not as a 9th
  instrument colour).
- **B-V4 (recording, not-in-roster, settled)** `resolving` is `False` (or the
  key is absent — the safe default): solid **gray** mask/box, text label
  `"Unknown"`. Never `"Instrument N"`, in any state.
- **B-V5** The renderer keeps **no** per-id state (REVISED 2026-07-16): resolving
  is read fresh from the flag every frame, so it cannot latch and cannot grow.
  `set_recording(True)` has nothing to reset — a fresh recording gets a freshly
  reset linker with its own pending set. (The retired first-seen map / eviction
  are gone.)
- **B-V6** `set_recording` is called by the HTTP handlers: `True` right after
  `session.start(...)` succeeds, `False` right after `session.stop(...)`.
  Wrong-phase (409) paths must not flip it.

### B-A: API surface

- **B-A1** `InstrumentStatusModel.colour` = `roster_colour(roster, tracker_id)`
  with the roster taken from the current capture snapshot; if no snapshot
  exists (no frame yet), use `"#9ca3af"` (the gray) — a transient, harmless
  placeholder. `GET /status` recording entries all carry it.
- **B-A2** No other endpoint/shape changes. Unknown ids are absent from
  `/status` entirely (B-S1 guarantees it — no filtering code in main.py).
- **B-A3** The report (`/recording/stop`, `/report`) is unchanged in shape; its
  instruments are roster-only as a consequence of B-S1.

### B-F: fakes demo the whole story (`--fake` mode)

- **B-F1** `ScenarioState.present_ids()` during **recording** additionally
  includes any `ForeignWindow` id whose window contains `t`. During **setup**
  churn, foreign windows are ignored (no foreign objects while arranging).
- **B-F2** `ScenarioTracker.roster` = `frozenset(range(1, n_instruments+1))`,
  always (the fake "enrols" instantly; ForeignWindow ids must be chosen
  outside it — the default uses 9).
- **B-F3** `_draw_scenario` draws a present foreign object as a visibly
  different shape (e.g. dark rounded slab — a "phone"), at
  `ScenarioState.box(9, ...)` geometry so detections and pixels stay aligned.
- **B-F4** `DEFAULT_FOREIGN = (ForeignWindow(tracker_id=9, appear_s=40.0,
  disappear_s=48.0),)` — after instrument 1's return (35 s) and before
  instrument 3's loss (50 s), so a `--fake` run shows: pickup/return → foreign
  object goes gray → real loss. The existing DEFAULT_SCENARIO events are
  unchanged.
- **B-F5** With the default scenario, the final report still contains exactly
  ids 1–8 with instrument 3 missing — the foreign window must not perturb it.
- **B-F6 (added 2026-07-16)** `ScenarioTracker.update()` emits the seam's
  `data["resolving"]` flag row-aligned: a present foreign object reads `True` for
  its first `resolving_window_s` (default 1.0 s) after it appears, then `False`
  (settled); roster ids are always `False`. This reproduces the real linker's
  spinner-then-settle beat so `--fake` mode still demos it once the renderer
  reads the flag instead of a timer.

## Existing tests the new contract invalidates (test-writer owns migrating them)

`app/backend/tests/` — `test_capture.py` (Latest/on_frame arity, fake trackers
used there need a `roster` attr), `test_render.py` (render entry point becomes
`OverlayRenderer`; keep a thin compatibility check only if trivial),
`test_session.py` (observe signature — default arg keeps most tests intact),
`test_api.py` (new `colour` field; `_on_frame` arity), `test_fakes.py`
(ScenarioTracker roster/foreign). The suite runs with
`cd app/backend && uv run pytest -q`. All existing tests must end green
(migrated where the contract deliberately changed), plus new tests per
behaviour id above.

## Dependencies / reuse

- `sv.MaskAnnotator`/`BoxAnnotator`/`LabelAnnotator` with explicit
  `sv.ColorPalette`/custom colours (supervision supports custom palettes +
  `ColorLookup`); building per-detection colour arrays by hand with cv2 is
  also acceptable — whichever is less code.
- Files the coder may touch: `backend/capture.py`, `backend/render.py`,
  `backend/session.py`, `backend/main.py`, `backend/fakes.py`,
  `backend/thumbnails.py` only if a signature forces it. Nothing else; no
  frontend, no docs, no model/.

## Tunable parameters (never hard-assert values in tests)

the 8 palette hex values themselves (assert *distinctness* and *stability*, not
specific hexes), the gray placeholder hex, `DEFAULT_FOREIGN` timing values, and
the fake's `resolving_window_s` (a `--fake` demo-feel knob — how long a scripted
foreign object spins before settling).

_Retired 2026-07-16: `pending_s` and the B-V5 eviction timing — the renderer no
longer owns a resolving clock; it reads the linker's `data["resolving"]` flag._

## Acceptance beyond unit tests (orchestrator runs these, not the coder)

`uv run orc-demo --fake` + Chrome: foreign shape appears at ~40 s into a
recording, spins then goes gray "Unknown", panel never shows it, report
unchanged; a returned instrument's mask colour matches its panel swatch before
and after absence.
