# T04 — FastAPI layer

status: todo
depends-on: T02, T03
blocks: T08
spec: [api-contract.md](../api-contract.md) (normative, verbatim); [DESIGN.md](../DESIGN.md) D3, D4, D7, §Threading rules

## Goal

The HTTP surface: wire `CaptureLoop` + `Session` behind the frozen API
contract, plus the `--fake` entrypoint that runs the whole backend on
`ScenarioTracker` + `FakeCaptureSource`.

## In scope

`backend/main.py` + `tests/test_api.py` (FastAPI `TestClient`, fakes injected —
no camera, no sleep-based tests). Process: **blind-TDD**.

## Out of scope

Changing ANY response shape (the contract doc is frozen — a mismatch is a bug
here, not a contract edit). Frontend. Session/capture internals.

## Public interface (frozen)

The endpoints of `api-contract.md`: `GET /status`, `GET /stream`,
`POST /recording/start`, `POST /recording/stop`, `GET /report`. Plus:

```python
# backend/main.py
def create_app(capture: CaptureLoop, session: Session, model_version: str,
               clock=time.monotonic,          # monotonic seconds -> session t
               now=lambda: datetime.now().astimezone()) -> FastAPI: ...
               # now() -> TIMEZONE-AWARE datetime for the ISO started_at/
               # stopped_at stamps (contract shows an offset, e.g. +02:00 — a
               # naive datetime.now() would drop it). Injected so tests pin
               # timestamps; AC7.

def main(argv: list[str] | None = None) -> None:
    """CLI: `orc-demo --fake` | `orc-demo --camera N --weights PATH`.
    --fake wires ScenarioTracker(fps=10) + FakeCaptureSource(fps=10.0) (paced —
    T01) so scripted seconds ≈ wall seconds. Real mode does a LAZY import of the
    model factory INSIDE this branch only (`from orc_model... import
    load_tracker` — not yet shipped; a clear error if missing), so --fake and
    tests never import it. Registers capture.on_frame -> session.observe (under
    the session lock), starts the loop, serves uvicorn on :8000. CORS: allow the
    Vite dev origin (http://localhost:5173)."""
```

`pyproject.toml` gains the `orc-demo` script entry.

## Semantics

- **Session lock (T04 owns it).** T04 creates the `threading.Lock` DESIGN
  §Threading refers to, wraps `on_frame`→`session.observe` in it, and takes it
  in every handler that reads/mutates the session. Read `clock()` **inside** the
  lock right before the session call — otherwise a `clock()` sampled before a
  newer capture-thread `observe` lands can violate T02's `start/stop` monotonic
  guard and 500 intermittently.
- `POST /recording/start` → under the lock: `capture.reset_tracker()` **first**
  (satisfies the contract's "calls `tracker.reset()`" via the only thread
  allowed to touch the tracker — DESIGN §Threading), THEN
  `session.start(clock())`, so the session begins observing only post-reset
  frames. If `reset_tracker()` times out (stalled camera — see T03) → `503
  {"detail": "capture stalled"}`; if the tracker reset itself fails → `503
  {"detail": "tracker reset failed"}`. In both cases the session is unchanged. Records wall-clock
  `started_at = now().isoformat()` app-side. `POST /recording/stop` records
  `stopped_at = now()` and returns the report body directly (same shape as
  `GET /report`); `duration_s` comes from the session.
- `/status` maps `session.setup_status`/`recording_status` + `capture.health`
  into the contract shape: `setup` block non-null when `phase` is `setup` OR
  `finished` (D15 — `setup_status` is valid in FINISHED), `recording` block
  non-null only when `phase == recording`; `capture_health: "stalled"` unless
  `capture.health == "ok"`.
- `/stream` is an async MJPEG generator paced by `capture.generation` (send a
  part only when generation advances — no duplicate frames, no busy loop).
- Wrong-phase actions → `409 {"detail": ...}` (map `InvalidPhase`).
- Handlers doing blocking work are plain `def` (threadpool), never `async`.

## Acceptance criteria

- **AC1** Fresh app: `/status` → `phase="setup"`, `setup` non-null with
  `detected_count`/`stable_for_s`, `recording` null; report/stop → 409.
- **AC2** start → 200 with ISO `started_at` (aware, carries an offset) and
  invokes `capture.reset_tracker()` before `session.start` (assert ordering with
  a spy capture); reset timeout/failure → 503 with the session unchanged;
  second start while recording → 409; `/status` flips to `recording` with
  `elapsed_s` and lean identity/state instrument rows matching the contract
  field names exactly.
- **AC3** Drive the fakes through a scripted pickup: stop returns the contract
  report shape (`usage` windows with `off_s`/`on_s`, `completeness`,
  `model_version`, `duration_s`) and `GET /report` returns the identical body;
  every value JSON-native (a raw `json.dumps` of the dict round-trips — no
  numpy types).
- **AC4** start-after-finished → 200 and `/report` returns 409 until the next
  stop (D7: old report discarded). Also: while `phase == finished`, `/status`
  carries a non-null `setup` block (D15 — drives the run-2 Start gate) and null
  `recording`.
- **AC5** `/stream` responds `multipart/x-mixed-replace` and yields ≥ 2
  distinct JPEG parts as generation advances; parts stop advancing when
  generation stalls (asserted with a controllable fake capture, not sleeps).
- **AC6** `/status` shows `capture_health:"stalled"` when the loop reports
  stale/dead.
- **AC7** `create_app` never touches a real camera or wall clock: both `clock`
  and `now` are injectable, and a test pins `now` to assert a deterministic
  `started_at`/`stopped_at` in the report. `main(--fake)` constructs the fake
  stack (smoke: app starts, one `/status` OK via TestClient).
- **AC8** OpenAPI docs served at `/docs`; response models declared (Pydantic)
  so `openapi.json` reflects the contract for T08's type generation.

## Log

- (append dated one-liners here)
