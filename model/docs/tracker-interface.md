# InstrumentTracker contract

`orc_model.pipelines.tracking.InstrumentTracker` is the only model interface
used by the dashboard and demo backend. Consumers may change anything behind
this seam only by changing this contract deliberately.

## Interface

```python
class InstrumentTracker(Protocol):
    confidence: float
    def update(self, frame: np.ndarray) -> sv.Detections: ...
    def reset(self) -> None: ...
    @property
    def roster(self) -> frozenset[int]: ...
    @property
    def catalog(self) -> frozenset[int]: ...
    @property
    def class_names(self) -> dict[int, str]: ...
    @property
    def model_version(self) -> str: ...
```

This contract has taken three deliberate widenings since it was pinned:
`roster` (2026-07-15; see § "Identity semantics" → "The roster crosses the
seam"), the per-detection `data["resolving"]` flag (2026-07-16; see
§ "Identity semantics" → "The resolving flag splits the Unknown range"), and
`catalog` (2026-07-16, T11; see § "Identity semantics" → "The catalog is the
fixed set of known specimens").

`load_tracker(weights_path, confidence=0.5, ...)` builds the real RF-DETR →
workspace filter → Deep OC-SORT → SessionLinker composition. It also loads and
embeds the persistent specimen galleries eagerly, from `instruments_dir`
(default: the shipped `model/data/instruments`; pass `None` to disable binding).
In the default (non-catalog) mode a missing or empty directory logs and proceeds
with no galleries — it never raises. Under `catalog_only_enrolment` (the demo
config) a missing, empty, wrong-sized, or id-colliding catalog is instead a fatal
startup error: `load_tracker` fails closed rather than silently degrade to
session-only enrolment. A lightweight `FakeInstrumentTracker` implements the same protocol
without loading ML dependencies; its `catalog == roster ==
frozenset(range(1, n_instruments + 1))` (1-based, matching the real linker's
specimen-number session ids), since the fake enrols its whole catalog instantly.

## Input

`update(frame)` accepts one BGR `uint8` NumPy array with shape `(H, W, 3)`.
Frames arrive on one thread and in capture order. The tracker treats the array
as read-only and does not retain a mutable reference.

The stream is processed as fast as inference allows, not at camera capture
rate. On the demo M3 Max the complete uncached path measures about 3 fps.
Frame-based tracker and linker windows must be configured with that processed
rate; offline replay passes the effective sampled video rate explicitly.

## Output

`update()` returns an `sv.Detections` for the same input frame with all fields
row-aligned:

| field | required shape | meaning |
|---|---|---|
| `xyxy` | float32 `(N, 4)` | box in the input frame's pixel coordinates |
| `mask` | bool `(N, H, W)` | full-frame instance mask |
| `confidence` | float32 `(N,)` | detector confidence |
| `class_id` | int `(N,)` | key into `class_names` |
| `tracker_id` | int `(N,)` | stable session identity (∈ `roster`), or an offset Unknown ID (see § "Identity semantics") |
| `data["resolving"]` | bool `(N,)` | `True` while the linker is still deciding this track (pending in its evidence window, or deferred); `False` for a settled roster id and for a settled Unknown (see § "Identity semantics") |

An empty result is `sv.Detections.empty()`, never `None`. Valid frames do not
raise merely because nothing was detected.

Returned detections satisfy the configured confidence and workspace gates.
Detector boxes may extend outside the frame; consumers clamp before indexing.

## Identity semantics

The roster frozen by enrolment during setup defines the known physical objects
for one recording. The app's recording Start preserves that approved roster; it
does not reset the tracker.
An enrolled instrument that leaves and returns is re-emitted under its original
session ID after the link decision. A new, foreign, rejected, or still-undecided
track is Unknown, because its emitted ID is absent from the frozen roster.

The comparison/eligibility and open-set rules are specified in
[`linker-design.md`](./linker-design.md). Consumers must not maintain their own
alias map or retroactively rewrite IDs.

### The roster crosses the seam

`roster` is a read-only `frozenset[int]` of the session IDs enrolled at the
linker's enrolment freeze. It is **empty before that freeze and immediately after
`reset()`**, then constant for the rest of the recording. Consumers test
membership against it and derive Unknown from that test — nothing else.

Three rules consumers must not break:

- **Do not derive your own roster.** That was route (a), and it was rejected by
  grilling (Bram, 2026-07-15, wayfinder T10): a consumer's own phase-transition
  snapshot and the linker's enrolment freeze are different instants, and
  any disagreement between the two sets corrupts every Unknown decision for the
  whole recording. Reading the property removes the coordination entirely.
- **Do not assume contiguity, or that it starts at 1.** Session IDs come from
  gallery binding: a bound identity wears its specimen number, a session-only
  identity gets the next reserved number above the loaded specimens. A partial
  bind therefore yields a gapped roster such as `{1, 2, 5, 9, 10}`
  (`linker-design.md` §3, "Session numbering"). `> N` is not a valid test.
- **Sample it in the same tick as the detections you are judging.** The property
  and the emitted IDs are consistent per `update()` call — across calls, only if
  you read both from the same one.

This is a deliberate widening of this contract and of `app/docs/DESIGN.md` D8.
The seam still carries no status field and no alias map. It does expose exactly
one per-detection flag, `data["resolving"]` (a later widening — see below), but
that flag reports only whether a decision is still *pending*; it never pre-empts
the "is this ours?" roster-membership test, which stays the sole identity gate.

### The catalog is the fixed set of known specimens

`catalog` is a read-only `frozenset[int]` of the loaded persistent specimen IDs.
Unlike `roster`, it is **known from construction and constant for the tracker's
lifetime** — it is *not* emptied by `reset()` and does not wait for the enrolment
freeze, because the persistent galleries are loaded once at `load_tracker()` time
and held resident across every recording. It is the set of physical instruments
the installation could ever recognise; `roster` is the subset actually enrolled
this recording.

The seam offers an optional **catalog-only enrolment** mode (default off,
enabled for the KU Leuven demo via `app/mvp.toml`). When it is on, **only a track
that confidently binds to a loaded specimen may join the roster** — a foreign
object on the table during setup stays in the Unknown range instead of silently
enrolling. The invariant a consumer may rely on: in catalog-only mode
`roster ⊆ catalog`, and **no id outside `catalog` ever enters the roster** (there
are no session-only ids). In legacy mode `roster` may contain session-only ids
above `max(catalog)`, so it is not a subset. The `FakeInstrumentTracker` reports
`catalog == roster == frozenset(range(1, n_instruments + 1))`, enrolling its
whole catalog instantly.

This is the third deliberate widening of this contract (2026-07-16, T11), after
`roster` and `data["resolving"]`. Like `roster`, it carries no status field: it
is a plain membership set, sampled per `update()` if a consumer judges Unknown
against the catalog rather than the roster.

### Two disjoint emitted ID ranges

`update()` emits IDs from two ranges that never overlap:

| track | emitted `tracker_id` |
|---|---|
| mapped to a roster identity | its **session ID**, unchanged (∈ `roster`) |
| **not** mapped to a roster identity — pending in its evidence window, deferred, or settled Unknown | `raw_id + unknown_id_offset` (default **1000**) |

The offset applies to the *emitted* ID only; the linker's internal state stays
keyed by raw tracker IDs. Because renumbering pulls session IDs down into a small
integer range (`{1…8}` on the demo tray) while raw OC-SORT IDs live in the same
range (a tray of 8 commonly freezes as `{3,5,7,9,10,11,12,14}`), the two would
otherwise **collide** — an unknown emitting raw ID 3 would be indistinguishable
from roster identity 3. The offset keeps the ranges disjoint forever, which is
what makes "Unknown = not in `roster`" exact rather than approximate.

Raw tracker IDs never cross the seam in any form. If an assigned session ID ever
reached `unknown_id_offset` the linker logs a warning; demo-scale rosters do not
get near it.

### The resolving flag splits the Unknown range

The offset range above covers three states a consumer draws differently: a
track *pending* in its evidence window, a track *deferred* behind a still-
coasting active id, and a *settled Unknown*. `data["resolving"]` (bool, row-
aligned) splits them: `True` for the first two — the linker is still deciding —
and `False` once the track has settled, whether it settled into a roster id
(linked, so its emitted id also flips into `roster`) or into an Unknown. Roster
ids are always `False`; during the pre-freeze enrolment window every track reads
`True`, since nothing has a settled identity yet.

A settled Unknown is **stable, not permanent for the raw id.** While the raw
track stays present the linker watches it cheaply and may **re-arm** it — return
it to Pending — when its crop changes materially or a new link target becomes
available (`linker-design.md` §6.6). On a re-arm the flag flips back to `True`
(the spinner shows again) while a fresh evidence window is collected, then clears
once the linker either links it to its original session id or settles it Unknown
again. So a consumer must treat `resolving` as a live per-frame flag, not a
one-way latch: the same raw id can legitimately read `False → True → False` more
than once across a recording. The emitted id stays in the offset range for the
whole Pending re-check and flips into `roster` only if the re-check links.

This exists so a consumer can show a "resolving" spinner that follows the
linker's *actual* decision instead of a local timer. Before it, the app timed
the spinner off its own first-seen clock; a track deferred longer than that
clock (the normal coasting-handoff path) flickered to a settled "Unknown" and
then snapped to its instrument once it linked. Reading the flag removes that
disagreement entirely — the spinner clears exactly when, and only when, the
linker settles the track. Widening grilled with Bram 2026-07-16 (the wait-state
fix); it retires T10's renderer-owned `pending_s` timer and first-seen map
(`T10-BACKEND-SPEC.md` B-V3/B-V5).

A track with no `resolving` key at all (a bare test double, or any producer
that predates this widening) is read as settled — the safe default: no spurious
spinner.

### The freeze frame itself emits session IDs

Frames **strictly before** the enrolment freeze pass raw IDs through, as they
always have. **The freeze frame itself already emits session IDs** — the flip
happens on the very `update()` call in which the roster freezes, not on the
following one. On that frame, tracks that are not roster identities already emit
in the offset range, exactly like every later frame. So `roster` and the emitted
IDs sampled from the same call are **always** consistent, including the call that
creates the roster. This is the seam's consistency guarantee, and it is why
reading the property is safe.

> **Flag — this was a spec ambiguity the T08 orchestrator resolved, not something
> Bram grilled.** The original wording ("at the freeze the emitted ids flip") let
> the freeze frame read as either side of the boundary; the test-writer and the
> coder both read it leniently (flip on the *next* frame), Codex read it strictly.
> Ruled **strict**, and `T08-SPEC.md` §B-N5 was amended, for two reasons:
>
> 1. **The spec's own timing only works strict.** At the pinned 3 fps the
>    enrolment window is two frames: the freeze frame is t≈0.67 s, the next is
>    t≈1.0 s. Only flipping on the freeze frame is "~0.7 s in, inside the app's
>    1 s entry debounce" — the lenient reading lands the flip exactly *at* the
>    debounce boundary and destroys the safety argument.
> 2. **The lenient reading is a live cross-seam bug, not cosmetic lag.** The
>    freeze publishes the session-ID roster *before* the frame returns, and the
>    app samples `roster` and the emitted IDs in the same tick and intersects
>    them. Emitting raw IDs on that one frame lets them collide with session IDs:
>    raw `{3,5,7,9,10,11,12,14}` ∩ roster `{1…8}` = `{3,5,7}` — presence booked
>    against three instruments that are not on the table. Exposing the roster
>    exists to remove exactly that class of bug; the lenient reading reintroduces
>    it for a frame.

## Mutable confidence

`confidence` is a plain read/write runtime setting and may be changed between
frames. It is forwarded to detector filtering. Changing it mid-recording can
spawn or retire raw tracks, so the app permits changes only during setup/finished
and couples each changed value to `reset()` on the capture thread.

## Session boundary

`reset()` starts a new identity namespace. It clears OC-SORT and all
SessionLinker state while preserving loaded detector and embedding models —
**and the persistent specimen galleries**, which are read from disk and embedded
exactly once at `load_tracker()` time and are reused by every later enrolment.
`roster` reads empty from the moment `reset()` returns until the next enrolment
freeze, roughly 0.7 s of frames later.

## Metadata

- `class_names` is currently `{0: "surgical_instrument"}`.
- `model_version` combines the weights filename stem and a short SHA-256 hash.

## Consumer responsibilities

The model does not encode thumbnails, serve HTTP, write capture datasets, or
build reports. Consumers already own the frame and derive UI crops from
`frame + xyxy/mask/tracker_id`. The app owns roster-based usage/completeness
reporting, reads the roster from `roster`, and must never promote an Unknown ID
into it.

## Example

```python
tracker = load_tracker("model/weights/checkpoint_best_regular.onnx")
tracker.reset()

while streaming:
    frame = camera.read()
    detections = tracker.update(frame)
    render(frame, detections)
```
