---
id: T10
title: App-side Unknown & Pending — make T06's decision real in the UI
type: wayfinder:task
status: closed
assignee:
blocked-by: []
---

## Question

Implement [`linker-design.md` §7 "Rejection & Pending — app-side encoding"](../../../../model/docs/linker-design.md).
[T06](T06-unknown-objects-ux.md) **decided** that unmatched newcomers render as
Unknown and stay out of Usage/Completeness. Nothing was built. The model half
works; the app half does not exist.

## Why this is demo-visible

The demo choreography has **the audience placing random foreign objects on the
table**. Today the linker correctly settles them as Unknown internally — verified
on Take B, six foreign tracks (phone/pen/keys) all settled Unknown, none borrowed
a Missing identity. But the app doesn't know that. It renders
`f"Instrument {id}"` for **any** id it sees, so a phone becomes "Instrument 12"
and **counts as present for completeness**. That is precisely the failure T06 was
opened to prevent, and the audience will trigger it on purpose.

## §7's premise is now false — resolve this first

§7 opens: *"Nothing new crosses the seam. The app already has the frozen roster
`{1…N}`."* **Both halves are wrong today:**

1. **The app has no roster.** `grep -i roster app/backend/backend/` → nothing. The
   app never captured the enrolled set.
2. **The roster is not `{1…N}`.** Session ids are Deep OC-SORT's raw counter, so a
   tray of 8 yields something like `{3,5,7,9,10,11,12,14}` — the same root cause as
   [T08](T08-gallery-binding.md) ("Instrument 10" on a tray of 8). §7's colour
   mapping and its `> N` "provisional id" test both assume contiguity from 1 and
   silently break without it.

Two routes; **pick deliberately and write the choice into §7:**

- **(a) App derives the roster** from the ids present at Start — keeps "nothing
  new crosses the seam", works with arbitrary ids if the `> N` test is replaced by
  a set-membership test. ⚠️ **Coordination risk:** the app's Start snapshot and the
  linker's enrolment freeze are different instants (freeze is ~2 frames ≈ 0.5 s
  later, `windows_frames.enrolment = 2` at 3 fps). If the two disagree on the set,
  every downstream Unknown decision disagrees. Pin how they agree.
- **(b) Expose the roster across the seam** — the linker already holds it
  (`linker.roster`, a frozenset; covered by `test_session_linker.py`). Costs a
  seam change §7 explicitly avoided, but removes the coordination risk entirely.

**This is a grilling item for Bram, not a coder's call** — it changes the seam
contract that D8/`tracker-interface.md` pin.

## What §7 specifies (once the premise is fixed)

- **During the decision window:** a not-in-roster id present for **< the 1 s entry
  debounce** → small "resolving" **spinner**, gray mask.
- **On resolution:** id flips to a roster id → spinner clears, snaps to that
  instrument's fixed colour. Still not-in-roster at debounce → **gray Unknown**,
  out of Usage/Completeness.
- **Colours:** 8 roster ids → 8 distinct colours; not-in-roster → gray. Because
  the linker re-emits the *original* session id, a returned instrument **regains
  its original colour**.
- The spinner rides the app's debounce clock, not the linker's decision instant
  (the model carries no "decided-Unknown" flag by design) — a clear foreign object
  may spin ~1 s then settle gray. Harmless.

## Acceptance

- A foreign object placed mid-recording renders **gray Unknown**, is **never**
  labelled `"Instrument N"`, and does **not** count toward completeness.
- A returned instrument regains its original colour with no report trace from its
  provisional id.
- `app/docs/DESIGN.md` D8a updated from "not yet built" to the shipped behaviour;
  §7's premise paragraph rewritten with the chosen route.
- Frontend component tests (Vitest + RTL + MSW, per D12) cover Unknown, Pending,
  and colour-on-return.

## Relationship to T08 and T09

- **Not blocked by [T08](T08-gallery-binding.md)**, under either route — but T08
  makes route (a) *nicer* (ids become specimen numbers), so if T08 lands first,
  revisit the choice.
- **Independent of [T09](T09-first-live-check.md)** — but T09 is the run where a
  foreign object gets placed on a real tray, so **T09 will show this gap live**.
  Expect "the phone said Instrument 12" as a T09 finding; that is this ticket, not
  a linker bug.

## Resolution

Closed 2026-07-15 — **§7 shipped, both halves** (backend and frontend). Backend
spec: [assets/T08-T10/T10-BACKEND-SPEC.md](../assets/T08-T10/T10-BACKEND-SPEC.md);
the frontend half was a separate build on the same grilling.

### The premise question — resolved with Bram: route (b)

**The roster crosses the seam.** `tracker.roster` (the frozenset the linker already
held) is now on the `InstrumentTracker` protocol and is sampled by the capture loop
**in the same tick as `present_ids`** — which removes the coordination risk of route
(a) entirely, at the cost of the seam change §7 avoided. Two consequences the
grilling also pinned:

- **Unknowns are video-only.** A not-in-roster id gets a gray mask on the video and
  **no panel row at all** — it never becomes a `_Track`, never confirms, never
  reaches `/status.recording.instruments`, the report, `on_table_count`, Usage, or
  Completeness. Only the *recording* half filters; the Start id-set stability gate
  still sees **everything** detected (the operator's judgment covers the table).
- **Colour is a field, not a frontend guess.** Each roster instrument's `/status`
  entry carries `colour` — `roster_colour(roster, id)` = the palette hex at its index
  in `sorted(roster)`. The panel swatch and the overlay mask read the *same* hex, and
  because the linker re-emits the original session id, a returned instrument regains
  its colour with no renderer memory involved.

Note the roster is **empty for ~0.7 s after Start** (pre-freeze). Shorter than the
1 s entry debounce ⇒ the report is unchanged. T08's renumbering does not make route
(a) viable retroactively: a mixed bind leaves the roster non-contiguous anyway.

### What shipped

- **Backend** — `Latest.roster` + the 3-arg `OnFrame`; `OverlayRenderer` (stateful,
  capture-thread-only) replacing the plain `render` fn, with `set_recording()` flipped
  by the start/stop handlers (409 paths don't flip it); `Session.observe(t,
  present_ids, roster=None)` filtering the recording half by roster (default `None`
  keeps today's behaviour, Session stays pure per D11); `InstrumentStatusModel.colour`;
  `ForeignWindow` + `DEFAULT_FOREIGN` (id 9, t = 40–48 s) so `--fake` demos the whole
  story — pickup/return → foreign object goes gray → real loss, with the report still
  exactly ids 1–8 with 3 missing. **202 backend tests pass.**
- **Frontend** — Unknown, Pending, and colour-on-return covered per D12 (Vitest + RTL
  + MSW); the panel swatch renders the API's `colour`. **52 frontend tests pass.**
- Overlay states: roster id → `roster_colour` mask/box + `"Instrument {id}"`;
  not-in-roster and younger than `pending_s` (1.0, riding the app's debounce clock,
  not the linker's decision instant) → gray + an animated resolving spinner, no label;
  not-in-roster and settled → solid gray + `"Unknown"`. **Never `"Instrument N"`, in
  any state, at any age.** Not recording → today's overlay, unchanged.

### ⚠️ Engineering calls Bram did NOT grill — FLAG

1. **B-V6 renderer wiring route.** The spec never said how `create_app` reaches the
   `OverlayRenderer` to call `set_recording`. Ruled: **`create_app` owns the renderer**
   and wires it via a new `CaptureLoop.set_render_fn` (mirroring the existing
   `set_on_frame` precedent). **Consequence to know:** `create_app` now **replaces**
   any `render_fn` passed to `CaptureLoop`'s constructor.
2. **`roster_colour(roster, id)` returns the gray `"#9ca3af"` for a non-roster id
   rather than raising.** The spec defines it only for roster ids; B-S1 stops unknowns
   ever reaching it, and this collapses B-A1's no-snapshot placeholder into one path.
3. **A residual race, named and accepted, in `OverlayRenderer._age`.** Two races were
   **fixed** (`set_recording` now clears *before* publishing the flag; `_evict` mutates
   in place and never rebinds). A third **survives**: `_age` is a read-modify-write
   (get then store), so if `set_recording(True)`'s clear lands between those two ops,
   an in-flight recording-1 frame writes recording-1's `first_seen` back into
   recording-2's fresh map — that foreign object then shows **"Unknown" immediately**
   instead of the resolving spinner, until eviction ~5 s later. Requires a GIL switch
   inside a 2-op window **and** a frame in flight across a stop→start (i.e. stop and
   start within one ~100–300 ms render tick). **Accepted as cosmetic and improbable.**
   Closing it needs an epoch counter (~4 lines) or a lock on the capture hot path —
   **Bram can call for the epoch counter.**
4. **A pre-existing race, NOT T10's** (found independently by two agents):
   `session.start(clock())` samples the clock under the lock while the capture thread's
   in-flight frame carries an earlier `t`, so that frame's `observe` arrives backwards
   ⇒ `ValueError: t is not monotonic`, caught and logged by `_notify`'s AC8 handler,
   frame skipped. **It happens in `orc-demo` today.** Flagged, not fixed.
5. **BRANDING.md tension (frontend).** BRANDING says *"The camera detection overlay
   should use the functional status colours, not the logo spectrum"* and *"colour must
   carry meaning"*. **An 8-hue roster palette is the thing that line pushes back on.**
   Built as specced — the grilling decided it, and the hue *does* carry meaning
   (mask↔row identity, not decoration) — but the panel now shows up to 8 hues on a
   surface branded "teal + semantic only". **Either BRANDING carves out
   identity-mapping, or the doc and the product disagree.**
6. **The frontend swatch is `aria-hidden` + `data-testid`** — a **new convention** (the
   repo had none) rather than `role="img"`: a hue is meaningless to a screen reader,
   and `role="img"` would have broken `LiveScreen.test.tsx`'s `queryByRole("img")`
   assertion.
7. **B-V4 has no gate: nothing tests that the spinner STOPS at settle** ("solid gray"
   implies it). The shipped code does stop it; **no test holds it there.**
