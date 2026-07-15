---
id: T10
title: App-side Unknown & Pending — make T06's decision real in the UI
type: wayfinder:task
status: open
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
