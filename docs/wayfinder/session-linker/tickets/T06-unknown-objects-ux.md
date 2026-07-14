---
id: T06
title: Unknowns in the demo story — what the app shows for unmatched new tracks
type: wayfinder:grilling
status: closed
assignee: bram
blocked-by: []
---

## Question

When a new track matches no missing instrument (linker rejection — a foreign
object the audience placed, or a genuinely new instrument), what do the live
panel and the report show? (HITL — app-side product decision, Bram's.)

Today's behavior: the unmatched track gets a fresh id and, after the 1 s entry
debounce, registers as "Instrument N" — so a phone on the table becomes
"Instrument 7", appears in the report, and counts as *present* for
completeness. Is that acceptable demo behavior, or do we want unknowns
distinguished (e.g. only ids registered during Setup are "instruments";
mid-recording newcomers render as "Unknown object" and stay out of
completeness)?

[Linking eval scenarios in the harness + real fps pin](T04-harness-and-fps.md)
still measures whether the single-class detector fires on random objects; that
evidence determines how often this path appears, not the product rule. The seam
already gives the app what it needs: the Start roster is known and anything
registering later is distinguishable without catalog classification.

Resolution = the decision recorded here; if it changes the live panel or
report, the delta lands in `app/docs/DESIGN.md` and the T05 spec.

## Resolution

Resolved with Bram on 2026-07-14:

- The **known roster freezes at Start-time enrolment**. Only those physical
  objects participate in Usage and Completeness for that recording.
- A later detected object that does not confidently match a currently-missing
  known identity may still receive a raw track and overlay, but is presented as
  **Unknown object**. It does not become "Instrument N", expand the known
  roster, or enter Usage/Completeness.
- An Unknown/Pending raw track may still be promoted to the original session id
  if later frames produce a confident link to a missing known identity within
  the resolution budget. Rejection is never forced into a known identity.
- If the detector never fires on a foreign object, it remains naturally absent
  from the system; the behavior above covers the case where it does fire.
- This is session identity, not catalog classification. Labels for known items
  remain generic "Instrument N".

The exact enrolment timing, Pending representation, and whether Unknown status
is carried internally or derived app-side from roster membership are linker
design/spec questions. The externally visible invariant is fixed: post-Start
newcomers never contaminate completeness.
