# Product

Strategic design context for the **ORC demo app frontend** (`app/frontend`) — the
one operator screen for the UZ Leuven CSSD demo (2026-07-20). Scoped to this app,
not the whole repo: `data-collection/dashboard` and `model/` are separate surfaces
governed elsewhere. The visual system lives alongside this in
[`BRANDING.md`](BRANDING.md) and the tokens in `app/frontend/src/index.css`; the
product behaviour and architecture are in [`DESIGN.md`](DESIGN.md).

## Register

product

## Platform

web

## Users

The hands-on user is a **single operator** — one of us, or a clinician we've
briefed — standing at a laptop beside a camera on a tripod that films a sterile
table of 5–6 surgical instruments. One person, one screen, one recording at a
time. Their job across three screens: confirm every instrument is detected, press
Start, watch instruments get picked up and returned, press Stop, and walk through
a per-instrument report that matches what physically happened.

But the interface also speaks to a second audience it doesn't belong to: the
**UZ Leuven CSSD stakeholders** (Central Sterile Services Department) watching the
run and deciding whether ORC is worth pursuing. They are the people who live with
instruments going missing and only finding out downstream, too late to act. The
operator drives; the room judges. Every screen therefore has to be both operable
up close and legible from across the room.

## Product Purpose

ORC is a computer-vision demo that tracks surgical instruments in the sterile
field, live, during a simulated procedure. Three screens — **Setup → Recording →
Report** — and three buttons — **Start, Stop, New recording**. It detects every
instrument on the table, tracks each one as it is picked up and carried out of
frame and brought back (any order, any number of times), and after Stop reports,
per instrument, its **Usage** (the windows it spent off the table) and its
**Completeness** (present on the table at Stop, or missing).

Success is a single, unforgiving bar: **the report matches what physically
happened in the run**, and the stakeholders come away believing the detection and
tracking are accurate. This is a proof of concept for a controlled in-vitro setup,
not a shipping product — recording history across sessions is an explicit non-goal
for the MVP.

## Positioning

Every instrument on the sterile table is seen and accounted for — a missing tool
is surfaced **at the table, the moment the procedure ends**, not discovered
downstream in CSSD when it is already too late to act.

## Brand Personality

Calm, precise, clinically clear. The interface is quiet on purpose: the live
camera feed and the data do the visual work, and the UI stays out of the way of
the only question that matters — *is every instrument accounted for?* Confident
without being flashy; understated rather than salesy.

The single impression the demo should leave a watching stakeholder is **calm,
precise, and trustworthy** — the count and the report are believable precisely
because nothing on screen looks hand-wavy or performative.

The identity is *quietly* branded. The rainbow ring — from the logo, echoed in
the Track "halo" glow — is the one sanctioned brand signature, and it appears the
moment the product does. The register is product with **room to flourish**:
product clarity is the floor on routine controls, but a hero moment (arming Track,
a first-run or empty state) may carry brand-grade polish. Flourish is reserved for
those moments; it never leaks into buttons, badges, tables, or the detection
overlay.

## Anti-references

Documented in [`BRANDING.md`](BRANDING.md) §"Not our look" — the things ORC must
never become:

- A rainbow-themed dashboard: rainbow buttons, navigation, badges, or multicolour
  charts by default. The full spectrum belongs to the logo, never to routine UI.
- Bright cyan set with white type (insufficient contrast).
- Cream, beige, or heavily tinted page backgrounds.
- Large soft gradients behind routine UI; the feed and the data should carry the
  visual weight.
- Excessively rounded, floating "glass" cards.
- Anything that reads as a flashy hackathon demo rather than a credible clinical
  instrument.

## Design Principles

1. **The report must match reality.** Correctness of what is shown is the whole
   product. Never dress up, imply, or interpolate data the engine cannot back —
   the word is "missing", not "lost", because the camera can only observe absence.
   Trust is earned by accuracy, not polish.
2. **Quiet clinical confidence.** Roughly 90–95% white and neutral; teal and the
   semantic status colours carry the remaining 5–10% and always mean something.
   The camera and the data are the show; the chrome stays calm so nothing
   competes with the instrument check.
3. **Legible from across the room.** One person operates it, but a room reads it.
   Counts, live state, the Start gate, and the report must be unmistakable at a
   glance and at distance. Never rely on colour alone — pair every status with a
   label, icon, or shape.
4. **Say why, not just what.** Gated and blocked states explain themselves in
   plain words (why Start is disabled, what the health banner means), no jargon
   and no mystery. The 3-screen, 3-button workflow should teach itself.
5. **Quietly branded, honestly scoped.** The ring is the one brand moment; keep
   routine controls restrained. Build demo-first for 2026-07-20 without
   foreclosing a future product — no throwaway hacks, but no premature product
   scaffolding either. The horizon is deliberately open.

## Accessibility & Inclusion

The working bar (concrete ratios per [`BRANDING.md`](BRANDING.md); no formal WCAG
level has been committed):

- Body text ≥ 4.5:1 against its background; large text and important icons ≥ 3:1.
  Placeholder and supporting text held to the same 4.5:1 — no pale-grey-on-tint.
- Never communicate presence, absence, or an error by colour alone. Every status
  is paired with its label, icon, or shape.
- Pale logo colours are never used as text on white; the bright focus cyan is an
  outline only, never a fill or text colour.
- Reduced motion is honoured: the Track halo and every animation have a
  reduced-motion path (crossfade or instant) that preserves the lit-vs-off
  readiness signal without movement.
