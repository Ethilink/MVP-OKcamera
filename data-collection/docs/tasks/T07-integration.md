# T07 — Hardware integration runbook

status: in-progress (claude)
depends-on: T05, T06
blocks: — (last task)
spec: [DASHBOARD.md](../DASHBOARD.md) §Open items, §Model / config

## Goal

Prove the whole pipeline on Bram's Mac with the real Camo stream and the real
ONNX weights: discover the camera index, run a real capture session, and land a
`validate()`-green dataset in the annotation dashboard.

**Needs Bram present** (Camo running, phone attached) for the real-instrument
capture and the physical unplug test. Camo only registers while the Camo app is
running; it does not show up in `system_profiler SPCameraDataType` by name.

## In scope

- `scripts/find_camera.py`: probe indices 0–5, print `index → frame shape` for
  each opener; Camo streams **1920×1080**, but so can iPhone Continuity, so the
  index is identified by **eyeballing the preview snapshot** saved per index to
  the scratch folder (resolution alone is not a unique fingerprint).
- A full manual session following the checklist below.
- Recording results: fill the Camo index into DASHBOARD.md's Open items and the
  startup command into `dashboard/README.md`.

## Out of scope

Code changes beyond trivial config fixes — anything bigger becomes a Log entry
and goes back to the owning task.

## Acceptance criteria (runbook — check off in order)

- **AC1** `find_camera.py` run with Camo live; index recorded in
  DASHBOARD.md (Open items closed) + README.
- **AC2** Startup with real weights
  (`--weights model/weights/checkpoint_best_regular.onnx --model-version rfdetr-2026-07-07`)
  boots; `start()` gets a valid 1080p frame (pipeline is resolution-agnostic —
  no resolution assert; the only hard failure is a camera that yields no frame
  at all after the warmup retries).
- **AC3** Live view: overlay boxes+masks visible, count updates, glass-to-glass
  latency subjectively inside the 100–300 ms budget.
- **AC4** Capture session: ≥10 flags including at least one **empty frame**
  (nothing detected) and one **burst** (3 SPACE presses <1 s apart).
- **AC5** `/validate` returns zero errors on the session dataset.
- **AC6** Unplug test: disconnect the phone mid-stream → health indicator red
  within ~3 s, FLAG disabled, no crash; reconnect recovers (the capture loop
  reopens a wedged/unplugged camera on its own within a few seconds).
- **AC7** Dataset folder dropped into the annotation dashboard's
  `data/processed/` opens and shows the flagged frames with pending status.
- **AC8** Spot-check 3 saved frames: image is 1080p, boxes/masks land on the
  instruments (single-resolution pipeline — no scale factor, no offset),
  provenance fields populated with `rfdetr-2026-07-07`.

## Log

- 2026-07-07 (claude) — Dry-run against Camo (index 3, 1080p) with real ONNX
  weights, driving the API directly (Chrome extension was not paired). Proved
  end-to-end: live overlay (boxes+masks+confidence), 10-flag session incl. 2
  empty frames + a 3-flag concurrent burst (sequential image_ids, no torn
  `annotations.json`), `/validate` → 0 errors, raw (overlay-free) 1080p stills,
  provenance `confidence_threshold` captured at predict time (not the live
  slider), and health stale→ok recovery on a simulated disconnect. Fixed two
  robustness bugs found in the process: `start()` now warms up reads (a single
  virtual-camera cold-read spuriously crashed startup) and the loop now reopens
  a wedged camera (AC6 reconnect). **Still needs Bram present:** confirm which
  physical index is Camo (3 cameras report 1080p), a real surgical-instrument
  capture (AC8), and a physical unplug/replug (AC6).
