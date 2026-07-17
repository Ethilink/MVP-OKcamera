# Backend TODO — expose per-detection data during `setup`

**Status:** ✅ implemented (2026-07-13) on `dev/frontend-change` — `/status`
setup/finished now returns `detections[]` with lazily encoded thumbnails outside
the session lock; as of 2026-07-17 masks produce transparent PNG cutouts with a
defensive JPEG fallback. The `--fake` backend churns the count during
setup and draws shapes aligned with the detections; `DetectionConstellation`
renders the crops with an icon fallback. Backend 123 tests + frontend 47 tests
green; `orc-demo --fake` verified end-to-end. Kept below as the design record.
**Raised:** 2026-07-13, during the Halo setup-screen redesign (`dev/frontend-change`).

> **Fake behaviour (updated 2026-07-17):** the count wobbles only during setup
> (ids 6–8 come and go; core ids 1–5 stay present). Recording Start does not reset
> the tracker; a separate fake-only `begin_recording()` signal anchors the clean
> pickup script to the accepted Start while preserving production semantics.

## Why

The redesigned first page (setup screen) shows a floating **"N instruments"** badge
with the detected instruments arranged around it as small cut-out tiles, connected
to the badge by thin connector lines (see the sketch / `LiveScreen` setup layout).

Today the `setup` branch of `GET /status` returns only:

```jsonc
"setup": { "detected_count": 5, "stable_for_s": 3.2 }
```

That is enough for the **count** but not for real per-instrument tiles. The frontend
therefore renders `detected_count` **representative** surgical-instrument icons
(scalpel / scissors / forceps / clamp …) as a stand-in — they are decorative, not the
actual detections.

---

## Proposed design (2026-07-13)

Everything needed already flows through `CaptureLoop._capture_tick`
(`app/backend/backend/capture.py`): the raw frame stays un-annotated (render draws
on a copy) and the tracker output has `xyxy` boxes + `tracker_id`s. We currently
throw that away before publishing `Latest`. So: **no new endpoints and no
per-tile encoding in the capture thread — carry the raw material in the snapshot
and crop lazily at poll time.**

This does **not** add anything to `InstrumentTracker`. The model continues to
return same-frame, row-aligned `xyxy`/`mask`/`tracker_id` data. The app backend,
which already owns the input camera frame, owns all crop geometry, encoding and
HTTP transport. See
[`tracker-interface.md` §Consumer-generated crops and previews](../../model/docs/tracker-interface.md#consumer-generated-crops-and-previews)
and [`DESIGN.md` D17](DESIGN.md).

### 1. `capture.py` — `Latest` carries the raw material

- Add `frame_bgr` (a **`frame.copy()`** — OpenCV does not guarantee the camera
  buffer isn't reused, so the snapshot must own its pixels) and a tuple of
  `(tracker_id, bbox)` pairs, converted to plain Python `int`/`float`
  (no numpy/`sv.Detections` references kept alive).
- Mark the owned NumPy frame read-only before publication; immutability must be
  real, not just a frozen container around a writable array.
- Cost: one 1080p BGR allocation/copy per frame (~6 MB held and copied each
  publication), zero thumbnail encoding in the capture thread. Measure capture
  throughput on the demo machine; optimize only if it materially regresses.

### 2. Thumbnail helper + `main.py` orchestration — crop lazily for `GET /status`

Only in the `setup`/`finished` branch (per the contract, `finished` keeps observing
the table, so it gets detections too):

- Keep OpenCV geometry/encoding out of the HTTP handler. Put it in a focused,
  unit-testable helper module; `main.py` only reads state, asks the helper for
  detection response models, and assembles the response.
- Expand each rectangular bbox by ~15 %, clamp it to the frame, then resize
  while preserving aspect ratio and letterbox/pad into a ~96×96 canvas. Do not
  force a square source crop: long surgical instruments must not lose their
  ends. `cv2.imencode` JPEG → base64 **data-URI**. Work scales with polling
  (2 Hz), not camera fps (10–30).
- Computed **outside the session lock** — the capture thread takes that lock on
  every frame (`_on_frame`), so no image work may run under it. Read phase/count
  under the lock, then `capture.snapshot()` (immutable → thread-safe) and encode after.
- **Defensive per tile:** validate each bbox (finite, ordered, non-zero area,
  clamped); check the `imencode` success flag. A bad crop logs and returns
  `thumbnail: null` — it must never 500 the whole endpoint.

### 3. Contract shape

```jsonc
"setup": {
  "detected_count": 5,
  "stable_for_s": 3.2,
  "detections": [                    // ALWAYS present (may be []), sorted by tracker_id
    {
      "tracker_id": 1,               // matches recording.instruments[].tracker_id naming
      "label": "Instrument 1",       // always a string; single class today
      "thumbnail": "data:image/jpeg;base64,…"   // string | null (null = crop failed)
    }
  ]
}
```

- `tracker_id`, not `id` (consistency with the recording block — Codex agrees).
- Sorted by `tracker_id` → stable tile ordering between polls, no reshuffling.
- Return all current detections. The frontend may display only the first seven;
  the backend contract is not tied to one layout's tile cap.
- Payload ~3–7 KB/tile → roughly 50–70 KB per poll for 7–10 instruments at 2 Hz
  on localhost — fine.
- Update `api-contract.md` §/status, then regenerate frontend types
  (`npm run gen:api` against the running backend).

### 4. `fakes.py` — visible thumbnails in `--fake` mode

`FakeCaptureSource` emits black frames today, so real crops would be black squares
during all frontend dev. Fix: draw simple colored instrument-ish shapes into the
fake frames at `ScenarioTracker`'s scripted positions (same module, ticks are 1:1).

**Codex caught a bug in the naive version:** on recording Start,
`ScenarioTracker.reset()` restarts its clock but `FakeCaptureSource`'s frame counter
keeps running — the drawn shapes would drift away from the detections after Start.
Fix: both fakes share **one scenario-state object that resets together** (shared
epoch), not two copies of the same formula.

### 5. Frontend — `DetectionConstellation.tsx`

Wire tiles to real crops + labels (`<img src={thumbnail}>`); representative icons
stay as the fallback when `detections` is absent/empty or a `thumbnail` is null.
Note: regenerating OpenAPI types alone wires nothing — the component needs a small
props change (today it only takes `detectedCount`).

---

## Codex review (gpt-5.6-terra, 2026-07-13) — verdict summary

| Question | Verdict |
|---|---|
| Lazy crops in `/status` vs. encoding in capture thread | **Agree** — lazy is right at 2 Hz; just keep it outside the session lock |
| Raw frame in `Latest` | **Conditional** — safe only as an owned `frame.copy()`; don't trust `read()` buffer semantics |
| Masked silhouette cutouts (PNG + alpha) | **Originally deferred; implemented 2026-07-17** — the validated mask contract now drives alpha, with JPEG fallback when a mask is missing/empty |
| Fake-mode shared geometry | **Agree, with fix** — needs the shared reset epoch (see §4) |
| `tracker_id` naming / nullable `thumbnail` | **Agree** — `detections` always present, `thumbnail` nullable |
| Failure modes | **Agree if defensive** — per-tile null on failure, `finished` included, tolerate count-vs-tiles skew |

Accepted tradeoffs: possible 1-frame skew between badge (`detected_count`, from the
session) and tiles (from the snapshot) — imperceptible at 2 Hz; frontend must
tolerate `detected_count !== detections.length`. When capture is stalled the tiles
show the last frame's crops — the UI already dims in stalled mode, which covers it.

### Implementation risks Codex flagged

- `test_api.py` constructs `Latest` directly → update its fake snapshots with
  complete crop material. Do not make production snapshot fields optional only
  to preserve an old test fixture; `capture.snapshot() is None` already models
  the not-yet-published state.
- The existing overlay encode ignores `imencode`'s success flag (`capture.py`) —
  thumbnail handling should be stricter than that path.

---

## Tile presentation

The constellation currently caps at **7 tiles** (`MAX_TILES`), while real trays
can hold 10+. `/status` returns every detection sorted by `tracker_id`; the
frontend displays the first seven and the badge shows the true count. The layout
can later be changed to fit more without changing the backend contract.

## Completed follow-up (2026-07-17)

- True silhouette cutouts from masks now use PNG with transparent background.
  A missing, empty, or misaligned mask falls back to the rectangular JPEG crop,
  so preview failure remains isolated per instrument.

## Deferred (polish, after real-camera visual check)

- Caching / single-flight of crops (irrelevant for one localhost client).
- A dedicated crop endpoint (`crop_url`) — rejected: extra route + cache for no gain.

## Frontend seam (already in place)

The setup composition is data-driven and degrades gracefully: with no `detections`
field it falls back to representative icons keyed off `detected_count`. When the backend
adds `detections`, wire the tiles to it (real crop + real label) — the layout, connectors,
and states do not need to change.

Files: `app/frontend/src/components/DetectionConstellation.tsx` (tiles + connectors),
`app/frontend/src/api/schema.d.ts` (regenerated by `npm run gen:api` once the schema
changes), and `app/frontend/src/api/types.ts` (re-exports generated schema types).
