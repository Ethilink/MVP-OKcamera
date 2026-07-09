# Post-pass runs as an idle-draining queue, not a blocking veil

## Status

**SUPERSEDED (2026-07-09) by
[`0002-keyframe-only-synchronous-stop.md`](0002-keyframe-only-synchronous-stop.md)**
— the all-frames offline post-pass was removed, so the queue this ADR describes
(and all its pause/resume/retry/halt-on-failure machinery) no longer exists.
Kept as historical record of why the queue was introduced. Everything below
described the (now-deleted) queue model.

~~accepted — supersedes the "Detector sharing (settled)" section of `RECORDING.md`.~~

## Decision

The post-pass no longer blocks the tool behind a full-screen veil until it
finishes. Instead, each recording's post-pass is enqueued, and the **queue drains
only while the tool is idle** (not recording), using the **single shared
detector**. Starting a new recording **pauses** the in-flight post-pass at its
current frame and reclaims the detector for the live overlay; the job **resumes**
from that frame when the tool goes idle again. A compact status chip in the top
bar shows the current job's progress, the number of jobs queued behind it, and a
rough ETA, so the operator can record a batch of takes back-to-back and know not
to close the dashboard while work remains.

## Why

RF-DETR ONNX on CPU runs the post-pass at ~0.6 fps — about 50× the clip length
(a 1-min take ≈ ~50 min). The original blocking design would freeze the whole
tool for that entire time after every Stop, making multi-take sessions
impractical. Draining the queue only when idle keeps one detector (a second
instance would double memory and cause CPU contention that slows both live
inference and the post-pass) while still letting the operator record continuously;
pause/resume is what makes immediate re-recording possible instead of a ~50-min
wait between takes.

## Considered and rejected

- **Two detector instances** (live + post-pass concurrently) — rejected: doubles
  memory and the CPU contention makes both paths crawl on this machine.
- **Wait for the current job before allowing a new recording** — rejected: a
  ~50-min wait between takes defeats the batch-recording workflow.

## Consequences

- A take's detections are not available while later takes are still being
  recorded — they all process once the operator stops. Acceptable for the
  record-a-batch-then-let-it-grind rhythm of a collection session.
- The post-pass loop must be pausable/resumable by frame index. This stays safe:
  an interrupted or crashed job leaves a valid *partial* video project and is
  re-runnable from the MP4 via Retry (idempotent by construction).
- The main lever on the underlying slowness is the CoreML/Neural-Engine execution
  provider for the detector (see `docs/tasks/REDESIGN.md` §S1 — measured 2.4×,
  ~50× → ~15–25× clip length). That helps but does **not** retire this queue:
  a 1-min clip is still ~20 min, so idle-draining + pause/resume stay load-bearing.
- While a job drains, the single detector is borrowed, so the live overlay is
  boxless. A mid-drain **image-mode flag** does not accept that gap: `/flag`
  reclaims the detector for one predict on the frozen frame
  (`capture.predict_now`), serialized on the same detector lock as the drain so
  it never overlaps a drain predict (INV-3) — effectively a one-frame pause of
  the drain. Chosen (U4, Bram) over 409-while-draining and accept-boxless,
  because a silently empty/stale label poisons the active-learning set. See
  `docs/RECORDING.md` §Queue model.
