# TR7 — Recording integration + 1080p60 spike

status: todo
depends-on: TR5, TR6
blocks: — (last recording task)
spec: [RECORDING.md](../RECORDING.md) §Acceptance criteria (1–8), §FPS — 30 vs 60 (spike), §Encoder

## Goal

Prove the whole recording pipeline end-to-end with a **fake numbered camera** (no
hardware needed for the correctness ACs), then do the one thing that genuinely
needs Bram's Mac + Camo: **measure** the delivered rate at requested 1080p60 (the
spike that must land before anyone flips `capture_fps` to 60), and run the real
end-to-end capture on real weights.

## In scope

- `tests/test_recording_e2e.py` — **new**: a full record → keyframe → stop →
  post-pass run driven by a fake numbered camera + `FakeDetector`, asserting
  RECORDING.md ACs 1–5 and 7 end-to-end (the numbered frames make off-by-one
  detectable). This is the automated correctness gate; it needs no camera.
- `scripts/spike_fps.py` — **new**: open the camera at requested 1080p60, time
  300 `cap.read()`s, print the **delivered** fps + negotiated frame shape.
- Runbook: record the measured 1080p60 rate + the Camo behaviour into
  RECORDING.md §FPS Open items and the recording startup command into
  `dashboard/README.md`.

## Out of scope

Code changes to the recording modules — anything beyond a trivial config fix goes
back to the owning TRx task as a Log entry. The image-mode hardware runbook is
T07's.

## Acceptance criteria

Correctness ACs (**AC1–AC6**) run with fakes — no camera, no ffmpeg, no weights —
so they gate in CI. Hardware ACs (**AC7–AC9**) need Bram + Camo.

- **AC1 (video project on disk — spec AC1).** A completed fake-camera run produces
  an entry that satisfies the video-project discovery rule: `annotations.json`
  present + non-empty `video/`.
- **AC2 (keyframe fidelity — spec AC2/AC3).** Each SPACE during the run yields
  exactly one `selected_frames` entry, one `images[]` record, and one JPEG; the
  JPEG's pixels equal the video decoded at that `frame_number`; and the marked
  frame is the one that was **displayed** at press time (echo the `X-Frame-Number`
  through `/keyframe` and assert the numbered-frame identity round-trips — no
  off-by-one, no newest-at-press-time lag).
- **AC3 (sidecar completeness — spec AC4).** `full_frame_detections.json` contains
  every frame `0..frame_count-1`, empty frames included, coordinates in original
  video pixel space; dims equal the stream dims everywhere (`video`,
  `images[].width/height`, RLE `size`).
- **AC4 (keyframe == image-mode structure — spec AC5).** For one `sv.Detections`,
  a keyframe annotation matches image-mode structure (shared helper) plus
  `video_id` / `frame_number` / unique `track_id`.
- **AC5 (kill → partial → retry — spec AC7).** Kill the post-pass mid-run: the
  entry is a valid partial project (video + `selected_frames.json`, no
  `annotations.json`); `/record/retry` completes it; the result is byte-identical
  to an uninterrupted run.
- **AC6 (validator advisory — spec AC8).** `scripts/validate_import.py`'s
  `validate()` runs advisory-clean on the finished entry (noting it does not yet
  check the video-specific contract — import doc §7).
- **AC7 (1080p60 spike — needs Camo).** `spike_fps.py` run with Camo at requested
  1080p60 reports the delivered fps + negotiated shape; the number is recorded in
  RECORDING.md §FPS Open items. (Do not flip the default to 60 — this only
  measures.)
- **AC8 (real end-to-end — needs Camo + weights).** With real weights + Camo at
  default 30 fps: record a short clip, mark a few keyframes, stop, watch the
  post-pass progress to completion, and confirm live overlay FPS while recording ≈
  idle FPS (spec AC6 — encoder adds no meaningful load). Drop the entry into the
  annotation dashboard's `data/processed/` and confirm it opens as a video project.
- **AC9 (encoder reality check).** Confirm the recorded file is genuinely H.264 MP4
  (which backend opened — `avc1` VideoWriter or the ffmpeg fallback) and that
  `release()` finalized a seekable moov atom (the post-pass reopened it cleanly).

## Log

- 2026-07-08 — Brief created (recording-mode decomposition of RECORDING.md, task
  cut T-R7).
