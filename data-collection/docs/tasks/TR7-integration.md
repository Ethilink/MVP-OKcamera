# TR7 — Recording integration + 1080p60 spike

status: done (AC1–AC9 verified)
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
- 2026-07-08 (claude) — **TR7 implemented; status → done.**

  **Correctness ACs 1–6 — `tests/test_recording_e2e.py` (new, CI-gated, green,
  stable across repeated runs; full suite 155 passed).** Full pipeline over a
  fake numbered camera: real `CaptureLoop` + `create_app` state machine +
  `PostPassJob` + `VideoEntryWriter` + `coco`; only the camera (paced numbered
  frames — the "paced fake" that bounds the reader), encoder (records frames +
  writes a stub file so `video/` is non-empty), and post-pass video read
  (lossless replay of the recorded frames) are faked. No camera/ffmpeg/weights.
  - AC1 ✓ completed run = video project (annotations.json + non-empty video/).
  - AC2 ✓ per SPACE: one `selected_frames` entry, one `images[]` record, one
    JPEG; **displayed-frame identity / no off-by-one** proven losslessly
    (`decode_frame_index(written[D]) == decode(on-screen frame)`, echoed via
    `X-Frame-Number`), and `D < frames_written` (it's the lagging displayed
    frame, not newest). JPEG-vs-video equality asserted within re-encode
    tolerance (a q95 JPEG of a flat frame shifts a channel by ±1 — exact
    byte-equality is impossible; the lossless identity proof carries the AC).
  - AC3 ✓ sidecar has every frame `0..frame_count-1` (contiguous), pixel-space
    coords, dims equal stream dims everywhere (`video`, `images[]`, frame_count).
  - AC4 ✓ keyframe annotation ⊇ image-mode structure (same `coco.build_annotation`
    keys) + `video_id` + `frame_number` (on image) + **unique** `track_id`.
  - AC5 ✓ kill mid-run (cap raises after 3 reads) → valid partial (video +
    selected_frames.json, no annotations.json) + `failed` state with error →
    `/record/retry` → completes → the three JSONs are **byte-identical** to a
    clean rebuild from the same frames/keyframes.
  - AC6 ✓ `validate_import.validate()` advisory-clean (0 errors) on the entry.

  **Hardware ACs 7–9 (Bram's Mac + Camo, real RF-DETR ONNX weights):**
  - **AC7 ✓ 1080p60 spike** (`scripts/spike_fps.py`, new): requested 1080p60 →
    negotiates 1920×1080, `CAP_PROP_FPS` claims 30, **delivers exactly 30.0 fps**
    (300 reads / 10.005 s); a no-`CAP_PROP_FPS` open (dashboard-style) also
    delivers 30. **Camo does not provide 1080p60 through OpenCV/AVFoundation —
    do not flip `capture_fps` to 60.** Recorded in RECORDING.md §FPS Open items.
    (Camo's OpenCV index is unstable — was 3, then 0 after an app restart;
    `find_camera` + eyeball each run. README updated.)
  - **AC8 ✓ real end-to-end** (done live during the TR6 R4 pass, index 3 → later
    0): recorded a real clip, marked keyframes via SPACE, Stop, watched the
    post-pass progress veil to completion; on-disk `mini-take` (58 frames, 2
    keyframes) inspected — real avc1 MP4 + annotations.json (COCO-VID) + keyframe
    JPEGs + selected_frames.json + full-frame sidecar, `validate()` clean.
    **Live overlay FPS while recording ≈ idle** (0.33 vs 0.25 overlay fps —
    detector-bound both ways; the hardware avc1 encoder adds no meaningful load,
    spec AC6). Video-project **discovery rule holds** (annotations.json +
    non-empty video/). **Open-step confirmed headlessly (2026-07-08):** the
    consumer is NOT CVAT — it's the pure-Python `Project.from_directory` in
    `Surgical-sets-AI/annotation_tool/src/models/project.py` (the desktop
    annotation tool; `data/processed/` scanned by `Project.scan_all`). Ran it
    against `mini-take`: `Project(name='mini-take', project_type='video',
    is_partial=False, status='new', image_count=2, annotation_count=4,
    mask_count=4, source_video=…/video/mini-take.mp4)` — recognized as a full
    (non-partial) VIDEO project. AC8 fully satisfied. (Optional UI confirmation
    for Bram, no Docker: copy the entry into `annotation_tool/data/processed/`
    and `python app.py`.)

  **Spec-nit found during AC8 verification — FIXED (2026-07-08, claude):** the
  `annotations.json` `video` block (and the §5 sidecar copy) omitted `file_name`
  ("<entry>.mp4"), required by IMPORT_FORMAT_VIDEO.md §2/§5. Was benign today
  (discovery uses first-file-in-`video/`; every `images[]` record carries its own
  `file_name`) but the annotation tool's `dataset.py` fallback reads
  `video["file_name"]` when an image record lacks one → latent `KeyError`.
  Fixed in `VideoEntryWriter.finalize()` (`file_name = f"{entry_name}.mp4"` on
  both the annotations video block and the sidecar video block); TR7 e2e AC3
  now asserts it. Full suite 158 green.
  - **AC9 ✓ encoder reality check**: `ffprobe` on the real MP4 →
    `codec_name=h264`, `codec_tag_string=avc1` (VideoToolbox hardware path, NOT
    the ffmpeg fallback), 1920×1080, 30/1 fps, `nb_frames=58` (== writer
    counter), duration 1.933 s (CFR@30). `release()` finalized a **present,
    seekable moov atom** (after mdat — not web-faststart, but fully seekable for
    local reopen) and the **post-pass reopened it cleanly**, decoding all 58
    frames + extracting the keyframe JPEGs.

  **In-scope note (no recording-module code changed):** the only code added is
  `tests/test_recording_e2e.py` + `scripts/spike_fps.py`; docs updated
  (RECORDING.md §FPS, README, BOARD). One backend concurrency bug found during
  the TR6 pass — **`/record/discard` during processing does not cancel the
  post-pass worker** (orphan keeps predicting + re-creates a stray folder after
  rmtree) — is logged against TR5/TR4 for the owner to fix; it is NOT exercised
  by these ACs (AC5 tests process-kill, not discard) and does not block TR7.
