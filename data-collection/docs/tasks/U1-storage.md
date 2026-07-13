# U1 — Backend storage layout + entry auto-suffix (blind-TDD SPEC)

status: done (blind-tdd, 2026-07-09)

Self-contained brief for the blind-TDD run of **U1**. Cut from
[`REDESIGN.md`](REDESIGN.md) §U1 (the frozen plan). Terms are from
[`../../CONTEXT.md`](../../CONTEXT.md): **Output path** (base drop-zone),
**Dataset** (`images/<base>/`), **Entry** (`videos/<base>_NNN/`), **Base name**
(the single Settings field). Owns `backend/app.py` (`/settings`,
`/record/start`) + all of `tests/`. Do **not** touch `dataset_writer.py`,
`postpass.py`, `video_writer.py`, `main.py`.

---

## Goal

Image and video capture targets split under the Output path by mode, same base
name:

```
output_path/
├── images/<base>/          ← Dataset (image mode: N stills accumulate)
└── videos/<base>_001/      ← Entry   (video mode: one take each)
    videos/<base>_002/         counter auto-suffixes, zero-padded to 3
```

A Dataset and an Entry are never the same folder. Today both land directly at
`output_path/<name>/`; U1 introduces the `images/` and `videos/` split and the
video counter.

---

## Frozen endpoint contract (what changes)

| endpoint | change |
|---|---|
| `POST /settings {…, output_path, dataset_name}` | image Dataset resolves to `output_path/images/<dataset_name>/`. `dataset_name` = base name. Collision-reject (409) only when that exact Dataset folder already exists on disk. |
| `POST /record/start {entry_base}` | **request field renamed** `entry_name` → `entry_base`. Mints `output_path/videos/<entry_base>_{NNN}/` — scan `videos/<entry_base>_*` for `max(existing counter)+1`, zero-padded to 3. Response returns the resolved `entry_name` (the `<base>_NNN`). 422 on bad base, 400 on unwritable path, 409 if already recording / no output path / no frame captured. |

Everything else in the recording state machine (`/keyframe`, `/record/stop`,
`/record/discard`, `/record/retry`, `/record/status`, `/flag`, `/status`,
`/confidence`, `/frame`, `/stream`) is **unchanged** by U1.

---

## Implementation seam facts (for the coder — `app.py` only)

- `writer_factory(output_path, dataset_name) -> DatasetWriter` is injected.
  `DatasetWriter` builds its dataset dir as `Path(output_path) / dataset_name`,
  raises `ValueError` on a bad name (422) and `FileExistsError` when that dir
  already exists (409). It only creates the folder on the **first flag**, so a
  configured-but-never-flagged Dataset leaves nothing on disk.
  → Nest images by calling `writer_factory(str(Path(body.output_path) / "images"), body.dataset_name)`.
- `app.state.output_path` stays the **base** Output path (recording reads it to
  build `videos/…`). Do not overwrite it with the `images/` path.
- `/record/start` today builds `entry_dir = output_path / name` then
  `entry_dir/video/<name>.mp4`. New: resolve `name = f"{entry_base}_{n:03d}"`,
  `entry_dir = output_path / "videos" / name`, mp4 at `entry_dir/video/<name>.mp4`.
  `rec.entry_name`, `rec.entry_dir`, and the `post_pass_factory(entry_dir,
  entry_name, …)` call must all use the **resolved** name — `postpass.py` /
  `video_writer.py` reopen `entry_dir/video/<entry_name>.mp4` and build all
  artifacts inside `entry_dir`, so passing the resolved values is all they need.
- Counter scan: match directory names under `videos/` against
  `^{re.escape(entry_base)}_(\d+)$`; next = `max(ints)+1` (or `1` if none / no
  `videos/` dir yet). **max+1, never first-free** — a gap (`_001`, `_003`) still
  yields `_004`. Never reuse or overwrite an existing Entry.
- Base validation is unchanged: reject empty, `/`, `\`, or leading `.` with 422.
  The resolved `<base>_NNN` is always a single valid component.
- The old "entry already exists → 409" guard is **removed** (auto-suffix never
  collides). Keep the `OSError → 400` unwritable-path guard, the no-output-path
  409, the no-frame 409, and the already-recording 409 (state `!= "idle"`).
- Keep every threading rule intact: `/settings`'s writer swap under
  `dataset_lock`; `/record/start` under `recording_lock`; both plain `def`.

## Acceptance criteria (tests named per AC — `test_ac1_…` … `test_ac6_…`)

1. `POST /settings {dataset_name: X}` makes the image Dataset resolve to
   `output_path/images/X/` — stills land at `images/X/images/…` and annotations
   at `images/X/annotations/annotations.json`.
2. `POST /record/start {entry_base: X}` creates `output_path/videos/X_001/`; a
   second start with the same base creates `X_002`; the response body's
   `entry_name` is the resolved `X_001` / `X_002`.
3. Counter survives gaps: with `X_001` and `X_003` on disk, the next start
   mints `X_004` (max+1, not first-free); an existing Entry is never reused or
   overwritten.
4. Base-name validation unchanged (single path component, no leading dot),
   applied to `entry_base`; a bad base → 422 and no folder is minted.
5. Image-mode collision-reject (409) still fires for a genuine re-use — an
   existing `images/X/` Dataset — and does **not** fire merely because the
   `images/` parent already exists from a different Dataset.
6. All prior image-mode and recording tests stay green under `uv run pytest`
   (the three legacy files are migrated to this contract, not deleted).

## Test-writer scope (all of `tests/`)

- Add `tests/test_u1_storage.py` with the six AC tests above, built against
  `create_app` via `TestClient` (camera-free / ffmpeg-free / onnx-free — reuse
  the existing fakes: `_writer_factory` with the real `DatasetWriter`,
  `RecordingStubCapture`, the fake encoder/job factories from
  `test_recording_api.py`).
- Migrate the three legacy files to the new contract so the suite stays green:
  - `test_api.py` — image settings now nest under `images/`: the annotations
    path assertion becomes `images/<name>/annotations/annotations.json`; the
    collision test must create the nested `images/<name>/` folder (or flag once)
    to trigger 409, and prove a bare `images/` parent alone does **not** collide.
  - `test_recording_api.py` — `/record/start` now takes `entry_base`; folders
    are `videos/<base>_NNN/`; a repeat base auto-suffixes (no longer 409 on the
    folder); read the resolved name from the response and derive `entry_dir`
    under `videos/`. Keep every state-machine assertion (stop/discard/retry/
    keyframe/status/concurrency) — only the start contract + paths change.
  - `test_recording_e2e.py` — same start-contract migration in `_run_recording`:
    send `entry_base`, take the resolved `entry_name` from the response, and
    derive `entry_dir = tmp_path/"videos"/resolved`.
- Only `app.py` is the coder's; a test that looks wrong goes back through the
  Log, never a coder-side test edit.

## Log

- 2026-07-09 — Brief cut from REDESIGN §U1 for the blind-TDD run.
- 2026-07-09 — **Done via blind-TDD** (orchestrator-driven). Coder (Sonnet, blind
  to tests) edited only `backend/app.py`: `/settings` nests the writer under
  `output_path/images/`; `/record/start` takes `entry_base`, scans
  `videos/<base>_*` for `max+1` (zero-padded 3), mints `videos/<base>_NNN/`, and
  returns the resolved `entry_name`. `dataset_writer.py`/`postpass.py`/
  `video_writer.py`/`main.py` untouched (they consume the resolved
  `entry_dir`/`entry_name`). Test-writer (Sonnet) added `tests/test_u1_storage.py`
  (AC1–AC6) and migrated the legacy contract tests — `test_api.py`,
  `test_recording_api.py`, `test_recording_e2e.py`, **and `test_discard_cancel.py`**
  (the 4th, missed by the original list, caught by the green gate).
- 2026-07-09 — Reviews: Phase-1.5 test review (Opus) caught a broken `test_ac2`
  (discarded between starts → folder deleted → counter correctly reuses `_001`,
  so asserting `_002` would reject a spec-correct impl) → fixed to finalize the
  first take via stop→idle. Phase-3 code review: **Opus** = no blockers;
  **Codex** = 2 blockers → both fixed: (1) `/status` now reports the base Output
  path via `app.state.output_path` (was leaking the `images/` nesting), locked by
  a regression assert in `test_ac6`; (2) `OSError` from the counter scan now maps
  to the friendly 400 (was a bare 500). **Fable** adjudicated: APPROVE, green
  integrity confirmed (no legacy state-machine assertions weakened). Suite: **164
  passed** (`uv run pytest`). Not committed (awaiting Bram).
- Open notes for downstream: `/status` `output_path` is the base (U3 relabels the
  name field per mode). Unrelated `static/app.js`/`index.html` working-tree edits
  predate this run — not part of U1; don't fold them into a U1 commit.
