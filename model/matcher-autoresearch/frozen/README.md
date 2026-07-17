# `frozen/` — never edit

This package is the frozen half of the split mandated by `program.md` §3. It
owns:

- **`loader.py`** — reads `model/data/instruments`, `model/data/other_objects`
  (read-only). Hands out raw, unmasked bbox-crops + aligned binary masks —
  masking strategy is a method-level choice.
- **`cv_split.py`** — leave-frame-group-out CV split (5-seed averaging) +
  return-window chunking for multi-frame voting.
- **`holdout.py`** — the deterministic locked holdout (last 3 frames/instrument
  for query, first 5 for gallery; last foreign source image). Not seed-shuffled,
  not touched during CV/selection.
- **`interface.py`** — the fixed `build_gallery / score / accept` contract
  (mirrors `linker-design.md` §6) every `method/` variant must implement.
- **`eval.py`** — `run_cv(...)` for selection, `run_locked_holdout(...)` for the
  one-time champion report. Calls a method ONLY through the interface.

**Rules** (program.md §3, §8):

- A challenger may add files under `method/` or `runs/<experiment>/` — never
  here.
- `score()` / `accept()` must never see a query's ground-truth identity;
  `eval.py` enforces this structurally (it keeps the true label eval-side only).
- `run_locked_holdout` may be called only AFTER hyperparameters are fixed from
  `run_cv` — never inside a search loop, never to pick a threshold.
- Any suspiciously large jump vs the champion is presumed a leak until a Codex
  review clears it (program.md §3).

If a challenger's method family genuinely needs the interface to change (e.g.
a set-to-set matcher that doesn't fit `build_gallery` cleanly), that is a
finding to write up in `FINDINGS.md`, proposing the change — not a same-run
edit here.
