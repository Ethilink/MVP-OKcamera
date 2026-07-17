# Demo video validation

This is the durable recording and replay procedure for the session linker.
Generated detection caches and JSON traces are local artifacts and are not
versioned.

## Replay the July 15 recordings

Run from the repository root:

```bash
model/.venv/bin/python model/scripts/replay_session.py \
  --video matching/data/testing/15-07-26-001/videos/15-07-26-001.mp4 \
  --out /tmp/orc-take-a \
  --stride 6

model/.venv/bin/python model/scripts/replay_session.py \
  --video matching/data/testing/15-07-26-002/videos/15-07-26-002.mp4 \
  --out /tmp/orc-take-b \
  --stride 6
```

The uncached run exercises RF-DETR, the workspace filter, Deep OC-SORT, and
the SessionLinker. It also writes `<out>.dets.npz`. During linker-only
iteration, reuse those exact detections with `--from-cache --cache <file>`.

Acceptance checks:

- enrollment roster is exactly eight identities, renumbered `1..8`;
- every identity binds to its persistent specimen gallery, or falls back to
  session-only — never to a wrong specimen;
- normal decisions report eight comparison galleries, even with one Missing;
- an Active comparison winner remains Unknown;
- a likely raw-ID handoff during the old track's coast window is deferred and
  revalidated after the identity becomes Missing;
- foreign phone/pen/keys cases never take a Missing identity;
- recycled raw IDs after death go through a new matcher decision;
- genuine returns preserve their original session identity;
- full uncached throughput remains close to the measured 3 fps baseline.

### Choosing which persistent galleries load (`--instruments-dir`)

`replay_session.py` takes `--instruments-dir`, which is forwarded to
`load_tracker()` and decides what T08 binding has to work with:

| value | effect |
|---|---|
| *(omitted)* | the shipped `model/data/instruments` — all 8 specimens, 15 views each |
| a directory holding a **subset** of the `instrument{N}` folders | forces a **partial bind**: some identities bind, the rest fall back to session-only |
| `none` (literal string) | **binding disabled** — no galleries load, everyone is session-only (today's pre-T08 behaviour, plus renumbering) |

The subset and `none` modes are how the Trap-4 atom-count asymmetry below was
measured, and how it can be re-measured. A subset directory is just a folder of
symlinks:

```bash
mkdir -p /tmp/instr-partial4
for n in 1 2 3 4; do
  ln -s "$PWD/model/data/instruments/instrument$n" /tmp/instr-partial4/
done
```

## Video recording checklist

Use this beside the table while filming. The goal is simple: give the full
camera → detector → tracker → matcher pipeline clear examples of instruments
leaving, returning, changing position, and foreign objects appearing.

The names below are stable project IDs, not clinical instrument names:

| project ID | visual cue |
|---|---|
| instrument1 | Large ring-handled forceps |
| instrument2 | Smaller ring-handled forceps |
| instrument3 | Small dark-handled tool |
| instrument4 | Small dark round-handled tool |
| instrument5 | Long angled metal instrument |
| instrument6 | Short broad-ended metal instrument |
| instrument7 | Short T-handled instrument |
| instrument8 | Long thin ring-handled instrument |

## Before every take

1. Use the same overhead camera setup and blue mat as the existing recordings.
2. Put **instrument1** through **instrument8** on the table, in that left to
   right order. Leave a small gap between them.
3. Start recording. Keep both hands out of frame for **3 seconds**.
4. For every instruction below: do the action slowly, then remove your hands
   and hold the new table state for **3 seconds**.
5. Do not talk or hold up cards. The fixed starting order and this checklist
   are the ground truth.

If an action goes wrong, leave the final table state still for 5 seconds,
reset it, and repeat that action. Do not discard the whole video.

## Take A — returns and position changes

Start with all eight instruments present.

1. Leave the full table still for 3 seconds.
2. Put a hand over **instrument1** for about half a second; do not pick it up.
3. Remove **instrument2**. Wait 3 seconds. Put **instrument2** back in its
   original place.
4. Remove **instrument5**. Keep it out of view for **12 seconds**. Put
   **instrument5** back in its original place.
5. Remove **instrument1**, **instrument6**, and **instrument7**. Wait 3
   seconds.
6. Put back **instrument6** only. Wait 3 seconds.
7. Put back **instrument1** and **instrument7** at the same time, each in its
   original place.
8. Remove **instrument3** and **instrument4**. Wait 3 seconds.
9. Put them back in the **opposite places**: instrument3 where instrument4
   was, and instrument4 where instrument3 was.
10. Remove **instrument8** and end the video without returning it.

## Take B — random objects and difficult returns

Reset the table: all eight instruments back in their original left-to-right
order. Start a new video and hold still for 3 seconds.

1. Remove **instrument4**. Put a **phone** in its empty place. Remove the
   phone. Put **instrument4** back.
2. Remove **instrument5** and **instrument6**. Put two random objects (for
   example a pen and keys) in their empty places. Remove both random objects.
   Put **instrument5** and **instrument6** back.
3. With all eight instruments present, put one random object in empty space
   beside the row. Remove it again.
4. Remove **instrument1** and **instrument2**. Put them back in their own
   original places.
5. Remove **instrument1** and **instrument2** again. Put them back in the
   opposite places.
6. Remove **instrument8**, turn it over outside the camera view, and put it
   back in its original place.
7. Remove **instrument3**, turn it over outside the camera view, and put it
   back in its original place.

## Most important if time is short

Film Take A steps 1–7, then Take B steps 1–3. Those give us the clearest
evidence for normal matching and rejection of objects that are not in the
instrument set.

## After filming

- Name the files `take-a.mp4` and `take-b.mp4`.
- Keep the original video files; do not trim or speed them up.
- Make a note of any repeated or mistaken action.
- Record any repeated or mistaken action alongside the file before scoring.

## Latest results

Validated on 2026-07-15 with stride 6 (1,226 sampled frames for Take A; 1,862
for Take B). RF-DETR was run uncached after the detector post-processing fix.
The final linker-only coasting-handoff change was then replayed from those exact
detection caches; no detector output was regenerated or altered between passes.

**Re-validated after T08 gallery binding** (2026-07-15) by replaying both takes
from those same detection caches. Every count below reproduces exactly; the two
latency/throughput rows are pre-T08 uncached measurements and are kept as
recorded, because milliseconds drift with machine load (see § "Synchronous-path
latency").

| result | Take A (`001`) | Take B (`002`) |
|---|---:|---:|
| frozen roster | 8 | 8 |
| identities bound to a specimen (T08) | 8 of 8 | 8 of 8 |
| identities left session-only (T08) | 0 | 0 |
| full-roster comparison violations | 0 | 0 |
| decision batches | 24 | 46 |
| coasting handoffs deferred, then revalidated | 0 | 7 |
| stale raw-ID resurrection shortcuts | 0 | 0 |
| maximum matcher batch latency *(pre-T08, uncached)* | 90.3 ms | 142.6 ms |
| full uncached throughput *(pre-T08)* | 2.38 fps | 2.88 fps |

### T08 gallery binding — measured 2026-07-15

Five cached replays: `take-a-t08` and `take-b-t08` (all 8 galleries),
`take-b-partial4` (4 of 8, forced partial bind), `take-b-nogal` (binding
disabled), `take-b-clean` (quiet-machine latency). All `--from-cache`, so real
OC-SORT and a real `SessionLinker` ran against byte-identical detections.

**Everything that had to hold, held.** The roster freezes at 8 on both takes and
renumbers to `{1..8}`; all 8 identities bind on both takes, `session_only=[]`;
`comparison_galleries=8` on every batch log (0 full-roster violations); batch
counts stay at exactly **24** (Take A) and **46** (Take B), unchanged from the
pre-T08 baseline; the six Take B foreign tracks all still settle Unknown (t=29.2,
122.2, 122.4, 123.4, 123.8, 165.0 s); and both fail-closed rejects still fail
closed (302.0 s → `{71: 'linked:1', 72: 'unknown'}`, 368.0 s → `{88: 'unknown'}`).

**Renumbering is what makes the tray read 1..8.** The raw OC-SORT ids at the
freeze are non-contiguous — Take B froze `{3,5,7,9,10,11,12,14}`. Renumbering
(`linker-design.md` §3) maps them onto `{1..8}` whether or not any gallery loads.

#### What binding actually buys, on these takes: one re-identification

Be honest about the size of this. Comparing `take-b-t08` against `take-b-nogal`,
binding converts **exactly one** Unknown into a correct re-identification:

- **t = 261.0 s, raw track 60** — `'unknown'` with no galleries → `'linked:2'`
  with them.

That single event *is* the entire delta: 24 vs 23 links, 16 vs 17 unknowns. It is
the only behavioural difference binding makes across both takes. The other value
of T08 is definitional rather than measured — "Instrument 3" now means specimen
3, stable across recordings, which is what the demo narrative needs and what no
replay metric scores.

#### Trap 4 — atom-count asymmetry: measured, no mitigation needed

SRC is sensitive to how many atoms each candidate contributes. Atom spread per
comparison gallery, read off the batch logs (`atom_counts`):

| configuration | atoms per gallery | spread |
|---|---:|---:|
| no galleries (pre-T08 baseline) | 2..5 | 2.5× |
| all 8 bind (both real takes) | 17..20 | **1.18×** — better than today |
| mixed bind (forced 4 of 8) | 2..20 | **10.0×** — 4× worse than today |

The build brief's prediction is **confirmed**: a mixed bind is the bad case, and
it is meaningfully worse than the imbalance we ship with today.

**And it changes nothing.** `take-b-partial4` vs `take-b-t08`: **0 decision
differences across all 46 batches**. The identity numbers differ (the four
unbound identities take reserved session-only numbers), but the permutation is a
**consistent bijection at every timestamp** — full8→partial4:
`1→1, 2→2, 3→3, 4→4, 5→6, 6→8, 7→5, 8→7` — so every link lands on the same
*physical* instrument. A 10× dictionary-size bias does not flip a single
decision: the champion's absolute `cos_tau=0.60` gate plus `bind_tau` dominate
SRC's relative SCI bias.

**Honest limits.** This is one take and one partial configuration (specimens 1–4
— a natural split, not an adversarial one). The mechanism Trap 4 describes is
real; what is measured is that it currently has no decision-level consequence.
**Re-check if `tau`, `bind_tau`, or `cos_tau` ever move, or if a real take ever
produces a mixed bind** — neither take has yet.

#### Trap 5 — persistent galleries are embedded exactly once

Confirmed. All 8 specimens × 15 views embed in `load_persistent_galleries()` at
`load_tracker()` time: **2,926 ms, once, at startup**. Never per freeze, never
per event. `reset()` keeps them. The enrolment freeze pays only for its setup
crops plus the binding solves; recording Start happens later without a reset.

#### Negative finding — §8 is NOT subsumed by §3's real poses

The build brief hypothesised that "§3's 15 real poses largely subsume §8"
(rotation/mirror augmentation) and asked for evidence either way. **The evidence
is negative.** With all 15 persistent views bound, Take B's 368.0 s flipped
instrument **still settles Unknown** rather than identity 6. §8 remains a real,
unaddressed gap. It fails safe — a false reject, never a wrong link — so it does
not block the demo.

#### Tuning signals to know before the live camera test

> **Flag — instrument 3's Take A bind is razor-thin.** Its bind scores are
> `(0.3007, 0.0928)` against `bind_tau = 0.30`: it cleared by **0.0007**, i.e.
> **0.23%**. It binds today, but any drift in lighting or pose sends it to
> session-only. This is **not a bug** — degrading to session-only is exactly the
> design, and it degrades safely — but it is the thinnest number in the build and
> worth knowing before T09. Take B's instrument 3 scored **0.4683**, comfortably
> clear. **`tau` was not lowered**, and must not be.

> **Flag — `bind_margin` is doing almost no work.** Nearly every second-best bind
> score is **0.0**, because the champion's K=1 `cos_tau = 0.60` gate zeroes
> hopeless candidates before the margin is consulted. The absolute `bind_tau`
> gate carries the decision essentially alone. Anyone tuning `bind_margin` at
> T07/T09 expecting leverage should read this first.

> **Flag — the freeze log's bound mapping is tautological, and should not be.**
> B-O1 asked for the bound mapping as `{session_id: specimen}`. But B-N1 makes
> `session_id == specimen` for every bound identity, so the line always reads
> `{1:1, 2:2, ...}` and reveals nothing. The code matches the spec exactly; the
> spec asked for the wrong thing. What a debugger actually wants is
> `raw_id -> specimen`: which raw tracker id claimed which photo set. One extra
> line, real diagnostic value at T07/T09. **BUILT 2026-07-16** — the freeze log
> now carries `raw_binds={raw_id: specimen}`, and the `orc-demo --debug` console
> renders it as `Instrument N ← raw track R  specimen S  score X`.

### Gallery bake-off — session 1 vs session 2 vs merged-30 (2026-07-16)

Two capture sessions of the same 8 specimens exist: `model/data/instruments/`
(**session 1**, the shipped default, all T08 numbers) and
`model/data/instruments_session2/` (**session 2**, a day fresher, previously
undocumented). T07 confirmed both are the demo-tray specimens. To settle whether
to bind session 1, session 2, or a merged 30-view set, all three were replayed
from cache on both takes (`--instruments-dir`).

**All three bind 8/8, roster `{1..8}`, 46 batches (B) / 24 (A), foreign-reject and
both fail-closed cases intact.** They differ in only two ways:

| | session 1 (shipped) | session 2 | merged 30 |
|---|---:|---:|---:|
| instrument 3 bind headroom over τ=0.30 (Take A) | **+0.0007** | +0.0819 | +0.0958 |
| instrument 1 return, Take A t=180.4s | `linked:1` ✅ | `unknown` ❌ | `unknown` ❌ |
| Take A links / unknown | **20 / 6** | 19 / 7 | 19 / 7 |
| Take B decisions | (baseline) | identical | identical |

**Verdict: keep session 1. Do not merge.** Merging rescues instrument 3's
knife-edge bind but **loses a genuine return re-identification** (instrument 1 at
180.4s). Both failures are fail-safe, but they differ in demo cost: session 1's
thin bind degrades to session-only (audience-invisible, instrument still tracks
and returns on Start crops), whereas merged's lost return shows "Unknown" on
camera at the exact "it left and came back as itself" moment. The flipped-instrument
§8 reject at 368s stays Unknown even with 30 views — more views do not subsume a
flip. Session 2 / merged-30 are post-demo experiments; the merged builder and all
six traces are in the session scratchpad (volatile).

### Offline cold-start — verified (2026-07-16)

The venue has no internet, so a cold model cache would hang `load_tracker()` at
startup (the models are lazily fetched). Forced-offline construction
(`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`) **succeeded in 16.6 s** — DINOv2 +
the mobilenet embedder + the ONNX detector all cache-resident, 8 galleries loaded,
CoreML+CPU. ⚠️ Verified on *this* machine; re-run with the network actually off on
the demo box if it differs. This is [wayfinder T09](../../docs/wayfinder/session-linker/tickets/T09-first-live-check.md)'s
gating pre-flight.

### Synchronous-path latency — both halves (re-measured after T08, 2026-07-15)

`update()` does its matcher work synchronously. Two costs sit on that path, and
they are timed by **separate** counters — an earlier pass quoted only `total_ms`
(the solve) and concluded §9 async was unnecessary. That conclusion holds, but
only once the embed is timed too, since the embed is the half everyone assumed
was expensive.

Measured on `take-b-clean`, a quiet-machine cached replay with all 8 galleries
bound:

| synchronous cost | counter | n | median | max | was (pre-T08) |
|---|---|---:|---:|---:|---|
| SRC solve (score + assign) | `total_ms` | 46 | 43.1 ms | 79.3 ms | 40.1 / 70.1 ms |
| embed at track death | `build_ms` | 26 | 55.0 ms | 82.6 ms | 41.0 / 67.3 ms |
| embed at enrolment freeze | `build_ms` | 1 | 495.9 ms | — | 260.4 ms |

**§9 stays closed, still by measurement.** The enrolment freeze roughly doubles —
260 → 496 ms, the one real cost of binding — but lands **2.0× inside the 1.0 s
resolve contract**, on a still table with nothing yet tracked. Do not build
threading.

Two things worth reading off this table:

- **The SRC solve barely moved** (43 vs 40 ms) even though the dictionary grew
  3–8×. Persistent views cost far less at solve time than their atom counts
  suggest — the MultiTaskLasso solve is not linear in atoms at this scale.
- **The death-embed rise (41 → 55 ms) is machine drift, not T08.** `_kill` is
  untouched and still embeds ≤3 session crops; nothing about that path changed.
  Counts reproduce, milliseconds drift — this is drift, and presenting it as a
  finding would be wrong.

Freeze `build_ms` scales with how many galleries actually load, which is the
binding solves being paid for:

| galleries loaded | freeze `build_ms` |
|---|---:|
| none (`--instruments-dir none`) | 240.1 ms |
| 4 (forced partial bind) | 484.9 ms |
| 8 (shipped default) | 495.9 / 508.9 ms |

`n = 26` deaths against the baseline's 25 is the extra track lifecycle from the
one re-identification binding rescues (t = 261.0 s), not a regression.

This retires the original **"~0.2–0.5 s per link event"** estimate (MAP.md) — a
guess, and ~6× pessimistic. Embeds are ~3 crops, batched, on MPS.

**Reproduce:**

```bash
# shipped configuration (all 8 persistent galleries)
model/.venv/bin/python model/scripts/replay_session.py \
  --video matching/data/testing/15-07-26-002/videos/15-07-26-002.mp4 \
  --out <prefix> --cache /private/tmp/orc-take-b-final-uncached.dets.npz --from-cache

# binding disabled — the pre-T08 baseline
model/.venv/bin/python model/scripts/replay_session.py \
  --video matching/data/testing/15-07-26-002/videos/15-07-26-002.mp4 \
  --out <prefix> --cache /private/tmp/orc-take-b-final-uncached.dets.npz --from-cache \
  --instruments-dir none

# forced partial bind — how the Trap-4 asymmetry is measured
model/.venv/bin/python model/scripts/replay_session.py \
  --video matching/data/testing/15-07-26-002/videos/15-07-26-002.mp4 \
  --out <prefix> --cache /private/tmp/orc-take-b-final-uncached.dets.npz --from-cache \
  --instruments-dir /tmp/instr-partial4
```

Take A is `15-07-26-001` with the `take-a` cache. Read `build_ms` / `total_ms`,
the freeze's bind scores, and each batch's `atom_counts` /
`comparison_galleries` out of the trace's `logs` array.

⚠️ **Compare counts, not milliseconds.** The **batch count** (46 on Take B), the
roster, the bind outcomes, and the per-decision links all reproduce exactly.
**Absolute latency drifts with machine load** — the 142.6 ms in the table above
is from the uncached run competing with the detector, 79.3 ms from a quiet cached
re-run, and the death-embed median moved 14 ms between two runs of identical
code. Never read a millisecond delta as a behavioural change.

Take B's deliberate foreign-object introductions produced six raw tracker
tracks across the phone/pen/keys windows (29.2 s, 122.2–123.8 s, and 165.0 s).
All six settled Unknown; none borrowed a Missing identity. **Still true after
T08**, at the same six timestamps — the larger bound dictionaries did not soften
the open-set reject, which was the risk worth checking.

The 66.6 s raw-ID handoff initially exposed a timing bug: the new raw ID was
decided one frame before the old identity became Missing. The final code defers
that accepted Active-but-absent candidate, then revalidates it at 67.0 s and
links it back to session identity 7. All seven such coasting handoffs in Take B
followed this defer-then-link path.

Two deliberately difficult genuine returns still fail closed in Take B —
**unchanged by T08, with all 15 persistent views per specimen bound**:

- at 302.0 s, the swapped smaller ring-handled instrument returns as raw 72;
  its true identity 2 scores only 0.0688 and the row is Unknown
  (`{71: 'linked:1', 72: 'unknown'}`);
- at 368.0 s, the flipped dark-handled instrument returns as raw 88 and remains
  Unknown instead of identity 6 (`{88: 'unknown'}`).

The 368.0 s case was the explicit test of whether §3's real poses subsume §8's
rotation/mirror augmentation. **They do not** — see the negative finding above.

No wrong identity link was observed in any reviewed trace, before or after T08.
These two false rejects are appearance/gallery limitations, not assignment
failures. Do not lower `tau` or force a one-to-one assignment to hide them: raw
72's best wrong candidate scores 0.2654, already much higher than its true
candidate. The next matcher iteration should benchmark rotation/flip gallery
augmentation (§8) and a KNN or hybrid matcher against these events while
retaining the foreign-object reject floor.

Cached replay speed (15.26 fps Take A, 14.72 fps Take B) is included only as a
linker-regression turnaround metric. It is not detector throughput.
