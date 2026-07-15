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

- enrollment roster is exactly eight identities;
- normal decisions report eight comparison galleries, even with one Missing;
- an Active comparison winner remains Unknown;
- a likely raw-ID handoff during the old track's coast window is deferred and
  revalidated after the identity becomes Missing;
- foreign phone/pen/keys cases never take a Missing identity;
- recycled raw IDs after death go through a new matcher decision;
- genuine returns preserve their original session identity;
- full uncached throughput remains close to the measured 3 fps baseline.

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

| result | Take A (`001`) | Take B (`002`) |
|---|---:|---:|
| frozen roster | 8 | 8 |
| full-roster comparison violations | 0 | 0 |
| decision batches | 24 | 46 |
| coasting handoffs deferred, then revalidated | 0 | 7 |
| stale raw-ID resurrection shortcuts | 0 | 0 |
| maximum matcher batch latency | 90.3 ms | 142.6 ms |
| full uncached throughput | 2.38 fps | 2.88 fps |

### Synchronous-path latency — both halves (re-measured 2026-07-15, Take B cached)

`update()` does its matcher work synchronously. Two costs sit on that path, and
they are timed by **separate** counters — an earlier pass quoted only `total_ms`
(the solve) and concluded §9 async was unnecessary. That conclusion holds, but
only once the embed is timed too, since the embed is the half everyone assumed
was expensive:

| synchronous cost | counter | n | median | max |
|---|---|---:|---:|---:|
| SRC solve (score + assign) | `total_ms` | 46 | 40.1 ms | 70.1 ms |
| embed at track death | `build_ms` | 25 | 41.0 ms | 67.3 ms |
| embed at enrolment freeze | `build_ms` | 1 | — | 260.4 ms |

Worst case both land on one frame: **~137 ms** of linker work on top of the
detector's ~330 ms — an occasional ~470 ms frame against a 333 ms nominal budget
at 3 fps. A spike, not a stall, and far inside the 1.0 s resolve contract. The
260 ms enrolment freeze is by design: the table is still and nothing is tracked
yet.

This retires the original **"~0.2–0.5 s per link event"** estimate (MAP.md) — a
guess, and ~6× pessimistic. Embeds are ~3 crops, batched, on MPS.

**Reproduce:**

```
model/.venv/bin/python model/scripts/replay_session.py \
  --video matching/data/testing/15-07-26-002/videos/15-07-26-002.mp4 \
  --out <prefix> --cache /private/tmp/orc-take-b-final-uncached.dets.npz --from-cache
```

then read `build_ms` / `total_ms` out of the trace's `logs` array. The **batch
count** (46 on Take B) reproduces exactly; **absolute latency drifts with machine
load** — the 142.6 ms above is from the uncached run competing with the detector,
70.1 ms from the cached re-run. Compare counts, not milliseconds, across runs.

⚠️ **§3 gallery binding ([T08](../../docs/wayfinder/session-linker/tickets/T08-gallery-binding.md))
will move these numbers**: a bound gallery embeds 15 persistent views + Start
crops, not ≤3 crops. Embed persistent galleries **once at enrolment and cache
them**; re-embedding per event would grow the freeze from 260 ms toward seconds
and reopen §9. Re-measure after T08.

Take B's deliberate foreign-object introductions produced six raw tracker
tracks across the phone/pen/keys windows (29.2 s, 122.2–123.8 s, and 165.0 s).
All six settled Unknown; none borrowed a Missing identity.

The 66.6 s raw-ID handoff initially exposed a timing bug: the new raw ID was
decided one frame before the old identity became Missing. The final code defers
that accepted Active-but-absent candidate, then revalidates it at 67.0 s and
links it back to session identity 7. All seven such coasting handoffs in Take B
followed this defer-then-link path.

Two deliberately difficult genuine returns still fail closed in Take B:

- at 302.0 s, the swapped smaller ring-handled instrument returns as raw 72;
  its true identity 2 scores only 0.0688 and the row is Unknown;
- at 368.0 s, the flipped dark-handled instrument returns as raw 88 and remains
  Unknown instead of identity 6.

No wrong identity link was observed in either reviewed trace. These two false
rejects are appearance/gallery limitations, not assignment failures. Do not
lower `tau` or force a one-to-one assignment to hide them: raw 72's best wrong
candidate scores 0.2654, already much higher than its true candidate. The next
matcher iteration should benchmark rotation/flip gallery augmentation and a
KNN or hybrid matcher against these events while retaining the foreign-object
reject floor.

Cached replay speed (15.26 fps Take A, 14.72 fps Take B) is included only as a
linker-regression turnaround metric. It is not detector throughput.
