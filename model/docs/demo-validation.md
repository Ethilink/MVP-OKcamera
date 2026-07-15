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

This section is updated only from a completed replay of the current code. Do
not paste intermediate agent logs here.

Results for the final structural-fix replay are added after both uncached runs
complete.
