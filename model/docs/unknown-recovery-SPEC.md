# SPEC — Recoverable Unknown via change-triggered re-identification

**Status:** ready for implementation handoff

**Date:** 2026-07-16

**Scope:** `model/` only; no app/API/frontend changes

## Goal

An enrolled instrument may receive a new raw tracker ID while a hand covers it.
The first linker evidence window can therefore contain mostly occluded crops and
be rejected as Unknown. When the hand is later removed, the same raw track must
be allowed to match again and recover the instrument's original session ID.

The recovery must be event-driven:

- a materially changed crop may start a new evidence window;
- a newly available target identity may start a new evidence window;
- unchanged settled Unknown tracks must not repeatedly invoke the matcher;
- all existing open-set, eligibility, collision, and roster-safety rules remain
  in force.

Target user-visible behavior:

> A hand-covered object may temporarily settle as gray Unknown. After the hand
> leaves and the linker has collected fresh evidence, it recovers its original
> instrument ID without requiring the operator to lift and replace it.

## Problem in the current implementation

`SessionLinker` currently treats a rejection as terminal for the lifetime of the
raw tracker ID:

1. `_run_batch_decision()` adds every unresolved raw ID to
   `_settled_unknown`.
2. `_process_row()` immediately returns the offset Unknown ID for a member of
   `_settled_unknown`.
3. No later crop is buffered and `matcher.score()` is never called again.

This behavior is explicitly pinned by
`test_b5_5_reject_stays_unknown_permanently`. That test must be replaced: a
settled Unknown remains settled while its evidence is unchanged, but is no
longer permanent.

## Decision

Replace the terminal `_settled_unknown: set[int]` with recoverable per-track
state. A settled Unknown stores the evidence fingerprint and target eligibility
from its last decision. While the raw track remains present, the linker cheaply
compares new crops with that decision evidence.

The track is re-armed as Pending when either condition becomes true:

1. **Material evidence change:** its current usable crop differs materially
   from the crop window used for the last decision; or
2. **Target-set expansion:** a roster identity that was not linkable during the
   previous decision has since become Missing or deferable/coasting.

Re-arming does not link anything. It only opens a fresh normal Pending evidence
window. The existing matcher and batch-assignment path makes the eventual
decision.

There is deliberately no arbitrary maximum lifetime or retry count. A static
Unknown causes no retries because its evidence and candidate set do not change.
Each actual retry advances the decision baseline, and a cooldown rate-limits
successive material changes. This permits recovery after a hand remains in place
for an unpredictable amount of time without continuously scoring foreign
objects.

## State model

| State | Emitted ID | `data["resolving"]` | Meaning |
|---|---:|---:|---|
| Enrolment | current pre-freeze behavior | `True` until settled | Unchanged |
| Active roster identity | session ID | `False` | Unchanged |
| Pending | offset raw ID | `True` | Initial decision or recheck evidence is being collected |
| Deferred | offset raw ID | `True` | Existing coasting-handoff behavior; unchanged |
| Settled Unknown | offset raw ID | `False` | Last evidence was rejected; cheap change observation continues |
| Missing roster identity | no emitted row | n/a | Unchanged |

Transitions:

```text
new raw track ──> Pending ──accepted eligible identity──> Active
                     │
                     ├──accepted coasting identity─────> Deferred ──> Pending
                     │
                     └──rejected/no target─────────────> Settled Unknown
                                                               │
                       material crop change or new target      │
                                                               ▼
                                                            Pending

Settled Unknown ──raw track absent past death threshold──> discarded
```

Unknown remains the safe default. No transition from Unknown to a roster
identity bypasses Pending, the matcher gate, or assignment eligibility.

## Public contract

No public seam widening is required.

- `InstrumentTracker.roster` is unchanged.
- `sv.Detections.tracker_id` keeps the two disjoint ranges.
- `data["resolving"]` keeps its existing meaning: `True` exactly while a
  decision is Pending/Deferred; `False` for settled Unknown and roster IDs.
- The app continues deriving Unknown from roster membership.
- Unknown remains video-only and excluded from Usage and Completeness.

When a settled Unknown is re-armed, it changes from `resolving=False` to
`resolving=True` while its fresh window is collected. If accepted, the emitted
ID flips to the original roster session ID and `resolving=False`. If rejected,
the offset ID remains and `resolving` returns to `False`.

## Internal state

Replace `_settled_unknown: set[int]` with a dictionary keyed by raw tracker ID.
The exact private names are free, but the state must represent:

```python
@dataclass
class _SettledUnknown:
    decision_fingerprint: _CropFingerprint | None
    decision_quality: float
    candidate_ids_at_decision: frozenset[int]
    last_decision_frame: int
    absence_streak: int = 0
```

`decision_fingerprint=None` means the last decision had no usable crop. The
first later usable crop is therefore new evidence.

The settled state does not need its own unbounded crop history. Once re-armed,
reuse `_PendingTrack.crop_buffer` and its existing `maxlen=evidence_frames`.
The first crop that triggers re-arming may seed that fresh buffer, but crops
from the rejected decision window must never be mixed into the new window.

When a Pending/rechecked track settles Unknown again, replace its previous
decision fingerprint, quality, candidate set, and decision frame with values
from the new attempt. Do not append decision history.

### Candidate target set

For retry triggering, the **candidate target set** is the set of identities that
could legitimately receive or defer a link on this frame and have a usable
gallery:

- Missing identities; plus
- Active identities whose raw ID is currently absent inside its death grace
  (`absence_streak > 0`), i.e. the existing deferable/coasting case.

This target set is only a cheap trigger. Matcher comparison remains against the
complete frozen roster dictionary, exactly as today. Acceptance remains limited
to Missing identities; an accepted coasting identity is Deferred; an accepted
fully present Active identity cannot receive the link.

A target-set retry is permitted only when the current candidate target set has
at least one member absent from `candidate_ids_at_decision`:
`current_candidate_ids - candidate_ids_at_decision` is non-empty. Merely
shrinking the set, or keeping the same members, does not trigger a retry. This
also handles one target disappearing while a different target becomes newly
eligible.

This condition covers the tracker-handoff ordering where the new raw track is
first judged before the old raw ID has become Missing.

## Cheap crop-change detector

Do not run DINO or `matcher.score()` merely to decide whether the crop changed;
that would defeat the event gate. Compute a deterministic, low-cost fingerprint
from the already extracted `_Crop`.

### Fingerprint

For one usable crop:

1. Replace pixels outside the instance mask with a fixed neutral gray.
2. Resize RGB and mask to a small fixed shape, initially `32 × 32`.
   Use bilinear interpolation for RGB and nearest-neighbor for the mask.
3. Store:
   - normalized RGB thumbnail (`float32`, `[0, 1]`);
   - normalized boolean mask;
   - the existing crop quality (`mask_area × confidence`).

Build the decision fingerprint from the highest-quality usable crop in the
decision window. This handles the observed case where the visible mask grows
substantially after the hand leaves, while remaining simple and deterministic.

### Material-change rule

Compare the current usable crop with the last decision fingerprint using three
independent signals:

- `appearance_delta`: mean absolute difference between normalized masked RGB
  thumbnails;
- `mask_iou`: intersection-over-union of normalized masks;
- `quality_gain`: `(current_quality / max(decision_quality, 1)) - 1`.

The crop is materially changed when **any** signal crosses its configured gate:

```text
appearance_delta >= unknown_recheck_appearance_delta
OR mask_iou <= unknown_recheck_mask_iou
OR quality_gain >= unknown_recheck_quality_gain
```

Initial values to validate against the KU Leuven recording/reproduction:

| Parameter | Initial value | Meaning |
|---|---:|---|
| `unknown_recheck_cooldown_s` | `0.75 s` | minimum time between completed decisions for one raw ID |
| `unknown_recheck_appearance_delta` | `0.15` | normalized masked RGB mean absolute difference |
| `unknown_recheck_mask_iou` | `0.75` | a mask overlap at or below this is materially different |
| `unknown_recheck_quality_gain` | `0.25` | 25% quality improvement is material |

These are engineering starting points, not research-calibrated constants. They
must be constructor parameters, converted to frames using the configured
processed `fps` where applicable. Tests pass explicit values and assert
behavior, never the defaults themselves.

If live/replay evidence shows RGB difference is noisy under reflections, tune
the threshold before adding a more complex perceptual model. Do not add another
neural network in this change.

## Required behavior

### B-UR1 — Rejection becomes settled-but-observed

After an initial or repeated decision rejects a raw track:

- emit its offset Unknown ID;
- set `resolving=False`;
- store/replace its decision fingerprint and candidate target set;
- continue examining later usable crops cheaply;
- do not call the matcher again without a re-arm trigger.

### B-UR2 — Meaningful crop change re-arms

Given at least one candidate target and an expired cooldown, a materially
changed usable crop moves the raw ID from Settled Unknown to a fresh Pending
window.

- Seed the Pending buffer with the triggering crop.
- Emit the same offset ID while Pending.
- Set `resolving=True` until the decision completes.
- Collect/score using only fresh-window crops.

The existing `evidence_frames` or `evidence_window_s` closure rule applies. Do
not introduce a second recovery-specific evidence-window implementation.

### B-UR3 — Newly available target re-arms

Given an expired cooldown and a usable current crop, expansion of the candidate
target set re-arms the settled Unknown even if its visual fingerprint is
unchanged.

This is required because a new raw track can appear before the original raw
track crosses Active → Missing. The retry occurs at most once for each target-set
expansion because the new set is saved at the next decision.

### B-UR4 — Recovery uses the normal safe decision path

On a recovery attempt:

- score against `_comparison_galleries()` (complete roster);
- apply `matcher.accept()` unchanged;
- link only to `_eligible_missing_ids()`;
- defer an accepted coasting identity using the current deferral behavior;
- retain current greedy collision handling for simultaneous claims;
- never force a link after `REJECT`, low score, low margin, or collision loss.

An accepted recovery calls the existing `_link()` and therefore transfers the
fresh Pending crops, publishes the original session ID, and removes all Unknown
state for that raw ID.

### B-UR5 — Static Unknown does no repeated matcher work

A settled Unknown whose crop stays below all material-change thresholds and
whose candidate target set does not expand remains settled indefinitely:

- offset ID remains stable;
- `resolving=False` remains stable;
- matcher call count does not increase.

Small crop noise, confidence jitter, and sub-threshold mask changes must not
cause a spinner/matcher loop.

### B-UR6 — Rejection advances the baseline

If a recovery attempt rejects, the new evidence becomes the next decision
baseline. The same changed crop must not immediately trigger another retry.
Only a later material change or a later target-set expansion may re-arm it,
after the cooldown.

### B-UR7 — No-evidence Unknown can recover

If a decision settled Unknown because it had no usable crops, store a `None`
fingerprint. When a usable crop later appears and a candidate target exists,
that is sufficient evidence change to start one fresh Pending window after the
cooldown.

### B-UR8 — Unknown state is aged out

Track absence must age settled Unknown state using the existing death threshold:

- reset `absence_streak` whenever the raw ID is present;
- increment it while absent;
- delete the settled state once absence exceeds `_death_threshold_frames`.

If that raw ID somehow appears again later, it is treated as a new Pending
track. This also prevents session-long growth of rejected-track state.

### B-UR9 — Reset remains complete

`reset()` clears all settled Unknown/recovery state, Pending state, fingerprints,
cooldowns, and candidate snapshots. Persistent specimen galleries remain
resident as today.

### B-UR10 — Observability

Add one structured debug/info event when an Unknown is re-armed. It must expose:

- raw ID;
- trigger: `appearance`, `mask`, `quality`, `target_expanded`, or
  `no_previous_evidence`;
- measured appearance delta, mask IoU, and quality gain when available;
- old and new candidate target IDs;
- frames since the previous decision.

Do not log crops, embeddings, or image data. Existing batch-decision outcome
logging remains authoritative for whether the recheck linked or rejected.

## Processing order

Preserve the current per-frame ordering with one addition:

1. Advance frame count.
2. Age Active identities and Pending tracks.
3. **Age settled Unknown tracks.**
4. Process current rows:
   - mapped roster raw ID → existing Active path;
   - Pending raw ID → existing Pending path;
   - settled Unknown raw ID → observe crop and possibly re-arm;
   - unseen raw ID → initial Pending path.
5. Collect closed Pending windows.
6. Run the existing batch decision once for all initial and recovery windows
   that close on this tick.
7. Emit row-aligned IDs and `resolving` flags from the final state.

All recovery attempts that close on the same tick participate in the same
existing batch/collision decision. Do not run a separate matcher loop for each
state type.

## Safety invariants

The implementation is unacceptable if any of these change:

1. A raw ID cannot map to more than one session identity.
2. A session identity cannot have more than one active raw ID.
3. A fully present Active identity is never an eligible link target.
4. Matcher rejection is never overridden by crop change or target expansion.
5. Every non-mapped track emits in the offset ID range.
6. The roster remains frozen after enrolment.
7. Unknown tracks never enter Usage, Completeness, or the report.
8. Comparison uses the complete frozen roster; eligibility is checked after
   acceptance.
9. Rejected-window crops are not mixed into the fresh retry window.
10. An unchanged settled Unknown performs zero repeated matcher calls.

## TDD implementation order

Write the tests first, in this order. Use the existing `FakeMatcher`, synthetic
frames, explicit thresholds, and derived frame counts in
`model/tests/pipelines/test_session_linker.py`.

1. **Primary regression:** an occluded window rejects; unchanged occluded crops
   do not retry; a materially different unobstructed crop opens a fresh window;
   the matcher then accepts and the same raw ID emits the original session ID.
2. **Static foreign object:** after rejection, many unchanged frames cause no
   additional `score()` calls and stay `resolving=False`.
3. **Fresh evidence only:** the recovery matcher's crop markers contain the new
   unobstructed window and none from the original rejected window.
4. **Minor jitter:** below-threshold appearance, mask, and quality changes do not
   re-arm.
5. **Rejected recovery:** a changed crop triggers exactly one retry; rejection
   updates the baseline; repeated identical crops do not trigger again.
6. **Target expansion:** an Unknown first decided with no Missing/deferable
   identity retries once when the original identity becomes Missing, even if
   the crop is unchanged.
7. **Eligibility safety:** a matcher result naming a fully present Active
   identity never links it.
8. **No previous evidence:** a no-crop rejection can re-arm after the first
   usable crop appears.
9. **Cooldown:** repeated above-threshold changes inside the configured cooldown
   do not create additional decisions.
10. **Age-out:** settled Unknown state is discarded after the existing death
    threshold and a later occurrence is treated as new Pending.
11. **Simultaneous recovery collision:** two rechecks claiming the same Missing
    identity preserve the existing higher-score-wins rule; the loser settles
    Unknown with its new baseline.
12. **Reset:** no recovery fingerprint/cooldown/candidate state survives.
13. **Seam:** every transition keeps all detection fields row-aligned and emits
    the correct `resolving` value.

Delete or rewrite `test_b5_5_reject_stays_unknown_permanently`; permanence is no
longer the contract. Preserve all other existing linker tests unless the old
terminal assumption is the direct subject of the assertion.

Run:

```bash
cd model
uv run pytest tests/pipelines/test_session_linker.py -q
uv run pytest -q
```

## Files in scope

Required:

- `model/src/orc_model/pipelines/session_linker.py`
- `model/tests/pipelines/test_session_linker.py`
- `model/docs/linker-design.md`
- `model/docs/tracker-interface.md`

Optional only if a pure helper materially improves readability:

- one small model-side crop-fingerprint/change-policy module and its focused
  tests.

No changes are required in `app/`: roster membership, offset IDs, and the
existing resolving flag already express the complete user-visible behavior.

## Documentation changes required with the implementation

Update the canonical docs in the same change:

- `linker-design.md` §6.5: replace “settled decisions are never revisited” and
  “recovery is lift-and-replace” with the event-driven recovery rule.
- `linker-design.md` §7: Settled Unknown is stable until new evidence or target
  eligibility appears, rather than permanent for the raw ID.
- `tracker-interface.md` → “The resolving flag splits the Unknown range”:
  document that settled Unknown can return to Pending and therefore show the
  spinner again before either linking or settling Unknown again.
- Update module/test comments that call `_settled_unknown` permanent.

Do not silently edit historical T05/T08/T10 build records; they describe what
was shipped at that time. Canonical current behavior lives in the model docs
above.

## Validation beyond unit tests

### Required replay/live scenario

Use the KU Leuven recording if available; otherwise reproduce with the same
camera/table setup:

1. Start with the full enrolled roster.
2. Cover one instrument with a hand long enough to provoke a new raw track and
   an initial Unknown decision.
3. Keep the instrument in place and remove only the hand.
4. Confirm a structured `unknown re-armed` event occurs because the crop/mask/
   quality changed.
5. Confirm the next batch decision links the raw track to the original session
   ID.
6. Confirm recovery completes within one normal fresh evidence window (target:
   ≤1.0 s after unobstructed evidence begins at the configured processed fps).
7. Place a static foreign object while an identity is Missing and verify it
   settles Unknown without repeated matcher calls.

Record the three change metrics for both the hand-removal case and the static
foreign object. Adjust the initial thresholds only from this evidence, then keep
them fixed for the acceptance replay.

### Performance

The cheap fingerprint/change comparison runs per present settled Unknown and
must be negligible relative to detector inference. DINO/matcher timing should
remain event-driven. Report:

- number of settled-Unknown observation frames;
- number of re-arm events;
- number of additional matcher decisions;
- maximum matcher work added to any frame.

A static Unknown over a long replay must produce zero additional matcher calls
after its settling decision unless its target set expands.

## Non-goals and follow-ups

Not part of this implementation:

- lowering matcher acceptance thresholds;
- forced nearest-identity linking;
- a hand detector or MediaPipe dependency;
- per-frame DINO embeddings;
- changing Deep OC-SORT `max_age`, IoU, or embedding parameters;
- enabling Deep OC-SORT `mask_crop=True`;
- catalog identification;
- app/API/frontend changes.

Masked Deep OC-SORT crops are a sensible separate replay experiment because
they may prevent the raw-ID split in the first place. Keep that experiment out
of this recovery change so prevention and recovery can be measured independently.

## Definition of done

- The primary hand-occlusion regression fails before production changes and
  passes afterward.
- All required behavior tests pass.
- The complete model test suite passes.
- Static Unknowns do not repeatedly call the matcher.
- A changed same-raw-ID Unknown can recover a Missing session identity.
- Existing open-set and collision safety tests remain green.
- Canonical linker/interface docs describe recoverable Unknown behavior.
- Replay/live evidence demonstrates hand-removal recovery and records the
  chosen change thresholds.
