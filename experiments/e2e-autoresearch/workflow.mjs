export const meta = {
  name: 'orc-e2e-autoresearch',
  description: 'Overnight end-to-end session-linker accuracy search on the two demo takes; sandbox-only, recommends a config (no shipping edits).',
  phases: [
    { title: 'Foundation', detail: 'confirm baseline reproduces demo-validation.md (hard gate)' },
    { title: 'FlipAug', detail: 'flip/rotation gallery augmentation vs the 368s flip case' },
    { title: 'DINOv3', detail: 'DINOv3-B/S backbone bake-off (crop frozen eval + e2e confirm)' },
    { title: 'MatcherSearch', detail: 'novel matcher families, champion/challenger loop' },
    { title: 'Synthesize', detail: 'FINDINGS + RECOMMENDATION' },
  ],
}

// ---- config -----------------------------------------------------------------
const SB = 'experiments/e2e-autoresearch'          // this sandbox (my scorer, caches, gt)
const MA = 'experiments/matcher-autoresearch'      // crop-level frozen eval + champion
const PY = 'model/.venv/bin/python'                // has torch+transformers+onnxruntime
const MATCHER_ROUNDS = budget.total ? 12 : 6       // novel-family loop cap
const CHALLENGERS_PER_ROUND = 2                    // LOW — everything contends on one MPS device
const RESERVE = 60_000

const SANDBOX_RULE = `HARD RULES (violating any = your result is void):
- Edit ONLY under ${SB}/ (and ${MA}/runs|method for crop-eval variants). NEVER edit model/src/**, matching/**, or ANY shipping code.
- End-to-end tests inject a variant by monkeypatching orc_model.pipelines.matching.ChampionMethod BEFORE calling load_tracker() (local import at tracking.py:266 picks it up). No shipping edit.
- Offline: prefix python with HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1. Use ${PY}.
- NEVER lower tau/margin/cos_tau to pass the 302s/368s hard cases. A wrong link is worse than a safe Unknown.
- Read ${SB}/BRIEF.md first. It is the source of truth.`

const E2E_HOWTO = `END-TO-END SCORING (the demo-safety gate):
- Replay from cache (fast, ~80-125s/take, no detector):
  ${PY} model/scripts/replay_session.py --video matching/data/testing/15-07-26-001/videos/15-07-26-001.mp4 --out <run>/take-a --cache ${SB}/caches/take-a.dets.npz --from-cache [extra flags]
  (take B: 15-07-26-002, cache take-b.dets.npz)
- To test a VARIANT matcher/embedder/gallery end-to-end, write a small driver that monkeypatches
  orc_model.pipelines.matching.ChampionMethod (or its embedder/gallery build) then calls the same
  replay logic. A gallery-only change can instead use --instruments-dir <augmented_dir>.
- Score: python3 ${SB}/score_e2e.py --trace <run>/take-a.json --gt ${SB}/gt_events.json --take A  (and --take B)
- DEMO-SAFE requires, on BOTH takes: roster {1..8}, foreign_reject_rate==1.0, zero wrong_links,
  zero lost/changed correct links (link_retention==1.0), binding_matches_baseline==true.
- A hardcase_change (raw 72 or raw 88 going from unknown -> a link) is a candidate WIN but must be
  hand-flagged in your notes (verify the link is to the physically-correct instrument) and must not
  come with ANY regression or a lowered threshold.`

const CROP_HOWTO = `CROP-LEVEL FROZEN EVAL (the trustworthy graded optimizer):
- Harness: ${MA}/frozen/ (loader, cv_split, eval, holdout). Champion: ${MA}/champion/ (SRC).
- Implement your variant behind the frozen build_gallery/score/accept interface (${MA}/frozen/interface.py)
  as a method module under ${MA}/runs/<exp>/. NEVER edit anything in ${MA}/frozen/.
- Run the frozen 5-seed CV (run_cv) + a hyperparameter re-sweep where relevant (a backbone swap MUST
  re-sweep alpha/tau/size_alpha/cos_tau). Report reidTop1 (mean), seedBand (std), foreignReject (mean),
  twinErrors. Reject-floor to beat: F >= 0.9733. Champion to beat: reidTop1 0.9333.`

const METRIC = {
  type: 'object',
  required: ['lever', 'summary', 'cropReidTop1', 'cropSeedBand', 'cropForeignReject',
             'e2eDemoSafeA', 'e2eDemoSafeB', 'hardcaseImproved', 'artifactPath', 'cheatRisk'],
  properties: {
    lever: { type: 'string', description: 'flip-aug | dinov3-vitb16 | dinov3-vits16 | family:<name>' },
    summary: { type: 'string' },
    cropReidTop1: { type: 'number', description: 'crop-eval CV re-ID top-1 mean, or -1 if not run' },
    cropSeedBand: { type: 'number' },
    cropForeignReject: { type: 'number', description: 'crop-eval CV foreign-reject mean, or -1' },
    e2eDemoSafeA: { type: 'boolean' },
    e2eDemoSafeB: { type: 'boolean' },
    e2eNotes: { type: 'string', description: 'link_retention, foreign_reject, any regressions/wrong_links per take' },
    hardcaseImproved: { type: 'string', description: 'none | raw72 | raw88 | raw72+raw88 — and to which instrument, verified' },
    artifactPath: { type: 'string' },
    cheatRisk: { type: 'boolean', description: 'true if you touched anything frozen/shipping or lowered a threshold' },
  },
}

const REVIEW = { type: 'object', required: ['clean', 'reason'],
  properties: { clean: { type: 'boolean' }, reason: { type: 'string' } } }

// challenger clears the bar only if it is not a cheat, is demo-safe on BOTH takes,
// and either raises crop re-ID beyond noise or legitimately fixes a hard case.
const beats = (c, champReid) => c && !c.cheatRisk && c.e2eDemoSafeA && c.e2eDemoSafeB &&
  ((c.cropReidTop1 > champReid + (c.cropSeedBand || 0.01)) || (c.hardcaseImproved && c.hardcaseImproved !== 'none'))

// ---- Phase 1: Foundation (hard gate) ---------------------------------------
phase('Foundation')
const found = await agent(
  `${SANDBOX_RULE}\n\nConfirm the end-to-end foundation reproduces the documented baseline, then record it.\n` +
  `1. Replay BOTH takes from cache with the SHIPPED config (no extra flags) into ${SB}/runs/baseline/.\n` +
  `2. Score each with ${SB}/score_e2e.py against ${SB}/gt_events.json.\n` +
  `3. Assert BOTH takes are demo_safe:true, foreign_reject_rate==1.0, link_retention==1.0, roster {1..8}.\n` +
  `4. Cross-check against model/docs/demo-validation.md "Latest results" (46 batches Take B, 6 foreign rejects, 302s {71:linked:1,72:unknown}, 368s {88:unknown}).\n` +
  `5. Create ${SB}/LEADERBOARD.md with a header + the baseline row (SRC / DINOv2-B, the shipped champion).\n` +
  `${E2E_HOWTO}\n\nReturn the METRIC for the baseline (lever="baseline"). Set e2eDemoSafeA/B from your scoring. If either take is NOT demo_safe, still return the metric but put the discrepancy in summary — this is a hard gate.`,
  { phase: 'Foundation', schema: METRIC, label: 'foundation-gate' }
)
if (!found || !found.e2eDemoSafeA || !found.e2eDemoSafeB) {
  log('FOUNDATION GATE FAILED — baseline did not reproduce demo-safe on both takes. Aborting before spending on challengers.')
  return { aborted: 'foundation', found }
}
log('Foundation OK — baseline reproduces demo-safe on both takes.')
const CHAMP_REID = 0.9333  // SRC crop-eval champion (from matcher-autoresearch LEADERBOARD)

// ---- Phase 2: Flip/rotation gallery augmentation ----------------------------
phase('FlipAug')
const flipVariants = [
  { lever: 'flip-aug', how: 'augment every gallery (persistent + Start) by ALSO embedding a horizontally-flipped copy of each masked crop, concatenated as extra dictionary atoms' },
  { lever: 'flip+rot180-aug', how: 'augment galleries with BOTH a horizontal flip and a 180-degree rotation of each masked crop' },
]
const flipResults = await parallel(flipVariants.map(v => () => agent(
  `${SANDBOX_RULE}\n\nLEVER: ${v.lever}. Hypothesis: gallery augmentation makes the flipped-instrument return (Take B raw 88 @368s, and any flip/rotation return) re-identify correctly WITHOUT lowering thresholds or breaking foreign-reject.\n` +
  `Implement: ${v.how}. Do it as a variant matcher/gallery-build under ${SB}/runs/${v.lever}/ (subclass or wrap ChampionMethod's build_gallery; keep score/accept identical).\n` +
  `Evaluate on BOTH surfaces:\n(a) crop frozen eval (does augmentation hold re-ID and foreign-reject?);\n(b) end-to-end on BOTH takes via injection, and specifically report raw 88's outcome (and raw 72's).\n` +
  `${CROP_HOWTO}\n${E2E_HOWTO}\n\nReturn METRIC. hardcaseImproved must state which raw id improved and to which instrument, verified physically (view the crop frames if needed).`,
  { phase: 'FlipAug', schema: METRIC, label: v.lever }
)))
for (const r of flipResults.filter(Boolean)) log(`FlipAug ${r.lever}: demoSafe A/B=${r.e2eDemoSafeA}/${r.e2eDemoSafeB} hardcase=${r.hardcaseImproved} reid=${r.cropReidTop1}`)

// ---- Phase 3: DINOv3 backbone bake-off --------------------------------------
phase('DINOv3')
const dinoVariants = [
  { lever: 'dinov3-vitb16', model: 'facebook/dinov3-vitb16-pretrain-lvd1689m' },
  { lever: 'dinov3-vits16', model: 'facebook/dinov3-vits16-pretrain-lvd1689m' },
]
const dinoResults = await parallel(dinoVariants.map(v => () => agent(
  `${SANDBOX_RULE}\n\nLEVER: ${v.lever}. Swap the matcher embedder to ${v.model} (cached offline; 768-dim CLS for vitb16). ChampionMethod(model_id="${v.model}").\n` +
  `CRITICAL: the SRC alpha/tau/size_alpha/cos_tau were calibrated for DINOv2 geometry. You MUST re-sweep them on the crop frozen eval (CV grid) — a naive swap is misleading. Report the best re-swept config's reidTop1/seedBand/foreignReject.\n` +
  `Work under ${MA}/runs/${v.lever}/ (crop eval) and ${SB}/runs/${v.lever}/ (e2e). If the re-swept variant beats SRC (reidTop1 > ${CHAMP_REID}+seedBand, foreignReject >= 0.9733) OR is close, run the end-to-end confirmation on BOTH takes with the re-swept params and report demo-safety.\n` +
  `${CROP_HOWTO}\n${E2E_HOWTO}\n\nReturn METRIC.`,
  { phase: 'DINOv3', schema: METRIC, label: v.lever }
)))
for (const r of dinoResults.filter(Boolean)) log(`DINOv3 ${r.lever}: reid=${r.cropReidTop1}±${r.cropSeedBand} reject=${r.cropForeignReject} demoSafe A/B=${r.e2eDemoSafeA}/${r.e2eDemoSafeB}`)

// ---- Phase 4: novel-family matcher search (champion/challenger) --------------
phase('MatcherSearch')
let round = 0
const keepGoing = () => round < MATCHER_ROUNDS && (!budget.total || budget.remaining() > RESERVE)
while (keepGoing()) {
  round++
  const challengers = await parallel(Array.from({ length: CHALLENGERS_PER_ROUND }, (_, i) => () => agent(
    `${SANDBOX_RULE}\n\nRound ${round}, challenger ${i}. Invent ONE genuinely NEW matcher method family (NOT a knob-tweak, NOT a family already in ${MA}/TRIED.md — SRC survived ~25 families across rounds 0-9). ${i === 0 ? 'Consider first reconstructing + leak-checking the unvalidated round-9 hyperbolic candidate (0.9833 CV) whose runs/ artifacts were lost.' : 'Rotate to a family not recently tried.'}\n` +
    `Implement behind the frozen build_gallery/score/accept interface under ${MA}/runs/r${round}-c${i}/. Grade on the crop frozen eval. Append your idea to ${MA}/TRIED.md. If it beats the champion on crop eval (reidTop1 > ${CHAMP_REID}+seedBand, reject >= 0.9733), ALSO run the end-to-end demo-safety gate on both takes.\n` +
    `NO backbone fine-tuning on the 8 specimens (a generic low-capacity learned combiner with held-out CV is fine).\n${CROP_HOWTO}\n${E2E_HOWTO}\n\nReturn METRIC. cheatRisk=true if you touched anything frozen/shipping.`,
    { phase: 'MatcherSearch', schema: METRIC, agentType: i === 0 ? 'codex:codex-rescue' : undefined, label: `r${round}-c${i}` }
  )))
  const live = challengers.filter(Boolean).filter(c => !(c.cropReidTop1 === 0 && c.cropForeignReject === 0))
  const winner = live.filter(c => beats(c, CHAMP_REID)).sort((a, b) => b.cropReidTop1 - a.cropReidTop1)[0]
  if (!winner) { log(`Round ${round}: no challenger beat the champion by margin AND demo-safe.`); continue }
  const review = await agent(
    `${SANDBOX_RULE}\n\nLeak-check the promoted challenger at ${winner.artifactPath} against ${MA}/frozen/. Look for: edited frozen eval/split/loader, query crops leaked into gallery, holdout peeking, hardcoded labels, or a lowered threshold used to pass a hard case. Append a dated "e2e round ${round} leak-check verdict" to ${MA}/TRIED.md. Return REVIEW {clean, reason}.`,
    { phase: 'MatcherSearch', schema: REVIEW, agentType: 'codex:codex-rescue', label: `leakcheck-r${round}` }
  )
  const suspicious = winner.cropReidTop1 - CHAMP_REID > 0.1
  if (review ? review.clean : !suspicious) {
    log(`Round ${round}: PROMOTED ${winner.lever} (reid ${winner.cropReidTop1}, demo-safe). Recording to LEADERBOARD.`)
    await agent(`${SANDBOX_RULE}\n\nRecord the new best matcher ${winner.lever} (${winner.artifactPath}) into ${SB}/LEADERBOARD.md with full crop-eval + end-to-end numbers, marking the previous best superseded.`,
      { phase: 'MatcherSearch', label: `record-r${round}` })
  } else {
    log(`Round ${round}: winner ${winner.lever} NOT promoted (leak-check: ${review ? review.reason : 'unavailable + suspicious jump'}).`)
  }
}

// ---- Phase 5: Synthesize ----------------------------------------------------
phase('Synthesize')
const synth = await agent(
  `${SANDBOX_RULE}\n\nWrite the deliverables from everything in ${SB}/LEADERBOARD.md, ${SB}/runs/, and ${MA}/TRIED.md:\n` +
  `1. ${SB}/FINDINGS.md — what helped / what didn't across flip-aug, DINOv3, and novel families; the flip-case (raw88) and twin-swap (raw72) verdicts; DINOv3 verdict; honest caveats (tiny data → directional; the two takes are the decision-relevant signal).\n` +
  `2. ${SB}/RECOMMENDATION.md — the SINGLE config to run at the demo tomorrow and WHY, expressed as an exact change to the shipped pipeline (a described diff of matching/champion.py params or load_tracker args, or "keep shipped config" if nothing beat it demo-safe). If a change is recommended, include the end-to-end demo-safety evidence on BOTH takes.\n` +
  `Do NOT edit shipping code. This is a recommendation for Bram to review and apply.`,
  { phase: 'Synthesize', label: 'synthesize' }
)
return { foundation: found, flip: flipResults.filter(Boolean), dinov3: dinoResults.filter(Boolean), matcherRounds: round, synth }
