export const meta = {
  name: 'matcher-autoresearch',
  description: 'Overnight champion/challenger loop that invents session-matcher techniques and keeps only held-out wins',
  phases: [
    { title: 'Baseline', detail: 'set up frozen/mutable harness + champion baseline' },
    { title: 'Explore', detail: 'fan out challengers, keep-if-better-by-margin, budget-paced' },
    { title: 'Synthesize', detail: 'FINDINGS.md + clean champion module' },
  ],
}

// --- config -----------------------------------------------------------------
// Contained sandbox: reads shared data READ-ONLY, writes ONLY under DIR.
const DIR = (args && args.dir) || 'experiments/matcher-autoresearch'
const PROGRAM = `${DIR}/program.md`
const CHALLENGERS_PER_ROUND = 3            // slow-but-steady: protects the tiny-data signal (more parallelism just adds noise-chasing)
const MAX_ROUNDS = 30                      // hard backstop so it can never run to the agent-count cap (~30*4 agents << 1000)
const RESERVE = 60_000                     // if a budget target IS set, it's just an extra ceiling (stop leaving room to synthesize)
// No dry-round auto-stop: the operator runs this until they say done (manual stop, or the
// MAX_ROUNDS/budget backstops below). dryStreak is still tracked and logged for visibility.

const METRIC = {
  type: 'object',
  additionalProperties: false,
  required: ['technique', 'reidTop1', 'foreignReject', 'trueAccept', 'twinErrors', 'seedBand', 'artifactPath', 'cheatRisk', 'notes'],
  properties: {
    technique:    { type: 'string', description: 'one-line name of the method tried' },
    reidTop1:     { type: 'number', description: 'held-out re-ID top-1 (0..1), CV-averaged' },
    foreignReject:{ type: 'number', description: 'foreign-object reject rate at operating threshold (0..1)' },
    trueAccept:   { type: 'number', description: 'genuine-return accept rate (0..1)' },
    twinErrors:   { type: 'number', description: 'instrument1<->2 confusion count' },
    seedBand:     { type: 'number', description: '±reidTop1 std across CV seeds (the noise band)' },
    artifactPath: { type: 'string', description: 'path under DIR to the method variant + its run log' },
    cheatRisk:    { type: 'boolean', description: 'true if it had to touch anything near the frozen eval/split/loader' },
    notes:        { type: 'string', description: 'what worked / failed / next idea' },
  },
}
const REVIEW = {
  type: 'object', additionalProperties: false,
  required: ['clean', 'reason'],
  properties: { clean: { type: 'boolean' }, reason: { type: 'string' } },
}

const beats = (c, champ) =>
  c && !c.cheatRisk &&
  c.foreignReject >= champ.foreignReject &&            // reject floor may only rise
  c.reidTop1 > champ.reidTop1 + (c.seedBand || 0.01)   // beat by more than noise

// --- 1. baseline champion ---------------------------------------------------
phase('Baseline')
let champion = await agent(
  `Read ${PROGRAM} in full, then set up the contained harness in ${DIR}: split the copied stage-1 code in ${DIR}/harness into a FROZEN part (data loader, eval, CV split, locked holdout) and a MUTABLE ${DIR}/method/ implementing build_gallery/score/accept. Implement the CHAMPION BASELINE described in program.md §5 (DINOv2-B, masked crops, moderate mask-size fusion, top-3-mean, multi-frame voting). Run the FROZEN eval and report the guarded held-out metrics. Write ${DIR}/champion/, start ${DIR}/LEADERBOARD.md and ${DIR}/TRIED.md. Data is READ-ONLY at model/data and matching/data/testing; write ONLY under ${DIR}.`,
  { schema: METRIC, phase: 'Baseline', label: 'baseline' },
)
if (!champion) { log('baseline failed — aborting'); return { error: 'baseline failed' } }
log(`baseline: re-ID ${champion.reidTop1}, reject ${champion.foreignReject}`)

// --- 2. explore (budget-paced champion/challenger) --------------------------
phase('Explore')
let round = 0
let dryStreak = 0
// Runs until the operator stops it (or the round-cap / budget backstop below).
const keepGoing = () =>
  round < MAX_ROUNDS &&
  (!budget.total || budget.remaining() > RESERVE)
while (keepGoing()) {
  round++
  const challengers = (await parallel(
    Array.from({ length: CHALLENGERS_PER_ROUND }, (_, i) => () =>
      agent(
        `Round ${round}, challenger ${i}. Read ${PROGRAM} (esp. §6) and ${DIR}/TRIED.md. Invent ONE genuinely NEW METHOD FAMILY — a different representation or matching paradigm, NOT a knob-tweak of the champion and NOT a family already in TRIED.md (classical keypoint/geometric matching, colour/texture, set-to-set, one-class/OOD scoring, multi-cue fusion, a different backbone, etc. — go beyond the menu). Beat the champion (held-out re-ID ${champion.reidTop1}, reject ${champion.foreignReject}). Implement it behind the SAME build_gallery/score/accept interface as an isolated variant under ${DIR}/runs/r${round}-c${i}/, run the FROZEN eval (never edit it), and report the guarded metrics — set the "technique" field to the family name. Do NOT fine-tune the backbone on the 8 demo specimens (generic low-capacity learned combiners with held-out CV are OK). Append your idea + family to ${DIR}/TRIED.md. If you had to touch anything frozen, set cheatRisk=true and explain.`,
        {
          schema: METRIC,
          phase: 'Explore',
          label: `r${round}-c${i}`,
          // challenger 0 uses Codex for idea diversity (program.md §7a). Intermittently returns an
          // unresolved placeholder stub (reidTop1=0, no artifact) instead of a real result on the
          // heavy invent+implement+eval task (seen round 1) rather than a clean error — but it has
          // also produced real, strong results (round 2's highest-reject challenger). Kept, and
          // failures are caught by the stub filter below rather than by disabling the role outright.
          agentType: i === 0 ? 'codex:codex-rescue' : undefined,
        },
      ),
    ),
  ))
    .filter(Boolean)
    // defensive: drop unresolved-placeholder stubs (e.g. a background codex run that didn't
    // finish in time) so they never get logged/reviewed as if they were real results
    .filter((c) => !(c.reidTop1 === 0 && c.foreignReject === 0))

  let promoted = false
  const winner = challengers.filter((c) => beats(c, champion)).sort((a, b) => b.reidTop1 - a.reidTop1)[0]
  if (winner) {
    // Codex leak-check before promotion; if the reviewer is unavailable, do NOT promote a big jump blind
    const review = await agent(
      `Leak-check a candidate matcher promotion. Inspect the diff/code at ${winner.artifactPath} against the FROZEN harness in ${DIR}/harness. Did it cheat: edit the eval/split/loader, leak a query crop into its own gallery, peek at the holdout, or hardcode labels? Answer clean=true only if it genuinely beat the champion by method, not by leakage. Whatever you conclude, append a dated "## Round ${round} leak-check verdict" entry to ${DIR}/TRIED.md recording the candidate (${winner.technique}, ${winner.artifactPath}), your clean/not-clean verdict, and your reasoning — so this verdict is visible in the written record, not just your own return value.`,
      { schema: REVIEW, phase: 'Explore', label: `review-r${round}`, agentType: 'codex:codex-rescue' },
    )
    const suspicious = winner.reidTop1 - champion.reidTop1 > 0.1
    if (review ? review.clean : !suspicious) {
      const supersededTechnique = champion.technique
      champion = winner
      promoted = true
      log(`round ${round}: NEW champion "${winner.technique}" — re-ID ${winner.reidTop1}, reject ${winner.foreignReject}`)
      // Promotion sync: without this, the in-memory `champion` above moves on (and every later
      // round's prompt correctly targets it) but the two human-facing "current state" artifacts
      // never do — LEADERBOARD.md still shows the old champion, champion/ still holds its old
      // code. Do this as its own step, right after the leak-check clears, every time.
      await agent(
        `The challenger at ${winner.artifactPath} (technique: "${winner.technique}", re-ID ${winner.reidTop1}, reject ${winner.foreignReject}, twin errors ${winner.twinErrors}, seedBand ${winner.seedBand}) just passed a Codex leak-check (see ${DIR}/TRIED.md's round ${round} leak-check verdict) and is the new champion, superseding "${supersededTechnique}". Two things, both required: (1) Replace the contents of ${DIR}/champion/ with this method as a clean, standalone module — port/clean up the code from ${winner.artifactPath} (which currently depends on ${DIR}/frozen for its eval driver), don't just copy the research variant verbatim; it must implement build_gallery/score/accept exactly per linker-design.md §6 with NO dependency on ${DIR}/frozen or ${DIR}/method, and champion/PARAMS.md must be rewritten with this method's hyperparameters, guarded metrics, and ablations. (2) Add a new dated row to ${DIR}/LEADERBOARD.md's ranking table for round ${round} marked **CHAMPION**, and edit the previous champion's row to mark it superseded (not delete it — it's the run's history).`,
        { phase: 'Explore', label: `promote-r${round}` },
      )
    } else {
      log(`round ${round}: winner "${winner.technique}" rejected by leak-check (${review ? review.reason : 'reviewer unavailable + suspicious jump'})`)
    }
  } else {
    log(`round ${round}: no challenger beat the champion by margin`)
  }
  dryStreak = promoted ? 0 : dryStreak + 1
  log(`round ${round}: dry streak ${dryStreak} (informational only, not a stop condition)${budget.total ? `, ~${Math.round(budget.remaining() / 1000)}k left` : ''}`)
}
log(`explore ended after ${round} rounds — ${round >= MAX_ROUNDS ? 'hit round cap' : 'budget ceiling'}`)

// --- 3. synthesize ----------------------------------------------------------
phase('Synthesize')
await agent(
  `Read ${DIR}/LEADERBOARD.md and ${DIR}/TRIED.md (including any "leak-check verdict" entries) and write ${DIR}/FINDINGS.md per program.md §9: the recommended values for model/docs/linker-design.md's open params — mask-size fusion weight, voting scheme + window, aggregation (nearest vs top-K), accept threshold, canonicalization yes/no — plus the instrument8 and instrument1<->2 verdicts, each with the held-out number that backs it. Confirm ${DIR}/champion/ is the clean winning module with the build_gallery/score/accept interface from linker-design.md §6. Mark every number DIRECTIONAL (tiny set; confirm at stage-2). IMPORTANT: writing ${DIR}/FINDINGS.md to disk is the explicitly-requested deliverable of this task per program.md §9 — this is a data file the research program requires, not a gratuitous docs/README file, so write it directly with your file-write tool rather than only returning the content as text.`,
  { phase: 'Synthesize', label: 'findings' },
)
return { champion, rounds: round }
