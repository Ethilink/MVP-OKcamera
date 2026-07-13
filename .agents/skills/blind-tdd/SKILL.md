---
name: blind-tdd
description: Build one focused Python module/feature with blind TDD + dual review. Orchestrator (Opus) writes a SPEC.md; a test-writer agent turns spec→tests; a separate Sonnet "lazy minimalist" coder turns spec→code WITHOUT seeing the tests; Codex + Opus review the code (consensus required, both must approve) BEFORE any test runs; then tests run with a blind fix loop and one test-revision escape hatch. Coder and reviewers never see test source. Use for a single new file or a tightly-scoped feature on the working tree (no git/worktree/PR machinery). For overnight multi-issue batches use night-coder instead.
user-invocable: true
args:
  - name: target
    description: The file or module to build (e.g. "src/frame_selection/weight_selector.py"). If omitted, infer from the conversation.
    required: false
---

Build ONE focused module/feature with role-separated agents and a hard blindness boundary: the coder and the reviewers never see the test source, so the code is written to the spec — not fitted to the tests.

This is the lightweight, local, single-feature sibling of `night-coder`. No GitHub issues, no worktrees, no pushing. Everything happens on the current working tree, driven by a SPEC.md you (the orchestrator) author from an already-converged design.

## Hard rules (the blindness boundary is the whole point)

- The **coder** and **fixer** agents must NOT see test source. Do not paste tests into their prompts. Tell them explicitly: "Do not read `tests/`, do not grep for `*_test*` / `test_*`." If they ask for tests, redirect to SPEC.md.
- The **reviewers** (Codex + Opus) must NOT see test source either. They review code against SPEC.md only.
- Only SPEC.md is shared across roles. Test-writer and coder agree because they share the spec, never each other's output.
- Each role is a FRESH sub-agent. Never reuse one agent across roles (context rot defeats the blindness).
- Coding model is **Sonnet**. Reviewers are **Codex** + **Opus**. Orchestrator (you) is Opus.
- Stay surgical: this skill builds the `target` and its test file only. No drive-by edits elsewhere.

## Skill loading per role (load these INTO each sub-agent)

Each sub-agent invokes its own skills as its first action. Put the instruction in the spawn prompt: "First invoke the `<skill>` skill(s), then…". Codex is prompt-composed by the orchestrator (it has no Codex Skill tool).

| Role | Model | Skills to load | Why |
|---|---|---|---|
| Orchestrator (SPEC.md) | Opus (you) | `solid` | Module boundaries, interfaces, minimal surface — design the contract before fan-out. |
| Test-writer | Sonnet | `tdd` + `dev-workflows-fullstack:testing-principles` | Red-green discipline + test quality/coverage. (`dev-workflows-fullstack:test-implement` if the patterns help.) |
| Coder | Sonnet | `solid` + explicit lazy-minimalist directive (below) | Best architecture, but only the minimum that satisfies the spec. |
| Codex reviewer | Codex (gpt-5.x) | composed via `codex:gpt-5-4-prompting`, run as `codex:codex-rescue` | Independent correctness / edge-case / bug lens. |
| Opus reviewer | Opus | `code-review` + `solid` | Review procedure + design-conformance rubric. |
| Final verify (optional) | Sonnet | `verify` | Actually run the feature and observe it behaves. |

**Task-specific overlay:** add the skill that matches what `target` touches. In THIS repo, if the code writes or modifies Gemini API calls, also load `gemini-api-dev` into the coder and both reviewers. The weight-selector target reuses existing `get_tray_roi`, so it does NOT need `gemini-api-dev` unless the coder changes a Gemini call. Add `security-review` to the Opus reviewer only if the target handles untrusted input / secrets / file paths from outside.

The lazy-minimalist directive (paste verbatim into the coder prompt):
> You are a lazy senior engineer: write the MINIMUM code that satisfies SPEC.md and nothing more. No speculative abstractions, no config knobs that weren't asked for, no error handling for impossible cases, no helpers used once. Match the surrounding code's style. If you'd write 200 lines where 50 do the job, write 50.

## Phase 0 — SPEC.md (orchestrator, serial gate)

Distill the converged design into `SPEC.md` next to the target (or under `docs/`). It is the single source of truth for everyone. It MUST contain:
- **Goal** — one paragraph, what the module does and does not do.
- **Public interface** — exact function/class signatures, argument types, return types.
- **Behavior** — input→output contract, ordering, edge cases, failure modes (what to do, not how).
- **Inputs/Outputs** — data shapes/formats, file paths, units.
- **Dependencies** — what existing code it reuses (cite `module:symbol`), what it must NOT touch.
- **Out of scope** — explicit non-goals, so the lazy coder stays lazy.
- **Tunable parameters** — values that are tuned-by-observation, not asserted (so the test-writer doesn't hard-assert them).

Then generate a **trivial stub** of `target` from the Public interface: real signatures, bodies that return trivially-wrong dummies (`None`, empty containers) — NOT `raise NotImplementedError`. The stub exists so Phase 1's red-check fails at the **assertion level**: assertions must actually evaluate against wrong values and catch them. `ImportError`-red or `NotImplementedError`-red only proves the module is absent, not that the tests can detect a wrong implementation.

If the design isn't actually converged, stop and converge it with the user first. Do not invent a spec.

**In THIS repo (MVP-OKcamera):** the dashboard tasks in `data-collection/docs/tasks/T##-*.md` are pre-authored specs — frozen Public interface + numbered ACs. Use the task file AS the SPEC.md (point every role at it) instead of authoring a duplicate; two sources of truth will drift.

## Phase 1 — Tests (test-writer agent, sees SPEC.md only)

Spawn a fresh Sonnet agent: "First invoke `tdd` and `dev-workflows-fullstack:testing-principles`. Read SPEC.md. Write tests that fail until the module is implemented. Test the behavior contract, not internal field names or exact tunable values. Do NOT write any implementation code. Name tests after the spec behavior/AC they cover."
- Run the tests **against the Phase 0 stub**. **Verify assertion-level red**: every test must fail on an assertion (or a spec-mandated exception check), not on `ImportError`/collection errors. Import/collection failures mean the stub or test wiring is wrong — fix that first. A test that PASSES against the dummy stub is broken — re-prompt once for a stricter test; if still broken, stop and surface to the user.
- The test file is now the hidden gate. Note its path so later prompts can exclude it.

## Phase 1.5 — Test review (blind the other way, cheap)

The tests are the gate that decides "done" — they get scrutiny too. Spawn a fresh Opus agent that sees SPEC.md + the test source and NOT the implementation (only the dummy stub exists anyway): "First invoke `code-review`. Check: (1) every Behavior item / AC in SPEC.md has ≥1 covering test — output the behavior→test mapping; (2) no assertion contradicts the spec; (3) no test asserts internals, exact tunable values, or timing. ✅ / 💡 / ❌ per concern."
- ❌ findings go back to the test-writer for one revision, then re-check.
- Keep the behavior→test mapping — Phase 5 reports it.

## Phase 2 — Code (coder agent, Sonnet, BLIND)

Spawn a fresh Sonnet agent: "First invoke `solid`. [lazy-minimalist directive]. Read SPEC.md and implement `<target>` (replace the Phase 0 stub). You do NOT have access to the test source and must not read `tests/` or grep for test files — tests are the validation gate, not a spec. Build exactly what SPEC.md describes."
- Do not pass test source. Do not let the agent read it.
- **Model knob:** default Sonnet. For concurrency-/timing-heavy targets (threads, locks, async generators, streams), use Opus for the coder — lazy-minimalist Sonnet writes plausibly-racy code whose bugs surface late and flaky.

## Phase 2.5 — Smoke (free, no blindness cost)

Before spending two reviewers: `python -c "import <module>"`, then ruff/pyright if the repo has them configured. Failures go straight back to a blind fixer. Never burn a consensus round on code that doesn't import.

## Phase 3 — Review (Codex + Opus in parallel, CONSENSUS gate, BEFORE tests run)

Both reviewers see `target` + SPEC.md, NOT tests. Run them in one batch (parallel):
- **Codex** (`codex:codex-rescue`, prompt composed via `codex:gpt-5-4-prompting`): "Review this code against SPEC.md for correctness, edge cases, bugs, and spec-conformance. List each concern as ✅ / 💡 / ❌."
- **Opus** (fresh Opus agent): "First invoke `code-review` and `solid`. Review this code against SPEC.md for architecture conformance, simplicity, dead code, and correctness. ✅ / 💡 / ❌ per concern."

**❌ is defined narrowly:** would produce behavior that violates SPEC.md, or crashes on in-scope input. Style, naming, hypothetical-scale concerns, "I'd structure this differently" → 💡. Put this definition in both reviewer prompts, or nit-❌s will burn the round cap.

**Consensus required:** proceed only when BOTH reviewers return zero ❌ blockers. 💡 are dropped as noise.
- If either has ❌: spawn a fresh blind Sonnet fixer ("address these blockers; you still cannot read tests"), then re-review (deltas only, both reviewers).
- Safety cap: 3 review→fix rounds. If consensus isn't reached by round 3, STOP and surface the unresolved blockers + both viewpoints to the user — do not auto-proceed to tests without consensus.

## Phase 4 — Run tests (blind fix loop + one escape hatch)

Now run the hidden tests — with `pytest --tb=line -q`. Full tracebacks leak test source: pytest's assertion introspection prints the exact assertion line (`assert result == [1.5, 3.0]`), which hands the fixer the expected values to hard-code against. `--tb=line` keeps the failure signal (test name, one-line reason) without the source.
- **Green:** go to Phase 5.
- **Red:** spawn a fresh blind Sonnet fixer with the `--tb=line` OUTPUT only, NOT the test source: "Tests fail with this output: <…>. You have SPEC.md and the current code, not the tests. Patch the code." Re-run. Max **5** iterations.
- If a fix changed code materially, re-run the Phase 3 consensus review (deltas only) before declaring done.
- **Escape hatch (one shot):** after 5 red iterations, spawn the test-writer once with SPEC.md + the failure output — **NOT the implementation**. Showing the reviser the code invites refitting tests to the implementation, the exact failure this skill exists to prevent. Ask: "Are these tests over-specified relative to SPEC.md (asserting internals, exact tunable values, timing)?" If over-specified → revise tests, re-run the Phase 1 stub red-check + Phase 1.5 review on the revision, freeze them, reset the fix counter to 3, loop. If the tests are faithful to the spec → the code is wrong by elimination: stop and surface to the user. No second test revision.

## Phase 5 — Done

Report to the user: what was built, the SPEC.md path, review rounds + consensus outcome, test result, the Phase 1.5 **behavior→test mapping** (re-verified if tests were revised — any spec behavior with no covering test is a hole the lazy coder was allowed to skip; call it out), and any flagged-but-unresolved items. Do not commit or push unless the user asks (then use `commit-commands:commit`).

## Anti-patterns (do not do)

- Letting the coder or reviewers see test source. Breaks the entire design.
- Calling "red" on `ImportError`/`NotImplementedError` — red must be assertion-level against the Phase 0 dummy stub, or the tests prove nothing.
- Passing full pytest tracebacks to the fixer (assertion introspection leaks test source) — `--tb=line -q` only.
- Showing the escape-hatch test-reviser the implementation.
- Reusing one agent across roles.
- Proceeding past review without true consensus (both at zero ❌).
- Looping reviews/fixes forever — bounded rounds, then escalate to the user.
- Drive-by edits outside `target` + its test file.
- Inventing a spec when the design isn't converged. Converge first.
- Overloading an agent with five skills — load only the rows above for that role, plus the task-specific overlay if it applies.
