---
name: file-issues
description: Publish code-review or design-review findings to GitHub as structured issues for Ethilink/MVP-OKcamera — one issue per finding, validated against a required-field template (summary, location, why it matters, category, severity, optional suggested fix) and labeled from a fixed taxonomy. Use when the user asks to file, publish, open, or create GitHub issues from findings, a review, or an audit, or says things like "turn these into issues" or "file these as issues".
---

# File Issues

Publishes a batch of findings (from a review just discussed, from `/code-review`'s structured output, or any other flagged list of issues) to GitHub as individual issues, each conforming to the same field template as `.github/ISSUE_TEMPLATE/finding.yml` — so an issue looks the same whether a human filed it through the GitHub web form or this skill filed it via `gh`.

## Required fields per finding

Before a finding is eligible to publish it must have all of:

- **Summary** — one sentence
- **Location** — file:line or component
- **Why it matters / failure scenario** — concrete impact
- **Category** — one of the fixed taxonomy below
- **Severity** — one of: high, medium, low

`Suggested fix` is optional. A finding missing any required field is excluded from the batch — report why it was dropped, don't publish it incomplete, and don't guess a value on the user's behalf.

## Label taxonomy (fixed — never invent a new label for this workflow)

| Category | Label | Use for |
|---|---|---|
| Correctness/bug | `bug` | existing GitHub default |
| Docs gap | `documentation` | existing GitHub default |
| Structural/design concern | `architecture` | |
| Promotion debt / missing abstraction / deferred work | `tech-debt` | |
| Lint/CI/dev-process gap | `tooling` | |

Severity → `severity: high`, `severity: medium`, or `severity: low`.

If a finding doesn't clearly fit one of the five categories, ask the user rather than inventing a new label.

## Workflow

1. **Gather** the findings to publish (ask the user which ones, if ambiguous).
2. **Validate** each against the required fields above; drop and report any incomplete ones.
3. **Check for duplicates** — for each remaining finding, run:
   ```
   gh issue list --repo Ethilink/MVP-OKcamera --state open --search "<key terms from title/location>"
   ```
   Flag likely matches. Don't auto-skip them — surface the warning in the preview and let the user decide.
4. **Render one batch preview** — every candidate's title, full rendered body (see template below), labels, and any duplicate warning — as a single message. Create nothing yet.
5. **Wait for one approval** covering the whole batch. The user may ask to exclude or edit individual items before approving.
6. **Ensure labels exist**:
   ```
   gh label list --repo Ethilink/MVP-OKcamera
   ```
   For any of the six taxonomy labels that don't already exist, create it (definitions below) before referencing it on an issue.
7. **Create the approved issues**, one `gh issue create` per finding, with the body passed via heredoc so formatting/quoting survives intact:
   ```
   gh issue create --repo Ethilink/MVP-OKcamera \
     --title "<short-form summary>" \
     --label "<category-label>" --label "<severity-label>" \
     --body "$(cat <<'EOF'
   ...rendered body...
   EOF
   )"
   ```
8. **Report back** the created issue URLs, and note anything skipped (duplicates, excluded by the user, or dropped for missing fields).

## Label definitions (only needed the first time a label is missing)

- `architecture` — color `5319e7`, description "Structural or design concern"
- `tech-debt` — color `fbca04`, description "Promotion debt, missing abstraction, or deferred work"
- `tooling` — color `1d76db`, description "Lint, CI, or dev-process gap"
- `severity: high` — color `b60205`, description "High severity — should be addressed soon"
- `severity: medium` — color `d93f0b`, description "Medium severity"
- `severity: low` — color `0e8a16`, description "Low severity — nice to have"

Create with:
```
gh label create "<name>" --repo Ethilink/MVP-OKcamera --color <color> --description "<description>"
```

`bug` and `documentation` are GitHub's built-in defaults and should already exist — don't recreate them.

## Body template

Use [templates/issue-body.md](templates/issue-body.md) to render each finding's body, substituting `{{summary}}`, `{{location}}`, `{{why_it_matters}}`, and `{{suggested_fix_or_dash}}` (use `_none_` when no suggested fix was given). This mirrors the field order of `.github/ISSUE_TEMPLATE/finding.yml`.

## Notes

- GitHub Issue Forms only enforce `required` fields in the web UI — `gh issue create` and the API bypass that entirely. This skill's own validation (step 2) is the actual enforcement point for issues it creates.
- Issue Forms can't apply labels conditionally on a dropdown answer, so an issue filed by a human through the web form won't get `category`/`severity` labels automatically — a maintainer should label it manually to match their answers.
