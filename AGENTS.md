# Agent instructions — ORC (OR Camera) MVP

## Project context lives outside this repo

This repo holds only the software (`model/`, `app/backend/`, `app/frontend/`). The actual project context — client brief, PRD, design docs, open issues, research — lives in the team's Obsidian vault, `company-brain`, under the `workspaces/surgical-sets` workspace. This repo does not duplicate that content; don't invent product decisions that belong there.

Read it before doing product/design work here (not needed for isolated code changes): `context/` (problem framing, glossary, scope), `mvp/docs/` (client-brief, prd, design, issues), `mvp/research/` (hardware specs, references).

## If `company-brain` isn't available

Check whether the `company-brain` MCP server is connected (e.g. via `/mcp`, or by trying to read from it). If it isn't, don't proceed without context or guess — offer to set it up with the user instead:

1. Ask the user for the absolute path to their local `company-brain` vault (machine-specific, e.g. `~/Documents/company-brain/root`).
2. Ask for their explicit permission before running anything — this changes their local Claude Code config.
3. Once permitted, register it at **local scope** so nothing machine-specific ever gets committed to this repo:

   ```bash
   claude mcp add company-brain --scope local -- npx -y @modelcontextprotocol/server-filesystem /absolute/path/to/company-brain/root
   ```

4. Verify with `/mcp`.

See the root `README.md` → "Agent setup" for the human-facing version of these steps.
