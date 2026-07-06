# Agent instructions — ORC (OR Camera) MVP

## Project context lives outside this repo

This repo holds only the software (`model/`, `app/backend/`, `app/frontend/`). The actual project context — client brief, PRD, design docs, open issues, research — lives in the team's Obsidian vault, `company-brain`, under the `workspaces/surgical-sets` workspace. This repo does not duplicate that content; don't invent product decisions that belong there.

Read it before doing product/design work here (not needed for isolated code changes): `context/` (problem framing, glossary, scope), `mvp/docs/` (client-brief, prd, design, issues), `mvp/research/` (hardware specs, references).

## If `company-brain` isn't available

Check whether the `company-brain` MCP server is connected (e.g. via `/mcp`, or by trying to read from it). If it isn't, don't proceed without context or guess — offer to set it up with the user instead: ask for their local vault path, get explicit permission before running anything (it changes their local Claude Code config), then follow the exact setup command in the root [`README.md`](README.md) → "Agent setup".
