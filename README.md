# ORC — OR Camera (MVP)

A computer-vision demo that tracks surgical instruments in the sterile field, live, during a simulated procedure.

Built for a demo at **UZ Leuven** (2026-07-20) with the hospital's Central Sterile Services Department (CSSD): instruments are frequently lost or misplaced in the OR, and CSSD only discovers it downstream, too late to act. ORC shows what's possible: detect every instrument on the table, track it as it's picked up and returned, and report usage and completeness once the procedure ends.

## Demo scenario

1. Instruments are laid out on the table, unobstructed.
2. **Start** is pressed — recording and live tracking begin.
3. Any instrument can be picked up, carried out of frame, and brought back, in any order, any number of times.
4. **Stop** is pressed — recording ends.
5. A dashboard shows, per instrument: **Usage** (when it was off the table) and **Completeness** (present or lost).

This is a demo/PoC, not a product: a controlled, in-vitro setup with our own instruments, not the client's.

## Repo structure

```
MVP-OKcamera/
├── model/              # offline: detection/tracking, produces the artifact app/backend consumes
└── app/
    ├── backend/        # live runtime: stream ingestion, phase state, reports — talks to model/ only via its artifact
    └── frontend/        # control screen (Start/Stop), live view, results dashboard — talks only to app/backend's API
```

- `app/backend` depends on `model`'s artifact only — never on `model`'s training code.
- `app/frontend` only talks to `app/backend`'s API, never to `model/` directly.
- `model/` has no dependency on `app/` at all — training runs and finishes before the app ever touches the artifact.

Each folder is a placeholder for now (see its own `README.md`) — internals (language, framework, packaging) aren't scaffolded yet.

## Agent setup

This repo doesn't ship a shared MCP config or any pinned Claude Code skills — nothing machine-specific is committed.

If you want the team's `company-brain` Obsidian vault available to Claude Code (client brief, PRD, design docs), register it yourself at **local scope** so it's stored in your own `~/.claude.json`, never in this repo:

```bash
claude mcp add company-brain --scope local -- npx -y @modelcontextprotocol/server-filesystem /absolute/path/to/your/company-brain/root
```

Verify with `/mcp` inside Claude Code. No plugins or skills are enabled by default — enable/pull in whatever you personally need (e.g. the org's `ethi-code` plugin) as you go.
