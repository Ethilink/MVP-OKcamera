# ORC — OR Camera (MVP)

A computer-vision demo that tracks surgical instruments in the sterile field, live, during a simulated procedure.

Built for a demo at **UZ Leuven** (2026-07-20) with the hospital's Central Sterile Services Department (CSSD): instruments are frequently lost or misplaced in the OR, and CSSD only discovers it downstream, too late to act. ORC shows what's possible: detect every instrument on the table, track it as it's picked up and returned, and report usage and completeness once the procedure ends.

## Demo scenario

1. Instruments are laid out on the table, unobstructed.
2. **Start** is pressed — recording and live tracking begin.
3. Any instrument can be picked up, carried out of frame, and brought back, in any order, any number of times.
4. **Stop** is pressed — recording ends.
5. A dashboard shows, per instrument: **Usage** (when it was off the table) and **Completeness** (present or missing).

This is a demo/PoC, not a product: a controlled, in-vitro setup with our own instruments, not the client's.

## Repo structure

```
MVP-OKcamera/
├── model/              # offline: detection/tracking, produces the artifact app/backend consumes
├── data-collection/    # capture tool: stream the camera, save stills + recordings as training material
└── app/
    ├── backend/        # live runtime: stream ingestion, phase state, reports — talks to model/ only via its artifact
    └── frontend/        # control screen (Start/Stop), live view, results dashboard — talks only to app/backend's API
```

- `app/backend` depends on `model`'s artifact only — never on `model`'s training code.
- `app/frontend` only talks to `app/backend`'s API, never to `model/` directly.
- `model/` has no dependency on `app/` at all — training runs and finishes before the app ever touches the artifact.

See each folder's own `README.md` for internals. `model.load_tracker()` provides
the real RF-DETR, Deep OC-SORT, and appearance-based session linker consumed by
`app/backend` in camera mode.

## Agent setup

Nothing machine-specific is committed — no shared MCP config, no pinned skills. Project context lives in the `company-brain` Obsidian vault; see [`docs/obsidian-vault-setup.md`](docs/obsidian-vault-setup.md) to connect it. Enable whatever plugins/skills you personally need (e.g. `ethi-code`) as you go.

See [`AGENTS.md`](AGENTS.md) (symlinked `CLAUDE.md`) for the agent-facing version.
