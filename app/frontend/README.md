# app/frontend

Control screen for the ORC demo: Start/Stop, live camera view, results dashboard
(Usage/Completeness per instrument). Talks only to `app/backend`'s HTTP API
(frozen in [`../docs/api-contract.md`](../docs/api-contract.md)), never to
`model/` directly.

Stack (frozen — DESIGN D2): **Vite + React + TS + Tailwind v4 + shadcn/ui**,
tested with Vitest + React Testing Library + MSW. Built against MSW fixtures
before the backend exists — see [`../docs/DESIGN.md`](../docs/DESIGN.md) and
tasks T05–T07.

## Dev

```bash
npm install
npm run dev        # Vite dev server
npm test           # Vitest (RTL + MSW)
npm run build      # tsc -b + vite build
```

Scaffolded in T01 (Vite react-ts, Tailwind v4 via `@tailwindcss/vite`, shadcn/ui
`base-nova`, Vitest+RTL+MSW). `@/*` aliases `src/*`.
