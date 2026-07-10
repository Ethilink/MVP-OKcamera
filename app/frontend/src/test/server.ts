import { setupServer } from "msw/node"

// Node (test) MSW server. T05 registers the scripted `handlers` here; the T01
// smoke test drives it ad hoc via `server.use(...)`.
export const server = setupServer()
