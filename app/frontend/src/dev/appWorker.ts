import { setupWorker } from "msw/browser"
import { devHandlers } from "./devHandlers"

// Dev-only MSW worker driving the real App under `npm run dev:msw` with the
// interactive dev backend (start/stop actually transition). Separate from
// test/browser.ts (T05's auto-cycling probe worker) so neither disturbs the other.
export const appWorker = setupWorker(...devHandlers)
