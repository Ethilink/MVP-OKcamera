import { setupWorker } from "msw/browser"
import { handlers } from "./handlers"

// Dev-only MSW worker (service worker at public/mockServiceWorker.js). Started
// from main.tsx only when VITE_MSW=true, so T06/T07 (and this scaffold) can run
// `npm run dev:msw` against fixtures before the backend exists.
export const worker = setupWorker(...handlers)
