import { StrictMode } from "react"
import { createRoot } from "react-dom/client"
import "./index.css"
import App from "./App.tsx"

// In `npm run dev:msw` (VITE_MSW=true) start the MSW worker and mount the dev
// StatusProbe (AC6). Both are dynamically imported inside a DEV-only branch, so
// the production build prunes them entirely.
async function bootstrap() {
  const root = createRoot(document.getElementById("root")!)

  if (import.meta.env.DEV && import.meta.env.VITE_MSW === "true") {
    const [{ worker }, { StatusProbe }] = await Promise.all([
      import("./test/browser"),
      import("./dev/StatusProbe"),
    ])
    await worker.start({ onUnhandledRequest: "bypass" })
    root.render(
      <StrictMode>
        <StatusProbe />
      </StrictMode>,
    )
    return
  }

  root.render(
    <StrictMode>
      <App />
    </StrictMode>,
  )
}

void bootstrap()
