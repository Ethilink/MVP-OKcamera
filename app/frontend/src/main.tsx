import { StrictMode } from "react"
import { createRoot } from "react-dom/client"
import "./index.css"
import App from "./App.tsx"

// In `npm run dev:msw` (VITE_MSW=true) start an MSW worker and mount the real
// App against the interactive dev backend (drive setup → record → report with no
// real backend). `?probe` still reaches T05's raw useStatus probe. Everything
// here is dynamically imported inside a DEV-only branch, so production prunes it.
async function bootstrap() {
  const root = createRoot(document.getElementById("root")!)

  if (import.meta.env.DEV && import.meta.env.VITE_MSW === "true") {
    if (new URLSearchParams(window.location.search).has("probe")) {
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

    const { appWorker } = await import("./dev/appWorker")
    await appWorker.start({ onUnhandledRequest: "bypass" })
    root.render(
      <StrictMode>
        <App />
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
