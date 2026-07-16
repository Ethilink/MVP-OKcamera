import { useEffect, useState } from "react"
import { api } from "@/api/client"
import { Card } from "@/components/ui/card"

/**
 * The live MJPEG feed: a bare `<img src={api.streamUrl}>` in a fixed 16:9 card.
 * Per D16 it degrades gracefully — MSW cannot intercept an `<img>` MJPEG load,
 * so under RTL/MSW and pre-backend `npm run dev` the `onError` fallback panel is
 * what renders; the real overlay only appears against `--fake`/real backend.
 * The same fallback covers a genuinely dead camera in production.
 */
export function VideoFeed() {
  const [failed, setFailed] = useState(false)
  const [retryAttempt, setRetryAttempt] = useState(0)

  useEffect(() => {
    if (!failed) return
    const retryId = window.setTimeout(() => {
      setRetryAttempt((attempt) => attempt + 1)
      setFailed(false)
    }, 1_000)
    return () => window.clearTimeout(retryId)
  }, [failed])

  const retrySeparator = api.streamUrl.includes("?") ? "&" : "?"
  const streamUrl =
    retryAttempt === 0
      ? api.streamUrl
      : `${api.streamUrl}${retrySeparator}retry=${retryAttempt}`

  return (
    <Card className="relative aspect-video w-full overflow-hidden bg-[oklch(0.19_0.02_220)] p-0 ring-1 ring-foreground/10">
      {failed ? (
        <div
          role="img"
          aria-label="no stream (dev mode)"
          className="flex h-full w-full flex-col items-center justify-center gap-3.5 bg-[radial-gradient(120%_120%_at_50%_0%,oklch(0.26_0.03_220)_0%,oklch(0.18_0.02_220)_60%)]"
        >
          <CameraGlyph className="size-9 text-white/35" />
          <div className="text-center">
            <p className="text-sm font-medium text-white/75">Camera preview</p>
            <p className="mt-0.5 text-xs text-white/45">
              Waiting for the OR camera feed
            </p>
          </div>
        </div>
      ) : (
        <img
          src={streamUrl}
          alt="live camera feed"
          className="h-full w-full object-contain"
          onError={() => setFailed(true)}
        />
      )}
    </Card>
  )
}

function CameraGlyph({ className }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 24 24"
      className={className}
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M3 8.5A2.5 2.5 0 0 1 5.5 6h1.2a1.5 1.5 0 0 0 1.25-.67l.6-.9A1.5 1.5 0 0 1 10.8 3.7h2.4a1.5 1.5 0 0 1 1.25.73l.6.9A1.5 1.5 0 0 0 16.3 6h1.2A2.5 2.5 0 0 1 20 8.5v8A2.5 2.5 0 0 1 17.5 19h-11A2.5 2.5 0 0 1 3 16.5Z" />
      <circle cx="12" cy="12.5" r="3.2" />
    </svg>
  )
}
