import { useState } from "react"
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

  return (
    <Card className="aspect-video w-full overflow-hidden bg-black p-0">
      {failed ? (
        <div
          role="img"
          aria-label="no stream (dev mode)"
          className="flex h-full w-full items-center justify-center bg-muted text-sm font-medium text-muted-foreground"
        >
          no stream (dev mode)
        </div>
      ) : (
        <img
          src={api.streamUrl}
          alt="live camera feed"
          className="h-full w-full object-contain"
          onError={() => setFailed(true)}
        />
      )}
    </Card>
  )
}
