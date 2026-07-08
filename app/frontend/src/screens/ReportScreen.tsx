import { Button } from "@/components/ui/button"

/**
 * T06 placeholder for the frozen T07 seam (BOARD "T07 seam"). It renders only the
 * "New recording" button wired to `props.onNewRecording` so App routing compiles
 * and the run-1 → run-2 flow is exercisable before T07 lands the real report.
 * T07 REPLACES this file; the `{ onNewRecording }` prop contract is identical.
 */
export function ReportScreen({
  onNewRecording,
}: {
  onNewRecording: () => void
}) {
  return (
    <main className="mx-auto flex min-h-svh max-w-5xl flex-col items-center justify-center gap-4 p-6">
      <p className="text-muted-foreground">Report (T07 renders here).</p>
      <Button onClick={onNewRecording}>New recording</Button>
    </main>
  )
}
