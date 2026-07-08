import { useStatus } from "@/api/useStatus"

// Dev-only page (AC6): proves the polling loop end-to-end by dumping raw
// useStatus output, which updates live as the MSW dev sequence advances. Not
// shipped in production (dynamically imported behind import.meta.env.DEV).
export function StatusProbe() {
  const { status, error } = useStatus()
  return (
    <main className="mx-auto max-w-2xl p-6 font-mono text-sm">
      <h1 className="mb-2 text-lg font-semibold">useStatus probe · MSW dev</h1>
      <p className="mb-1">
        phase: <b>{status?.phase ?? "—"}</b> · health:{" "}
        <b>{status?.capture_health ?? "—"}</b> · model:{" "}
        {status?.model_version ?? "—"}
      </p>
      {error && (
        <p className="mb-2 text-destructive">poll error: {String(error)}</p>
      )}
      <pre className="overflow-x-auto rounded bg-muted p-3">
        {JSON.stringify(status, null, 2)}
      </pre>
    </main>
  )
}
