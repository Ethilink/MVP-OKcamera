import { Badge } from "@/components/ui/badge"

/**
 * `present` → green PRESENT; `missing` → destructive MISSING. "Missing", never
 * "lost": the camera observes the symptom (not on the table at Stop); whether
 * the instrument is misplaced or genuinely lost is unknowable from here
 * (vault-glossary distinction — DESIGN §3).
 */
export function CompletenessBadge({
  completeness,
}: {
  completeness: "present" | "missing"
}) {
  if (completeness === "missing") {
    return <Badge variant="destructive">MISSING</Badge>
  }
  return (
    <Badge
      variant="outline"
      className="border-transparent bg-emerald-600/10 text-emerald-700 dark:text-emerald-400"
    >
      PRESENT
    </Badge>
  )
}
