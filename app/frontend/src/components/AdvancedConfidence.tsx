import { useEffect, useRef, useState } from "react"
import { ChevronDownIcon } from "lucide-react"
import { ApiError, api } from "@/api/client"
import type { DetectorControl } from "@/api/types"
import { Button } from "@/components/ui/button"
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible"
import {
  Field,
  FieldDescription,
  FieldError,
  FieldLabel,
} from "@/components/ui/field"
import { Slider } from "@/components/ui/slider"
import { cn } from "@/lib/utils"

/**
 * The Advanced setup control (T11/F4/D6): a collapsed disclosure holding the
 * runtime **Detection confidence threshold**. It is an operational fallback, not
 * an identity-safety knob — the exact-catalog Start gate still owns safety.
 *
 * Behaviour the spec pins:
 * - min/max/step/current/default all read from `status.detector_control`;
 * - the slider updates its displayed value immediately, but only PATCHes after a
 *   250 ms debounce (coalescing a drag — and a burst of keyboard steps — into ONE
 *   reset once the value settles), so a drag isn't a storm of resets. The spec (F4)
 *   allows a debounce OR a pointer/key commit; we deliberately do debounce-only —
 *   see the inline note at the Slider;
 * - a successful PATCH resets enrolment backend-side (readiness returns to
 *   recognising) — that surfaces through the next poll, not here;
 * - a failed PATCH rolls the slider back to the last server-confirmed value and
 *   shows an inline error;
 * - while a PATCH is in flight `onPendingChange(true)` disables Track (F4);
 * - `Reset to default` is disabled at the default value.
 *
 * The control is rendered only in a non-recording setup view (the caller hides it
 * during recording and on the report), per D6.
 */
export function AdvancedConfidence({
  control,
  disabled = false,
  onPendingChange,
  onReset,
}: {
  control: DetectorControl
  /** External hold (for example, another setup mutation) — disables interaction. */
  disabled?: boolean
  /** Reports our own PATCH-in-flight up so the caller can disable Track. */
  onPendingChange?: (pending: boolean) => void
  /** Fires after a successful confidence change (which resets enrolment backend-
   *  side) so the caller can hold "Recognising" until a fresh poll (D2). */
  onReset?: () => void | Promise<unknown>
}) {
  const [open, setOpen] = useState(false)
  const [value, setValue] = useState(control.confidence)
  const [pending, setPending] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const timer = useRef<number | undefined>(undefined)
  // The last SERVER-confirmed confidence. Updated only on a successful PATCH,
  // not from every polled prop render. It is the basis for the no-op check and
  // failure rollback, so a just-confirmed value (which polling only echoes a
  // beat later) never snaps the slider back (Fable M1).
  const confirmed = useRef(control.confidence)

  // The slider value is purely LOCAL: seeded from the server value at mount and
  // thereafter changed only by the user and by PATCH responses. It is deliberately
  // NOT re-adopted from `control.confidence` on re-render — this control is the
  // only thing that changes the confidence, so re-adopting would just let a stale
  // in-flight poll (still echoing the pre-PATCH value) snap the slider back for a
  // beat (Fable M1 / Codex). On a real external change the component remounts
  // (setup ⇄ recording) and re-seeds. `confirmed` tracks the last acknowledged
  // value for the no-op check and the failure rollback.

  useEffect(() => {
    onPendingChange?.(pending)
  }, [pending, onPendingChange])

  useEffect(() => () => window.clearTimeout(timer.current), [])

  async function commit(next: number) {
    window.clearTimeout(timer.current)
    // A no-op value equals the last CONFIRMED one → don't PATCH (avoids a needless
    // enrolment reset). Restore the DISPLAY to the confirmed value too, so a Reset
    // (or a drag-back) that lands on the confirmed value during a pending debounce
    // doesn't leave the slider stranded at the un-submitted value (Codex).
    if (next === confirmed.current) {
      setValue(confirmed.current)
      setError(null)
      return
    }
    setPending(true)
    setError(null)
    try {
      const updated = await api.setDetectionConfidence(next)
      confirmed.current = updated.confidence // server-confirmed source of truth
      setValue(updated.confidence)
      await onReset?.() // keep pending/Track held until mutation-fresh status
    } catch (err) {
      setValue(confirmed.current) // roll back to the last confirmed value
      setError(
        err instanceof ApiError ? err.detail : "Couldn't update confidence — try again.",
      )
    } finally {
      setPending(false)
    }
  }

  function onSlide(next: number) {
    setValue(next) // immediate visual feedback
    setError(null)
    window.clearTimeout(timer.current)
    timer.current = window.setTimeout(() => commit(next), 250) // debounced PATCH
  }

  const interactionDisabled = disabled || pending
  const atDefault = value === control.default_confidence

  return (
    <Collapsible open={open} onOpenChange={setOpen} className="w-full">
      <CollapsibleTrigger className="flex w-full items-center justify-center gap-1.5 rounded-md py-1 text-xs font-medium text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring">
        Advanced
        <ChevronDownIcon
          className={cn("size-3.5 transition-transform", open && "rotate-180")}
        />
      </CollapsibleTrigger>

      <CollapsibleContent className="mt-2">
        <div className="flex flex-col gap-2 rounded-xl bg-muted/40 p-3 ring-1 ring-border">
          <Field>
            {/* FieldLabel provides the design-system composition; Base UI's
                generated range input receives its name through getAriaLabel. */}
            <FieldLabel className="flex w-full flex-col items-stretch gap-2">
              <span className="flex items-baseline justify-between">
                <span className="text-xs font-medium text-foreground">
                  Detection confidence threshold
                </span>
                <span className="text-xs tabular-nums text-muted-foreground">
                  {value.toFixed(2)}
                </span>
              </span>
              {/* Debounced-only on purpose: each committed value triggers a
                  backend tracker RESET (re-enrolment), so we coalesce a drag —
                  and a burst of keyboard steps — into ONE PATCH 250 ms after the
                  value settles, rather than one reset per intermediate value. */}
              <Slider
                value={value}
                getAriaLabel={() => "Detection confidence threshold"}
                min={control.minimum}
                max={control.maximum}
                step={control.step}
                disabled={interactionDisabled}
                onValueChange={(v) => onSlide(firstValue(v))}
              />
            </FieldLabel>

            <FieldDescription className="text-[0.7rem] leading-snug">
              Higher hides weak detections; lower finds more objects but may include
              clutter.
            </FieldDescription>

            {error && <FieldError className="text-[0.7rem]">{error}</FieldError>}
          </Field>

          <Button
            type="button"
            variant="ghost"
            size="xs"
            onClick={() => commit(control.default_confidence)}
            disabled={atDefault || interactionDisabled}
            className="self-start"
          >
            Reset to default
          </Button>
        </div>
      </CollapsibleContent>
    </Collapsible>
  )
}

/** The base-ui Slider reports `number | number[]`; we drive a single thumb. */
function firstValue(v: number | readonly number[]): number {
  return Array.isArray(v) ? v[0] : (v as number)
}
