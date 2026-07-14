import { useRef, type PointerEvent, type ReactNode } from "react"
import { motion, useReducedMotion } from "motion/react"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"

/**
 * Steers the specular highlight toward the pointer — the "alive" light shared by
 * Track and Stop. Writes a CSS var only (no layout, no React render); gated on
 * `active` (armed) + reduced-motion by the caller passing `active=false`.
 */
function useSpecularPointer(active: boolean) {
  const ref = useRef<HTMLDivElement>(null)
  const reduceMotion = useReducedMotion()
  function onPointerMove(e: PointerEvent<HTMLDivElement>) {
    if (!active || reduceMotion) return
    const el = ref.current
    if (!el) return
    const r = el.getBoundingClientRect()
    el.style.setProperty("--track-mx", `${((e.clientX - r.left) / r.width) * 100}%`)
    el.style.setProperty("--track-my", `${((e.clientY - r.top) / r.height) * 100}%`)
  }
  function onPointerLeave() {
    const el = ref.current
    if (!el) return
    el.style.removeProperty("--track-mx")
    el.style.removeProperty("--track-my")
  }
  return { ref, onPointerMove, onPointerLeave }
}

/**
 * The one primary action on the setup screen: **Track**. An Apple-style
 * liquid-glass control whose surface bends its *own* light — an SVG displacement
 * lens ripples the highlight like gel, a specular glint tracks the pointer, and
 * the press is spring-loaded. (On the white panel there is nothing behind to
 * refract, so the glass lenses its own highlight instead; it reads on any
 * backdrop and in every engine.) Outside the pill it stays just glass; the one
 * flourish is a **very faint iridescent glint drifting inside** it while armed —
 * the spectral sheen you catch on wet glass in the rain. It only shimmers while
 * ready, so even that whisper is the readiness cue, not decoration. The caption
 * below is anchored so the button never shifts when the reason text appears.
 * Disabled Track states why.
 */
export function TrackButton({
  enabled,
  reason,
  pending,
  onTrack,
  layoutId,
}: {
  enabled: boolean
  reason?: string
  pending?: boolean
  onTrack: () => void
  /** Shared-layout id: hands the pill off to the red Stop on Start (LiveScreen). */
  layoutId?: string
}) {
  const armed = enabled && !pending
  const reduceMotion = useReducedMotion()
  const { ref, onPointerMove, onPointerLeave } = useSpecularPointer(armed)

  const state = armed ? "is-armed" : pending ? "is-pending" : "is-idle"

  return (
    <div className="flex flex-col items-center">
      <TrackLensFilter />
      <motion.div
        ref={ref}
        layoutId={layoutId}
        className={cn("track-cta", state)}
        onPointerMove={onPointerMove}
        onPointerLeave={onPointerLeave}
        whileTap={armed && !reduceMotion ? { scale: 0.965 } : undefined}
        transition={{ type: "spring", stiffness: 620, damping: 30, mass: 0.7 }}
      >
        <Button
          size="lg"
          disabled={!armed}
          onClick={onTrack}
          className="track-button relative h-14 gap-2.5 rounded-full px-9 text-base font-semibold"
        >
          <span className="track-button__lens" aria-hidden="true" />
          {/* The faint iridescent glint drifting inside the glass (armed only). */}
          <span className="track-button__prism" aria-hidden="true" />
          <span className="track-button__specular" aria-hidden="true" />
          <TargetIcon className="relative z-10 size-[1.15rem]" />
          <span className="relative z-10">{pending ? "Starting…" : "Track"}</span>
        </Button>
      </motion.div>

      {/* Anchored caption: the slot always occupies a line so arming/disabling
          never nudges the button. Empty (but reserved) once ready. */}
      <p className="mt-2.5 min-h-5 text-sm text-muted-foreground">
        {enabled ? "" : reason}
      </p>
    </div>
  )
}

/**
 * A live liquid-glass pill — the SAME glass as an armed Track (clear brightened
 * pane, white specular, warped lens) but WITHOUT the rainbow glint (that drift
 * is Track's readiness cue, not general chrome). Always interactive. `tone`
 * only recolours the ink: `stop` → red, `default` → teal. Stop and New recording
 * are both built from this, so the three primary actions read as one family.
 */
function GlassActionButton({
  label,
  pendingLabel,
  pending,
  onClick,
  icon,
  tone = "default",
  layoutId,
}: {
  label: string
  pendingLabel?: string
  pending?: boolean
  onClick: () => void
  icon: ReactNode
  tone?: "default" | "stop"
  layoutId?: string
}) {
  const reduceMotion = useReducedMotion()
  // Always live, so the specular always tracks the pointer.
  const { ref, onPointerMove, onPointerLeave } = useSpecularPointer(true)

  return (
    <div className="flex flex-col items-center">
      <TrackLensFilter />
      <motion.div
        ref={ref}
        layoutId={layoutId}
        className={cn("track-cta is-armed", tone === "stop" && "track-cta--stop")}
        onPointerMove={onPointerMove}
        onPointerLeave={onPointerLeave}
        whileTap={!reduceMotion ? { scale: 0.965 } : undefined}
        transition={{ type: "spring", stiffness: 620, damping: 30, mass: 0.7 }}
      >
        <Button
          size="lg"
          disabled={pending}
          onClick={onClick}
          className="track-button relative h-14 gap-2.5 rounded-full px-9 text-base font-semibold"
        >
          <span className="track-button__lens" aria-hidden="true" />
          <span className="track-button__specular" aria-hidden="true" />
          {icon}
          <span className="relative z-10">{pending ? pendingLabel ?? label : label}</span>
        </Button>
      </motion.div>
    </div>
  )
}

/**
 * **Stop** on the recording screen — red-ink glass. Shares `layoutId` with Track,
 * so on Start the teal Track slides down and recolours into this red Stop.
 */
export function StopButton({
  pending,
  onStop,
  layoutId,
}: {
  pending?: boolean
  onStop: () => void
  layoutId?: string
}) {
  return (
    <GlassActionButton
      tone="stop"
      label="Stop"
      pendingLabel="Stopping…"
      pending={pending}
      onClick={onStop}
      layoutId={layoutId}
      icon={<StopIcon className="relative z-10 size-[1.05rem]" />}
    />
  )
}

/** **New recording** on the report — the same glass, teal ink. */
export function NewRecordingButton({ onClick }: { onClick: () => void }) {
  return (
    <GlassActionButton
      label="New recording"
      onClick={onClick}
      icon={<PlusIcon className="relative z-10 size-[1.15rem]" />}
    />
  )
}

function StopIcon({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" className={className} fill="currentColor" aria-hidden="true">
      <rect x="6" y="6" width="12" height="12" rx="2.5" />
    </svg>
  )
}

function PlusIcon({ className }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 24 24"
      className={className}
      fill="none"
      stroke="currentColor"
      strokeWidth="1.75"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M12 5v14M5 12h14" />
    </svg>
  )
}

/**
 * The displacement filter that gives the glass its liquid warp. Referenced by
 * `.track-button__lens` via `filter: url(#track-lens)`, it ripples that layer's
 * sheen and rim glints — unlike `backdrop-filter` refraction (Chromium-only,
 * and invisible over a flat white fill), so the effect holds everywhere. Only
 * one Track is ever on screen, so a single definition is enough.
 */
function TrackLensFilter() {
  return (
    <svg
      aria-hidden="true"
      focusable="false"
      width="0"
      height="0"
      style={{ position: "absolute", width: 0, height: 0, pointerEvents: "none" }}
    >
      <defs>
        <filter
          id="track-lens"
          x="-25%"
          y="-25%"
          width="150%"
          height="150%"
          colorInterpolationFilters="sRGB"
        >
          <feTurbulence
            type="fractalNoise"
            baseFrequency="0.013 0.018"
            numOctaves="2"
            seed="7"
            result="noise"
          />
          <feGaussianBlur in="noise" stdDeviation="1.3" result="soft" />
          <feDisplacementMap
            in="SourceGraphic"
            in2="soft"
            scale="22"
            xChannelSelector="R"
            yChannelSelector="G"
          />
        </filter>
      </defs>
    </svg>
  )
}

function TargetIcon({ className }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 24 24"
      className={className}
      fill="none"
      stroke="currentColor"
      strokeWidth="1.75"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <circle cx="12" cy="12" r="8" />
      <circle cx="12" cy="12" r="3.25" />
      <path d="M12 2v3M12 19v3M2 12h3M19 12h3" />
    </svg>
  )
}
